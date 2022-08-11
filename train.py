from locale import textdomain
from datasets import load_dataset, load_from_disk
from numpy import broadcast
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler, DataLoader
from torch import nn
from torch.nn.functional import cross_entropy
import torch
from tqdm import tqdm
from transformers import AdamW, GPT2Tokenizer
from x_transformers.x_transformers import TokenEmbedding, AbsolutePositionalEmbedding
import wandb

from block_recurrent_transformer import BlockRecurrentBlock, long_sequence_splitter
from block_recurrent_transformer.transformer import VanillaTransformerBlock

# apex magic
from apex import amp

class WikiDataset:
    def __init__(self, data, max_char=100):
        self.data = data
        self.max_char = max_char
    
    def __getitem__(self, i: int):
        record = self.data[i]
        title = record['title']
        text = record['text']
        return f'{title}\n\n{text}'[:self.max_char] + ' [PAD]'
    
    def __len__(self):
        return len(self.data)


class BlockRecurrentDecoder(nn.Module):
    """As simple as I can make the model.
    """

    def __init__(self, config, num_tokens, dim, d_latent):
        super().__init__()
        self.embed = TokenEmbedding(dim, num_tokens)
        self.pos_embed = AbsolutePositionalEmbedding(dim, config.window_len + 10)
        self.layers = nn.ModuleList([BlockRecurrentBlock(config, dim, dim) \
                                        for _ in range(config.num_layers)])
        self.to_logits = nn.Linear(dim, num_tokens)
        self.to_gaussian = nn.Linear(d_latent, dim)
        self.norm = nn.LayerNorm(dim)
        self.norm_state = nn.LayerNorm(dim)
        self.dim = dim
    
    def forward(self, x, state):
        x = self.embed(x) + self.pos_embed(x)
        state = self.to_gaussian(state)
        for layer in self.layers:
            x, state = layer(x, state)
        x = self.to_logits(self.norm(x))
        return x, state


class LogLinearDecoder(nn.Module):
    """As simple as I can make the model.
    """

    def __init__(self, config, num_tokens, dim, d_latent):
        super().__init__()
        self.h0 = nn.Parameter(torch.zeros(1, config.state_len, 2*d_latent))
        self.to_logits = nn.Linear(d_latent, num_tokens)
        #self.to_logits.bias.data[:33] = -10000
        #self.to_logits.bias.data[34:39] = -10000
        #self.to_logits.bias.data[40:] = -10000
    
    def forward(self, x, state):
        return self.to_logits(state), state

class BiTransformer(nn.Module):
    """As simple as I can make the model.
    """

    def __init__(self, config, num_tokens, dim, d_latent):
        super().__init__()
        self.embed = TokenEmbedding(dim, num_tokens)
        self.pos_embed = AbsolutePositionalEmbedding(dim, config.window_len + 10)
        self.layers = nn.ModuleList([VanillaTransformerBlock(config, dim) \
                                        for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.to_gaussian = nn.Linear(dim, 2*d_latent)
        self.decode_latent = nn.Linear(d_latent, dim)
    
    def forward(self, x, state=None):
        x = self.embed(x) + self.pos_embed(x)
        if state is not None:
            latent = self.decode_latent(state)
            #latent = x.new_zeros((x.size(0), 1, x.size(-1)))
        else:
            latent = x.new_zeros((x.size(0), 1, x.size(-1)))
        x = torch.cat([x[:, :config.state_len, :], latent, x[:, config.state_len:, :]], dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.to_gaussian(self.norm(x))
        return x

class UpstreamLSTM(nn.Module):
    """As simple as I can make the model.
    """

    def __init__(self, dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(dim, dim, num_layers=num_layers, batch_first=True)
        self.h0 = nn.Parameter(torch.zeros(1, config.state_len, 2*dim))
        self.to_gaussian = nn.Linear(dim, 2*dim)
    
    def forward(self, x, hidden):
        if x.size(1) > 0:
            out, hidden = self.lstm(x, hidden)
            out = self.to_gaussian(out)
            return out, hidden
        return x.repeat(1, 1, 2), hidden

class UpstreamLSTMDiscrete(nn.Module):
    """As simple as I can make the model.
    """

    def __init__(self, num_tokens, dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(dim, dim, num_layers=num_layers, batch_first=True)
        self.to_vocab = nn.Linear(dim, num_tokens)
    
    def forward(self, x, hidden):
        if x.size(1) > 0:
            out, hidden = self.lstm(x, hidden)
            out = self.to_vocab(out)
            return out, hidden
        return x.repeat(1, 1, 2), hidden

class UpstreamTransformer(nn.Module):
    """As simple as I can make the model.
    """

    def __init__(self, config, num_tokens, dim):
        super().__init__()
        self.layers = nn.ModuleList([VanillaTransformerBlock(config, dim) \
                                        for _ in range(config.num_layers)])
        self.pos_embed = AbsolutePositionalEmbedding(dim, 999)
        self.norm = nn.LayerNorm(dim)
        self.to_gaussian = nn.Linear(dim, 2*dim)
        self.state_len = config.state_len
    
    def forward(self, x, hidden):
        x = torch.cat([hidden, x], dim=1)
        out = x + self.pos_embed(x[:,:,-1])
        for layer in self.layers:
            out = layer(out)
        return self.to_gaussian(self.norm(out[:, -self.state_len:, :])), x


def setup_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

from itertools import chain
def train(data, tokenizer, config):
    global_step = 0
    upstream_model = UpstreamLSTM(config.d_latent, num_layers=config.lstm_layers)
    #upstream_model = UpstreamLSTMDiscrete(len(tokenizer), config.d_latent, num_layers=config.lstm_layers)
    #upstream_model = UpstreamTransformer(config, len(tokenizer), config.d_latent)
    upstream_model.to(device)
    model = BlockRecurrentDecoder(config, len(tokenizer), config.d_model, d_latent=config.d_latent)
    #model = LogLinearDecoder(config, len(tokenizer), config.d_model, d_latent=config.d_latent)
    model.to(device)
    posterior_model = BiTransformer(config, len(tokenizer), config.d_model, d_latent=config.d_latent)
    posterior_model.to(device)
    opt = AdamW([{'params': model.parameters(), 'lr': config.lr, 'max_lr': config.lr},
                 {'params': posterior_model.parameters(), 'lr': config.lr, 'max_lr': config.lr}],
                betas=(config.adam_beta1, config.adam_beta2), weight_decay=0.0)
    opt1 = AdamW([{'params': upstream_model.parameters(), 'lr': 0.001, 'max_lr': 0.001}], weight_decay=0.0)
    
    # cyclical annealing
    import numpy as np
    def frange_cycle_linear(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
        L = np.ones(n_iter) * stop
        period = n_iter/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i+c*period) < n_iter):
                L[int(i+c*period)] = v
                v += step
                i += 1
        return L 
    annealing_sched = frange_cycle_linear(config.max_updates, start=0.0, stop=1.0, n_cycle=10)

    # apex magic
    models, opts = amp.initialize(
        [model, posterior_model, upstream_model],
        [opt, opt1],
        opt_level='O0',
        )
    model = models[0]
    posterior_model = models[1]
    upstream_model = models[2]

    opt = opts[0]
    opt1 = opts[1]
    
    train_data = WikiDataset(data['train'])
    data_loader = DataLoader(train_data,  batch_size = config.batch_size, sampler = RandomSampler(train_data), pin_memory=True)

    train_loss = 0
    recon_loss = 0
    
    #generate(model, upstream_model, tokenizer, config)
    for batch_id, raw_batch in enumerate(tqdm(data_loader), 1):
        opt.zero_grad()
        opt1.zero_grad()
        article_batch = tokenizer(raw_batch, return_tensors='pt', padding=True)['input_ids']
        
        article_batch[:article_batch.size(0)//2, ::2] = 33
        article_batch[:article_batch.size(0)//2, 1::2] = 39
        article_batch[article_batch.size(0)//2:, ::2] = 39
        article_batch[article_batch.size(0)//2:, 1::2] = 33
        
        #article_batch[:, -1] = tokenizer.eos_token_id
        # manually specify h0
        prev_state = [upstream_model.h0.repeat(article_batch.size(0), 1, 1)]
        hidden = (prev_state[0].new_ones((config.lstm_layers, article_batch.size(0), config.d_latent, )),
                  prev_state[0].new_ones((config.lstm_layers, article_batch.size(0), config.d_latent, )))
        sampled_z = None
        #hidden = prev_state[0].new_ones((article_batch.size(0), 0, config.d_latent))
        batch_rec_loss = 0.
        batch_prior_loss = 0.
        if batch_id <= config.warmup:
            for param_group in opt.param_groups:
                param_group['lr'] = param_group['max_lr'] * batch_id / config.warmup
            for param_group in opt1.param_groups:
                param_group['lr'] = param_group['max_lr'] * batch_id / config.warmup
        for i, text in enumerate(long_sequence_splitter(article_batch, config.window_len)):
            # add eos token so the low-level model knows when to stop
            #bos_text_eos = torch.cat([text.new_ones(text.size(0), 1)*tokenizer.bos_token_id, text, text.new_ones(text.size(0), 1)*tokenizer.eos_token_id], dim=-1).cuda()
            bos_text = torch.cat([text.new_ones(text.size(0), 1)*tokenizer.bos_token_id, text], dim=-1).cuda()
            inputs = bos_text[:, :-1]
            targets = bos_text[:, 1:]
            inputs_mask = (inputs != tokenizer.pad_token_id) & (targets != tokenizer.pad_token_id)
            # add bos token so our BERT can use it to summarize the block
            #bos_text = torch.cat([text.new_ones(text.size(0), config.state_len)*tokenizer.bos_token_id, text], dim=-1)
            # run q(z|x)
            state_posterior = posterior_model(bos_text, state=sampled_z)[:, :config.state_len]
            qz_mean = state_posterior[:, :, :config.d_latent]
            qz_logvar = state_posterior[:, :, config.d_latent:]
            # use z ~ q(z|x) to generate p(x|z)
            eps = qz_mean.new(qz_mean.shape).normal_(0, 1)
            torch.nn.init.trunc_normal_(eps)
            sampled_z = qz_mean + (0.5*qz_logvar).exp()*eps
            preds, _ = model(inputs, sampled_z)
            state, hidden = upstream_model(sampled_z, hidden)
            # reconstruction loss
            rec_loss = cross_entropy(preds[inputs_mask], targets[inputs_mask])
            # prior matching loss KL(qz||pz)
            if config.state_len > 0:
                pz_mean = prev_state[-1][:, :, :config.d_latent]
                pz_logvar = prev_state[-1][:, :, config.d_latent:]
                prior_loss = 0.5 * ((pz_logvar-qz_logvar) + (qz_logvar.exp()+(qz_mean - pz_mean)**2)/pz_logvar.exp() - 1).sum(-1).mean()
                #prior_reg = 0.001*(0.5 * ((-qz_logvar) + (qz_logvar.exp()+(qz_mean)**2) - 1).sum(-1).mean())
                prior_reg = 0.001*(0.5 * ((-qz_logvar) + (qz_logvar.exp()+(qz_mean)**2) - 1).sum(-1).mean())
                #prior_loss = ((qz_mean.detach() - pz_mean)**2).sum(-1).mean()
                #prior_loss = ((pz_logvar-qz_logvar.detach())**2 + (qz_mean.detach() - pz_mean)**2).sum(-1).mean()
            else:
                prior_loss = 0.
            prev_state.append(state)
            #if rec_loss > 1:
            #    batch_rec_loss += rec_loss #+prior_reg
            #else:
            batch_rec_loss += rec_loss
            batch_prior_loss += prior_loss #+prior_reg
            #loss.backward(retain_graph=True)
            
            train_loss += (rec_loss+prior_loss).item()
            recon_loss += rec_loss.item()
        #with amp.scale_loss(batch_loss, [opt, opt1]) as scaled_loss:
        #    scaled_loss.backward()
        #if type(batch_prior_loss) is torch.Tensor:
        #    (annealing_sched[global_step]*batch_prior_loss).backward(retain_graph=True)
        #torch.nn.utils.clip_grad_norm_(chain(amp.master_params(opt),
        #                                    amp.master_params(opt1)), config.gclip)
        (0.2*annealing_sched[global_step]*batch_prior_loss + batch_rec_loss).backward()
        torch.nn.utils.clip_grad_norm_(chain(amp.master_params(opt),
                                            amp.master_params(opt1)), config.gclip)
        opt.step()
        opt1.step()
        global_step += 1
        if batch_id % 100 == 0:
            ppl = valid(model, posterior_model, upstream_model, data, tokenizer, config)
            print(f'Train recon loss: {recon_loss / 100}, Train loss: {train_loss / 100}')
            wandb.log(dict(
                step=batch_id,
                train_loss=train_loss,
                valid_ppl=ppl,
            ))
            train_loss = 0
            recon_loss = 0
        if batch_id >= config.max_updates:
            exit(0)

@torch.no_grad()
def valid(model, posterior_model, upstream_model, data, tokenizer, config, num_samples=10):
    model.eval()
    posterior_model.eval()
    upstream_model.eval()
    valid_data = WikiDataset(data['train'])
    data_loader = DataLoader(valid_data,  batch_size=config.batch_size, pin_memory=True)
    valid_loss = 0.
    recon_loss = 0.
    total_tokens = 0
    for batch_id, raw_batch in enumerate(data_loader):
        article_batch = tokenizer(raw_batch, return_tensors='pt', padding=True)['input_ids']
        
        article_batch[:article_batch.size(0)//2, ::2] = 33
        article_batch[:article_batch.size(0)//2, 1::2] = 39
        article_batch[article_batch.size(0)//2:, ::2] = 39
        article_batch[article_batch.size(0)//2:, 1::2] = 33
        
        #article_batch[:, -1] = tokenizer.eos_token_id
        # manually specify h0
        prev_state = upstream_model.h0.repeat(article_batch.size(0), 1, 1)
        hidden = (prev_state[0].new_ones((config.lstm_layers, article_batch.size(0), config.d_latent, )),
                  prev_state[0].new_ones((config.lstm_layers, article_batch.size(0), config.d_latent, )))
        sampled_z = None
        #hidden = prev_state[0].new_ones((article_batch.size(0), 0, config.d_latent))
        for i, text in enumerate(long_sequence_splitter(article_batch, config.window_len)):
            # add eos token so the low-level model knows when to stop
            bos_text = torch.cat([text.new_ones(text.size(0), 1)*tokenizer.bos_token_id, text], dim=-1).cuda()
            #bos_text_eos = torch.cat([text.new_ones(text.size(0), 1)*tokenizer.bos_token_id, text, text.new_ones(text.size(0), 1)*tokenizer.eos_token_id], dim=-1).cuda()
            inputs = bos_text[:, :-1]
            inputs_mask = inputs != tokenizer.pad_token_id
            targets = bos_text[:, 1:]
            # add bos token so our BERT can use it to summarize the block
            bos_text = torch.cat([text.new_ones(text.size(0), config.state_len)*tokenizer.bos_token_id, text], dim=-1)
            # run q(z|x)
            state_posterior = posterior_model(bos_text.cuda(), state=sampled_z)[:, :config.state_len]
            qz_mean = state_posterior[:, :, :config.d_latent]
            qz_logvar = state_posterior[:, :, config.d_latent:]
            if config.use_mean:
                sampled_z = qz_mean
                preds, _ = model(inputs, qz_mean)
            else:
                eps = qz_mean.new(qz_mean.shape).normal_(0, 1)
                torch.nn.init.trunc_normal_(eps)
                sampled_z = qz_mean + (0.5*qz_logvar).exp()*eps
                preds, _ = model(inputs, sampled_z)
            state, hidden = upstream_model(sampled_z, hidden)
            # use z ~ q(z|x) to generate p(x|z)
            # reconstruction loss
            rec_loss = cross_entropy(preds[inputs_mask], targets[inputs_mask], reduction='sum')
            # prior matching loss KL(qz||pz)
            if config.state_len > 0:
                pz_mean = prev_state[:, :, :config.d_latent]
                pz_logvar = prev_state[:, :, config.d_latent:]
                # here we can't ignore the constants in KL like during training
                prior_loss = 0.5 * ((pz_logvar-qz_logvar) + (qz_logvar.exp()+(qz_mean - pz_mean)**2)/pz_logvar.exp() - 1).sum()
            else:
                prior_loss = 0.
            prev_state = state
            recon_loss += rec_loss.item()
            valid_loss += (rec_loss + prior_loss).item()
            total_tokens += inputs_mask.sum().item() #- inputs_mask.size(0)
        if batch_id == num_samples:
            break
    print(f'Recon PPL: {2.71828**(recon_loss/total_tokens)}, Valid PPL: {2.71828**(valid_loss/total_tokens)}')
    generate(model, upstream_model, tokenizer, config)
    model.train()
    posterior_model.train()
    upstream_model.train()
    return 2.**(valid_loss/total_tokens)


@torch.no_grad()
def generate(model, upstream_model, tokenizer, config):
    model.eval()
    upstream_model.eval()

    prev_state = upstream_model.h0.repeat(1, 1, 1)
    hidden = (prev_state[0].new_ones((config.lstm_layers, 1, config.d_latent, )),
              prev_state[0].new_ones((config.lstm_layers, 1, config.d_latent, )))
    #hidden = prev_state[0].new_ones((1, 0, config.d_latent))
    decoded = []
    for i in range(10):
        pz_mean = prev_state[:, :, :config.d_latent]
        pz_logvar = prev_state[:, :, config.d_latent:]

        eps = pz_mean.new(pz_mean.shape).normal_(0, 1)
        torch.nn.init.trunc_normal_(eps)
        sampled_z = pz_mean + (0.5*pz_logvar).exp()*eps
        bos_text = (pz_mean.new_ones(pz_mean.size(0), 1)*tokenizer.bos_token_id).long().cuda()
        for _ in range(config.window_len):
            preds, _ = model(bos_text, sampled_z)
            bos_text = torch.cat([bos_text,
                                  preds[:, -1, :].softmax(dim=-1).multinomial(num_samples=1)], dim=-1).cuda()    
        prev_state, hidden = upstream_model(sampled_z, hidden)
        decoded.append(tokenizer.decode(bos_text[0, 1:]))

    print('|'.join(decoded))

    model.train()
    upstream_model.train()


if __name__ == '__main__':
    device = 'cuda:0'
    data = load_dataset("wikipedia", "20220301.en")
    #data = load_from_disk("/home/v-edwardhu/.cache/huggingface/datasets/wikipedia/20220301.en.min10000/train")
    data.filter(lambda x : len(x['text'])>10000)

    tokenizer = setup_tokenizer()
    config = OmegaConf.load('configs/base.yaml')

    
    wandb.init(project="Hierarchical-LM", entity="edwardhu", config=config, tags=[config.wandb_tag], mode="disabled")
    wandb.define_metric("train_loss", summary="min", step_metric='step')
    wandb.define_metric("valid_ppl", summary="min", step_metric='step')

    torch.manual_seed(config.seed)

    train(data, tokenizer, config)
    