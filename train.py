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
    def __init__(self, data, max_char=1000):
        self.data = data
        self.max_char = max_char
    
    def __getitem__(self, i: int):
        record = self.data[i]
        title = record['title']
        text = record['text']
        return f'{title}\n\n{text}'[:self.max_char] + ' [EOS]'
    
    def __len__(self):
        return len(self.data)


class BlockRecurrentDecoder(nn.Module):
    """As simple as I can make the model.
    """

    def __init__(self, config, num_tokens, dim):
        super().__init__()
        self.embed = TokenEmbedding(dim, num_tokens)
        self.pos_embed = AbsolutePositionalEmbedding(dim, config.window_len + 10)
        self.layers = nn.ModuleList([BlockRecurrentBlock(config, dim, dim) \
                                        for _ in range(config.num_layers)])
        self.to_logits = nn.Linear(dim, num_tokens)
        self.to_gaussian = nn.Linear(dim, 2*dim)
        self.h0 = nn.Parameter(torch.zeros(1, config.state_len, 2*dim))
        self.norm = nn.LayerNorm(dim)
        self.norm_state = nn.LayerNorm(dim)
        self.dim = dim
    
    def forward(self, x, state=None):
        if state is None:
            broadcasted_h0 = self.h0.repeat(x.size(0), 1, 1)
            h0_mean = broadcasted_h0[:, :, :self.dim]
            h0_logvar = broadcasted_h0[:, :, self.dim:]
            eps = h0_mean.new(h0_mean.shape).normal_(0, 1)
            state = h0_mean + h0_logvar*eps
        x = self.embed(x) + self.pos_embed(x)
        for layer in self.layers:
            x, state = layer(x, state)
        x = self.to_logits(self.norm(x))
        state = self.to_gaussian(self.norm_state(state))
        return x, state

class BiTransformer(nn.Module):
    """As simple as I can make the model.
    """

    def __init__(self, config, num_tokens, dim):
        super().__init__()
        self.embed = TokenEmbedding(dim, num_tokens)
        self.pos_embed = AbsolutePositionalEmbedding(dim, config.window_len + 10)
        self.layers = nn.ModuleList([VanillaTransformerBlock(config, dim) \
                                        for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.to_gaussian = nn.Linear(dim, 2*dim)
    
    def forward(self, x):
        x = self.embed(x) + self.pos_embed(x)
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
        self.to_gaussian = nn.Linear(dim, 2*dim)
    
    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.to_gaussian(out)
        return out, hidden


def setup_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

from itertools import chain
def train(data, tokenizer, config):
    upstream_model = UpstreamLSTM(config.d_model, num_layers=config.lstm_layers)
    upstream_model.to(device)
    model = BlockRecurrentDecoder(config, len(tokenizer), config.d_model)
    model.to(device)
    posterior_model = BiTransformer(config, len(tokenizer), config.d_model)
    posterior_model.to(device)
    opt = AdamW([{'params': model.parameters(), 'lr': config.lr, 'max_lr': config.lr},
                 {'params': upstream_model.parameters(), 'lr': config.lr, 'max_lr': config.lr},
                 {'params': posterior_model.parameters(), 'lr': 0.2*config.lr, 'max_lr': 0.2*config.lr}],
                betas=(config.adam_beta1, config.adam_beta2), weight_decay=0.01)
    
    # apex magic
    models, opt = amp.initialize(
        [model, posterior_model, upstream_model],
        opt,
        opt_level='O0',
        )
    model = models[0]
    posterior_model = models[1]
    upstream_model = models[2]
    
    train_data = WikiDataset(data['train'])
    data_loader = DataLoader(train_data,  batch_size = config.batch_size, sampler = RandomSampler(train_data), pin_memory=True)

    train_loss = 0
    for batch_id, raw_batch in enumerate(tqdm(data_loader), 1):
        opt.zero_grad()
        article_batch = tokenizer(raw_batch, return_tensors='pt', padding=True)['input_ids']
        # manually specify h0
        prev_state = [model.h0.repeat(article_batch.size(0), 1, 1)]
        hidden = (prev_state[0].new_zeros((config.lstm_layers, article_batch.size(0), config.d_model, )),
                  prev_state[0].new_zeros((config.lstm_layers, article_batch.size(0), config.d_model, )))
        batch_loss = 0.
        if batch_id <= config.warmup:
            for param_group in opt.param_groups:
                param_group['lr'] = param_group['max_lr'] * batch_id / config.warmup
        for i, text in enumerate(long_sequence_splitter(article_batch, config.window_len)):
            # add eos token so the low-level model knows when to stop
            bos_text_eos = torch.cat([text.new_ones(text.size(0), 1)*tokenizer.bos_token_id, text, text.new_ones(text.size(0), 1)*tokenizer.eos_token_id], dim=-1).cuda()
            inputs = bos_text_eos[:, :-1]
            inputs_mask = inputs != tokenizer.pad_token_id
            targets = bos_text_eos[:, 1:]
            # add bos token so our BERT can use it to summarize the block
            bos_text = torch.cat([text.new_ones(text.size(0), config.state_len)*tokenizer.bos_token_id, text], dim=-1)
            # run q(z|x)
            state_posterior = posterior_model(bos_text.cuda())[:, :config.state_len]
            qz_mean = state_posterior[:, :, :config.d_model]
            qz_logvar = state_posterior[:, :, config.d_model:]
            # use z ~ q(z|x) to generate p(x|z)
            eps = qz_mean.new(qz_mean.shape).normal_(0, 1)
            sampled_z = qz_mean + (0.5*qz_logvar).exp()*eps
            preds, _ = model(inputs, sampled_z)
            state, hidden = upstream_model(sampled_z, hidden)
            # reconstruction loss
            rec_loss = cross_entropy(preds[inputs_mask], targets[inputs_mask])
            # prior matching loss KL(qz||pz)
            if config.state_len > 0:
                pz_mean = prev_state[-1][:, :, :config.d_model]
                pz_logvar = prev_state[-1][:, :, config.d_model:]
                prior_loss = 0.5 * ((pz_logvar-qz_logvar) + (qz_logvar.exp()+(qz_mean - pz_mean)**2)/pz_logvar.exp() - 1).sum(-1).mean()
            else:
                prior_loss = 0.
            prev_state.append(state)
            batch_loss += rec_loss+prior_loss
            #loss.backward(retain_graph=True)
            
            train_loss += batch_loss.item()
        with amp.scale_loss(batch_loss / i, opt) as scaled_loss:
            scaled_loss.backward()
        opt.step()
        if batch_id % 10 == 0:
            ppl = valid(model, posterior_model, upstream_model, data, tokenizer, config)
            print(f'Train loss: {train_loss / 10}')
            wandb.log(dict(
                step=batch_id,
                train_loss=train_loss,
                valid_ppl=ppl,
            ))
            train_loss = 0
        if batch_id >= config.max_updates:
            exit(0)

@torch.no_grad()
def valid(model, posterior_model, upstream_model, data, tokenizer, config, num_samples=10):
    model.eval()
    posterior_model.eval()
    valid_data = WikiDataset(data['train'])
    data_loader = DataLoader(valid_data,  batch_size=config.batch_size, pin_memory=True)
    valid_loss = 0.
    recon_loss = 0.
    total_tokens = 0
    for batch_id, raw_batch in enumerate(data_loader):
        article_batch = tokenizer(raw_batch, return_tensors='pt', padding=True)['input_ids']
        # manually specify h0
        prev_state = model.h0.repeat(article_batch.size(0), 1, 1)
        hidden = (prev_state[0].new_zeros((config.lstm_layers, article_batch.size(0), config.d_model, )),
                  prev_state[0].new_zeros((config.lstm_layers, article_batch.size(0), config.d_model, )))
        for i, text in enumerate(long_sequence_splitter(article_batch, config.window_len)):
            # add eos token so the low-level model knows when to stop
            bos_text_eos = torch.cat([text.new_ones(text.size(0), 1)*tokenizer.bos_token_id, text, text.new_ones(text.size(0), 1)*tokenizer.eos_token_id], dim=-1).cuda()
            inputs = bos_text_eos[:, :-1]
            inputs_mask = inputs != tokenizer.pad_token_id
            targets = bos_text_eos[:, 1:]
            # add bos token so our BERT can use it to summarize the block
            bos_text = torch.cat([text.new_ones(text.size(0), config.state_len)*tokenizer.bos_token_id, text], dim=-1)
            # run q(z|x)
            state_posterior = posterior_model(bos_text.cuda())[:, :config.state_len]
            qz_mean = state_posterior[:, :, :config.d_model]
            qz_logvar = state_posterior[:, :, config.d_model:]
            if config.use_mean:
                sampled_z = qz_mean
                preds, _ = model(inputs, qz_mean)
            else:
                eps = qz_mean.new(qz_mean.shape).normal_(0, 1)
                sampled_z = qz_mean + (0.5*qz_logvar).exp()*eps
                preds, _ = model(inputs, sampled_z)
            state, hidden = upstream_model(sampled_z, hidden)
            # use z ~ q(z|x) to generate p(x|z)
            # reconstruction loss
            rec_loss = cross_entropy(preds[inputs_mask], targets[inputs_mask], reduction='sum')
            # prior matching loss KL(qz||pz)
            if config.state_len > 0:
                pz_mean = prev_state[:, :, :config.d_model]
                pz_logvar = prev_state[:, :, config.d_model:]
                # here we can't ignore the constants in KL like during training
                prior_loss = 0.5 * ((pz_logvar-qz_logvar) + (qz_logvar.exp()+(qz_mean - pz_mean)**2)/pz_logvar.exp() - 1).sum()
            else:
                prior_loss = 0.
            prev_state = state
            recon_loss += rec_loss.item()
            valid_loss += (rec_loss + prior_loss).item()
            total_tokens += inputs_mask.sum().item() - inputs_mask.size(0)
        if batch_id == num_samples:
            break
    print(f'Recon PPL: {2.**(recon_loss/total_tokens)}, Valid PPL: {2.**(valid_loss/total_tokens)}')
    model.train()
    posterior_model.train()
    return 2.**(valid_loss/total_tokens)


if __name__ == '__main__':
    device = 'cuda:0'
    data = load_dataset("wikipedia", "20220301.en")
    #data = load_from_disk("/home/v-edwardhu/.cache/huggingface/datasets/wikipedia/20220301.en.min10000/train")
    data.filter(lambda x : len(x['text'])>10000)

    tokenizer = setup_tokenizer()
    config = OmegaConf.load('configs/base.yaml')

    
    wandb.init(project="Hierarchical-LM", entity="edwardhu", config=config, tags=[config.wandb_tag], mode="online")
    wandb.define_metric("train_loss", summary="min", step_metric='step')
    wandb.define_metric("valid_ppl", summary="min", step_metric='step')

    torch.manual_seed(config.seed)

    train(data, tokenizer, config)
    