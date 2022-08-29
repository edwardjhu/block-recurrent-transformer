from locale import textdomain
from datasets import load_dataset, load_from_disk
from numpy import broadcast
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler, DataLoader
from torch import nn
from torch.nn.functional import cross_entropy, softplus
import torch
from tqdm import tqdm
from transformers import AdamW, GPT2Tokenizer
from x_transformers.x_transformers import TokenEmbedding, AbsolutePositionalEmbedding
import wandb

import os

from block_recurrent_transformer import BlockRecurrentBlock, long_sequence_splitter
from block_recurrent_transformer.transformer import VanillaTransformerBlock

# apex magic
from apex import amp

class WikiDataset:
    def __init__(self, data, max_char=640):
        self.data = data
        self.max_char = max_char
    
    def __getitem__(self, i: int):
        record = self.data[i]
        title = record['title']
        text = record['text']
        return f'{title}\n\n{text}'[:self.max_char]
    
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

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.h0 = nn.Parameter(torch.zeros(1, config.state_len, 2*output_dim))
        self.to_output = nn.Linear(hidden_dim, 2*output_dim)
    
    def forward(self, x, hidden):
        if x.size(1) > 0:
            out, hidden = self.lstm(x, hidden)
            out = self.to_output(out)
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

    def __init__(self, config, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.ModuleList([VanillaTransformerBlock(config, hidden_dim) \
                                        for _ in range(config.num_layers)])
        self.pos_embed = AbsolutePositionalEmbedding(hidden_dim, 999)
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_gaussian = nn.Linear(hidden_dim, 2*output_dim)
        self.z_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.h0 = nn.Parameter(torch.zeros(1, config.state_len, 2*output_dim))
        self.state_len = config.state_len
    
    def forward(self, x, hidden):
        x = torch.cat([hidden, x], dim=1)
        out = self.z_to_hidden(x) + self.pos_embed(x[:,:,-1])
        for layer in self.layers:
            out = layer(out)
        return self.to_gaussian(out[:, -self.state_len:, :]), x

class PosteriorMLP(nn.Module):
    """As simple as I can make the model.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, double_output=True):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        if double_output:
            self.output_layer = nn.Linear(hidden_dim, 2*output_dim)
        else:
            self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.layers:
            x = nn.functional.relu(layer(x))
        return self.output_layer(x)

def setup_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

from itertools import chain
def train(data, tokenizer_posterior, tokenizer_decoder, config):
    global_step = 0
    global_recon_step = 0
    upstream_model = UpstreamLSTM(input_dim=config.d_latent,
                                  hidden_dim=config.d_model,
                                  output_dim=config.d_latent,
                                  num_layers=config.lstm_layers)
    #upstream_model = UpstreamLSTMDiscrete(len(tokenizer), config.d_latent, num_layers=config.lstm_layers)
    #upstream_model = UpstreamTransformer(config,
    #                                    input_dim=config.d_latent,
    #                                    hidden_dim=config.d_model,
    #                                    output_dim=config.d_latent)
    upstream_model.to(device)

    posterior_lstm = UpstreamLSTM(input_dim=config.d_model,
                                  hidden_dim=config.d_model,
                                  output_dim=config.d_latent,
                                  num_layers=config.lstm_layers)
    #upstream_model = UpstreamLSTMDiscrete(len(tokenizer), config.d_latent, num_layers=config.lstm_layers)
    #upstream_model = UpstreamTransformer(config, len(tokenizer), config.d_latent)
    posterior_lstm.to(device)

    posterior_mlp = PosteriorMLP(input_dim=config.d_model + config.d_latent,
                                  hidden_dim=config.d_model,
                                  output_dim=config.d_latent,
                                  num_layers=1)
    posterior_mlp.to(device)

    if os.path.exists(config.load_decoder):
        model = torch.load(config.load_decoder)
    else:
        #model = BlockRecurrentDecoder(config, len(tokenizer), config.d_model, d_latent=config.d_latent)
        #model = LogLinearDecoder(config, len(tokenizer), config.d_model, d_latent=config.d_latent)
        from transformers_overrides import OPTForCausalLM_ours
        model = OPTForCausalLM_ours.from_pretrained('facebook/opt-125m', d_latent=config.d_latent)
    
    model.to(device)
    #for p in model.parameters():
    #    p.requires_grad = False
    #model.z_to_hidden.weight.requires_grad = True
    #model.z_to_hidden.bias.requires_grad = True
    
    if os.path.exists(config.load_posterior):
        posterior_model = torch.load(config.load_posterior)
    else:
        #posterior_model = BiTransformer(config, len(tokenizer), config.d_model, d_latent=config.d_latent)
        posterior_model = RobertaModel.from_pretrained('roberta-base')
        critic_mlp = PosteriorMLP(input_dim=config.d_latent+config.d_model,
                                  hidden_dim=config.d_model,
                                  output_dim=1,
                                  num_layers=2,
                                  double_output=False)
    posterior_model.to(device)
    critic_mlp.to(device)
    #for p in posterior_model.parameters():
    #    p.requires_grad = False


    opt = AdamW([{'params': chain(posterior_model.parameters(),
                                  model.parameters(),
                                  posterior_mlp.parameters(),
                                  critic_mlp.parameters()), 'lr': config.lr, 'max_lr': config.lr}],
                betas=(config.adam_beta1, config.adam_beta2), weight_decay=0.0)
    opt1 = AdamW([{'params': chain(upstream_model.parameters(), ), 'lr': 4*config.lr, 'max_lr': 4*config.lr}],
                betas=(config.adam_beta1, config.adam_beta2), weight_decay=0.0)
    
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
    annealing_sched = frange_cycle_linear(config.max_updates, start=0.0, stop=1.0, n_cycle=50)

    train_data = WikiDataset(data['train'])
    data_loader = DataLoader(train_data,  batch_size = config.batch_size, sampler = RandomSampler(train_data), pin_memory=True)

    train_loss = 0
    recon_loss = 0
    
    for batch_id, raw_batch in enumerate(tqdm(data_loader), 1):
        opt.zero_grad()
        opt1.zero_grad()
        article_batch = tokenizer_decoder(raw_batch, return_tensors='pt', padding=True, truncation=True, max_length=32)['input_ids']
        article_batch = article_batch.cuda()
        '''
        article_batch[:article_batch.size(0)//2, ::2] = 33
        article_batch[:article_batch.size(0)//2, 1::2] = 39
        article_batch[article_batch.size(0)//2:, ::2] = 39
        article_batch[article_batch.size(0)//2:, 1::2] = 33
        '''
        # manually specify h0
        prev_state = [upstream_model.h0.repeat(article_batch.size(0), 1, 1)]
        hidden = (prev_state[0].new_zeros((config.lstm_layers, article_batch.size(0), config.d_model, )),
                  prev_state[0].new_zeros((config.lstm_layers, article_batch.size(0), config.d_model, )))
        #hidden = prev_state[0].new_zeros(article_batch.size(0), 0, config.d_latent)
        #hidden_posterior = \
        #         (prev_state[0].new_zeros((config.lstm_layers, article_batch.size(0), config.d_model, )),
        #          prev_state[0].new_zeros((config.lstm_layers, article_batch.size(0), config.d_model, )))
        #sampled_z = None
        sampled_z = prev_state[0].new_zeros(article_batch.size(0), 1, config.d_latent)
        batch_rec_loss = 0.
        batch_prior_loss = 0.
        if global_step <= config.warmup:
            for param_group in opt.param_groups:
                param_group['lr'] = param_group['max_lr'] * global_step / config.warmup
        if global_recon_step <= config.warmup:
            for param_group in opt1.param_groups:
                param_group['lr'] = param_group['max_lr'] * global_recon_step / config.warmup
        for i, text in enumerate(long_sequence_splitter(article_batch, config.window_len)):
            # add eos token so the low-level model knows when to stop
            #bos_text_eos = torch.cat([text.new_ones(text.size(0), 1)*tokenizer.bos_token_id, text, text.new_ones(text.size(0), 1)*tokenizer.eos_token_id], dim=-1).cuda()
            bos_text_decoder = torch.cat([text.new_ones(text.size(0), 1)*tokenizer_decoder.bos_token_id, text], dim=-1)
            inputs = bos_text_decoder[:, :-1]
            targets = bos_text_decoder[:, 1:]
            inputs_mask = (inputs != tokenizer_decoder.pad_token_id) & (targets != tokenizer_decoder.pad_token_id)
            # add bos token so our BERT can use it to summarize the block
            bos_text_eos_posterior = torch.cat([text.new_ones(text.size(0), 1)*tokenizer_posterior.bos_token_id,
                                                text,
                                                text.new_ones(text.size(0), 1)*tokenizer_posterior.eos_token_id], dim=-1)
            # run q(z|x)
            encoded_xi = posterior_model(bos_text_eos_posterior)['pooler_output']

            pooled_posterior = posterior_mlp(torch.cat([encoded_xi, sampled_z.squeeze(1)], dim=-1))
            # LSTM posterior
            #state_posterior, hidden_posterior = posterior_lstm(pooled_posterior.unsqueeze(1), hidden_posterior)
            #qz_mean = state_posterior[:, :, :config.d_latent]
            #qz_logvar = state_posterior[:, :, config.d_latent:]
            # use z ~ q(z|x) to generate p(x|z)
            #eps = qz_mean.new(qz_mean.shape).normal_(0, 1)
            #torch.nn.init.trunc_normal_(eps)

            #if sampled_z is None:
            #pooled_posterior = model.hidden_to_z(pooled_posterior)
            qz_mean = pooled_posterior.unsqueeze(1)[:, :, :config.d_latent]
            qz_logvar = pooled_posterior.unsqueeze(1)[:, :, config.d_latent:]
            #qz_var = softplus(pooled_posterior.unsqueeze(1)[:, :, config.d_latent:])
            eps = qz_mean.new(qz_mean.shape).normal_(0, 1)
            torch.nn.init.trunc_normal_(eps)
            qz_mean_true = qz_mean
            sampled_z = qz_mean_true + (0.5 * qz_logvar).exp() * eps
            #sampled_z = qz_mean_true + qz_var**0.5 * eps
            '''
            else:
                pooled_posterior = model.hidden_to_z(pooled_posterior)
                qz_mean = pooled_posterior.unsqueeze(1)[:, :, :config.d_latent]
                #qz_logvar = pooled_posterior.unsqueeze(1)[:, :, config.d_latent:]
                qz_var = softplus(pooled_posterior.unsqueeze(1)[:, :, config.d_latent:])
                eps = qz_mean.new(qz_mean.shape).normal_(0, 1)
                torch.nn.init.trunc_normal_(eps)
                qz_mean_true = sampled_z * 0.5 + qz_mean
                #sampled_z = qz_mean_true + (0.5 * qz_logvar).exp() * eps
                sampled_z = qz_mean_true + qz_var**0.5 * eps
            '''
            preds = model(inputs_embeds=torch.cat([model.z_to_hidden(sampled_z),
                                model.get_input_embeddings()(inputs)], dim=1))['logits'][:,sampled_z.size(1):,:]
            
            state, hidden = upstream_model(sampled_z, hidden)

            # critic model
            sampled_z = sampled_z.squeeze(1)
            critic_loss = -torch.cat([critic_mlp(torch.cat([sampled_z, encoded_xi.roll(i, dims=0)], dim=-1)) \
                                        for i in range(sampled_z.size(0))], dim=-1).log_softmax(dim=-1)[:, 0]
            critic_loss = critic_loss.mean()


            # reconstruction loss
            rec_loss = cross_entropy(preds[inputs_mask], targets[inputs_mask])
            # prior matching loss KL(qz||pz)
            if config.state_len > 0:
                pz_mean = prev_state[-1][:, :, :config.d_latent]
                pz_logvar = prev_state[-1][:, :, config.d_latent:]
                #pz_var = softplus(prev_state[-1][:, :, config.d_latent:])
                prior_loss = 0.5 * ((pz_logvar-qz_logvar) + (qz_logvar.exp()+(qz_mean_true - pz_mean)**2)/pz_logvar.exp() - 1).sum(-1).mean()
                #prior_loss = 0.5 * ((pz_var/qz_var.detach()).log() + (qz_var.detach()+(qz_mean_true.detach() - pz_mean)**2)/pz_var - 1).sum(-1).mean()
                # variants
                #prior_loss = ((pz_logvar-qz_logvar.detach())**2+(qz_mean_true.detach() - pz_mean)**2).sum(-1).mean()
                #prior_loss = 0.5 * ((pz_logvar-qz_logvar) + (qz_logvar.exp()+(qz_mean_true - pz_mean)**2)/pz_logvar.exp() - 1).sum(-1).mean()
                prior_reg = 0.1*(0.5 * ((-qz_logvar) + (qz_logvar.exp()+(qz_mean)**2) - 1).sum(-1).mean())
                #prior_reg = 0.005*(0.5 * ((-qz_logvar) + (qz_logvar.exp()+(qz_mean)**2) - 1).sum(-1).mean())
                #prior_loss = ((qz_mean.detach() - pz_mean)**2).sum(-1).mean()
                #prior_loss = ((pz_logvar-qz_logvar.detach())**2 + (qz_mean.detach() - pz_mean)**2).sum(-1).mean()
            else:
                prior_loss = 0.
            prev_state.append(state)
            #if rec_loss > 1:
            #    batch_rec_loss += rec_loss #+prior_reg
            #else:

            # critic + recon
            #batch_rec_loss += rec_loss + critic_loss #+ prior_reg
            # critic only
            batch_rec_loss += critic_loss #+ prior_reg
            batch_prior_loss += prior_loss #+prior_reg
            
        train_loss += (batch_rec_loss+batch_prior_loss).item() / i
        recon_loss += batch_rec_loss.item() / i
        
        # set the coe below to 0 to disable KL
        (0.0005*annealing_sched[global_step]*batch_prior_loss + batch_rec_loss).backward()
        #(0.01 * batch_prior_loss + batch_rec_loss).backward()
        #if global_recon_step < config.max_recon_updates:
        #if (global_step // 500) % 2 == 0:
        #torch.nn.utils.clip_grad_norm_(chain(posterior_mlp.parameters(), model.parameters(), posterior_model.parameters()), config.gclip)
        opt.step()
        global_recon_step += 1
        #else:
        #torch.nn.utils.clip_grad_norm_(chain(upstream_model.parameters()), config.gclip)
        opt1.step()
        global_step += 1
        log_freq = 100
        if batch_id % log_freq == 0:
            ppl = valid(model, posterior_model, upstream_model, posterior_lstm,
                        data, tokenizer_posterior, tokenizer_decoder, config)
            print(f'Train recon loss: {recon_loss / log_freq}, Train loss: {train_loss / log_freq}')
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
def valid(model, posterior_model, upstream_model, posterior_lstm, 
          data, tokenizer_posterior, tokenizer_decoder, config, num_samples=10):
    # VALIDATION LOGIC IS NOT UPDATE TO DATE.
    model.eval()
    posterior_model.eval()
    upstream_model.eval()
    posterior_lstm.eval()
    valid_data = WikiDataset(data['train'])
    data_loader = DataLoader(valid_data,  batch_size=config.batch_size, pin_memory=True)
    valid_loss = 0.
    recon_loss = 0.
    total_tokens = 0
    for batch_id, raw_batch in enumerate(data_loader):
        article_batch = tokenizer_decoder(raw_batch, return_tensors='pt', padding=True)['input_ids']
        article_batch = article_batch.cuda()
        '''
        article_batch[:article_batch.size(0)//2, ::2] = 33
        article_batch[:article_batch.size(0)//2, 1::2] = 39
        article_batch[article_batch.size(0)//2:, ::2] = 39
        article_batch[article_batch.size(0)//2:, 1::2] = 33
        '''
        # manually specify h0
        prev_state = upstream_model.h0.repeat(article_batch.size(0), 1, 1)
        hidden = (prev_state[0].new_zeros((config.lstm_layers, article_batch.size(0), config.d_model, )),
                  prev_state[0].new_zeros((config.lstm_layers, article_batch.size(0), config.d_model, )))
        #hidden = prev_state[0].new_zeros(article_batch.size(0), 0, config.d_latent)
        hidden_posterior = \
                 (prev_state[0].new_zeros((config.lstm_layers, article_batch.size(0), config.d_model, )),
                  prev_state[0].new_zeros((config.lstm_layers, article_batch.size(0), config.d_model, )))
        sampled_z = None
        #hidden = prev_state[0].new_zeros((article_batch.size(0), 0, config.d_latent))
        for i, text in enumerate(long_sequence_splitter(article_batch, config.window_len)):
            # add eos token so the low-level model knows when to stop
            bos_text_decoder = torch.cat([text.new_ones(text.size(0), 1)*tokenizer_decoder.bos_token_id, text], dim=-1)
            inputs = bos_text_decoder[:, :-1]
            targets = bos_text_decoder[:, 1:]
            inputs_mask = (inputs != tokenizer_decoder.pad_token_id) & (targets != tokenizer_decoder.pad_token_id)
            bos_text_eos_posterior = torch.cat([text.new_ones(text.size(0), 1)*tokenizer_posterior.bos_token_id,
                                                text,
                                                text.new_ones(text.size(0), 1)*tokenizer_posterior.eos_token_id], dim=-1)
            # run q(z|x)
            pooled_posterior = posterior_model(bos_text_eos_posterior)['pooler_output']
            state_posterior, hidden_posterior = posterior_lstm(pooled_posterior.unsqueeze(1), hidden_posterior)

            qz_mean = state_posterior[:, :, :config.d_latent]
            qz_logvar = state_posterior[:, :, config.d_latent:]
            if config.use_mean:
                sampled_z = qz_mean
                preds = model(inputs_embeds=torch.cat([model.z_to_hidden(sampled_z),
                              model.get_input_embeddings()(inputs)], dim=1))['logits'][:,1:,:]
            else:
                eps = qz_mean.new(qz_mean.shape).normal_(0, 1)
                torch.nn.init.trunc_normal_(eps)
                sampled_z = qz_mean #+ (0.5*qz_logvar).exp()*eps
                preds = model(inputs_embeds=torch.cat([model.z_to_hidden(sampled_z),
                              model.get_input_embeddings()(inputs)], dim=1))['logits'][:,1:,:]
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
            valid_loss += 0 #(rec_loss + prior_loss).item()
            total_tokens += inputs_mask.sum().item() #- inputs_mask.size(0)
        if batch_id == num_samples:
            break
    print(f'Recon PPL: {2.71828**(recon_loss/total_tokens)}, Valid PPL: {2.71828**(valid_loss/total_tokens)}')
    generate(model, upstream_model, tokenizer_decoder, config, mode='argmax')
    print('---------')
    generate(model, upstream_model, tokenizer_decoder, config, mode='sample')
    #model.train()
    #posterior_model.train()
    upstream_model.train()
    posterior_lstm.train()
    return 2.**(valid_loss/total_tokens)


@torch.no_grad()
def generate(model, upstream_model, tokenizer, config, mode='argmax'):
    model.eval()
    upstream_model.eval()

    prev_state = upstream_model.h0.repeat(1, 1, 1)
    hidden = (prev_state[0].new_zeros((config.lstm_layers, 1, config.d_model, )),
              prev_state[0].new_zeros((config.lstm_layers, 1, config.d_model, )))
    #hidden = prev_state[0].new_zeros(prev_state[0].size(0), 0, config.d_latent)
    #hidden = prev_state[0].new_zeros((1, 0, config.d_latent))
    decoded = []
    for i in range(10):
        pz_mean = prev_state[:, :, :config.d_latent]
        pz_logvar = prev_state[:, :, config.d_latent:]

        eps = pz_mean.new(pz_mean.shape).normal_(0, 1)
        torch.nn.init.trunc_normal_(eps)
        sampled_z = pz_mean + (0.5*pz_logvar).exp()*eps
        bos_text = (pz_mean.new_ones(pz_mean.size(0), 1)*tokenizer.bos_token_id).long().cuda()
        for _ in range(config.window_len):
            preds = model(inputs_embeds=torch.cat([model.z_to_hidden(sampled_z),
                        model.get_input_embeddings()(bos_text)], dim=1))['logits'][:,sampled_z.size(1):,:]
            if mode == 'sample':
                bos_text = torch.cat([bos_text,
                                    preds[:, -1, :].softmax(dim=-1).multinomial(num_samples=1)], dim=-1).cuda()
            elif mode == 'argmax':   
                bos_text = torch.cat([bos_text,
                                    preds[:, -1:, :].argmax(dim=-1)], dim=-1).cuda()    
        prev_state, hidden = upstream_model(sampled_z, hidden)
        decoded.append(tokenizer.decode(bos_text[0, 1:]))

    print('|'.join(decoded))

    #model.train()
    upstream_model.train()


if __name__ == '__main__':
    device = 'cuda:0'
    data = load_dataset("wikipedia", "20220301.en")
    data.filter(lambda x : len(x['text'])>10000)

    #tokenizer = setup_tokenizer()
    config = OmegaConf.load('configs/base.yaml')

    from transformers import RobertaTokenizer, RobertaModel, GPT2Tokenizer

    tokenizer_posterior = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer_decoder = GPT2Tokenizer.from_pretrained('facebook/opt-125m', add_bos_token=False)    

    #last_hidden_states = outputs.last_hidden_state
    
    wandb.init(project="Hierarchical-LM", entity="edwardhu", config=config, tags=[config.wandb_tag], mode="disabled")
    wandb.define_metric("train_loss", summary="min", step_metric='step')
    wandb.define_metric("valid_ppl", summary="min", step_metric='step')

    torch.manual_seed(config.seed)

    train(data, tokenizer_posterior, tokenizer_decoder, config)
    