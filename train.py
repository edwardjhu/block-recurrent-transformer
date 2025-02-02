from datasets import load_dataset, load_from_disk
from numpy import broadcast
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler, DataLoader
from torch import nn
from torch.nn.functional import cross_entropy
import torch
from tqdm import tqdm
from transformers import AdamW, GPT2Tokenizer
from x_transformers.x_transformers import TokenEmbedding
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
        self.layers = nn.ModuleList([BlockRecurrentBlock(config, dim, dim) \
                                        for _ in range(config.num_layers)])
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        self.to_gaussian = nn.Linear(dim, 2*dim, bias=False)
        self.h0 = nn.Parameter(torch.zeros(1, config.state_len, 2*dim))
        self.norm = nn.LayerNorm(dim)
        self.dim = dim
    
    def forward(self, x, state=None):
        if state is None:
            broadcasted_h0 = self.h0.repeat(x.size(0), 1, 1)
            h0_mean = broadcasted_h0[:, :, :self.dim]
            h0_logvar = broadcasted_h0[:, :, self.dim:]
            eps = h0_mean.new(h0_mean.shape).normal_(0, 1)
            state = h0_mean + h0_logvar*eps
        x = self.embed(x)
        for layer in self.layers:
            x, state = layer(x, state)
        x = self.to_logits(self.norm(x))
        state = self.to_gaussian(state)
        return x, state

class BiTransformer(nn.Module):
    """As simple as I can make the model.
    """

    def __init__(self, config, num_tokens, dim):
        super().__init__()
        self.embed = TokenEmbedding(dim, num_tokens)
        self.layers = nn.ModuleList([VanillaTransformerBlock(config, dim) \
                                        for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.to_gaussian = nn.Linear(dim, 2*dim, bias=False)
    
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.to_gaussian(x)
        return x


def setup_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

from itertools import chain
def train(data, tokenizer, config):
    model = BlockRecurrentDecoder(config, len(tokenizer), config.d_model)
    model.to(device)
    posterior_model = BiTransformer(config, len(tokenizer), config.d_model)
    posterior_model.to(device)
    opt = AdamW([{'params': model.parameters(), 'lr': config.lr, 'max_lr': config.lr},
                 {'params': posterior_model.parameters(), 'lr': config.lr, 'max_lr': 0.2*config.lr}],
                betas=(config.adam_beta1, config.adam_beta2), weight_decay=0.01)

    # apex magic
    models, opt = amp.initialize(
        [model, posterior_model],
        opt,
        opt_level='O2',
        )
    model = models[0]
    posterior_model = models[1]

    train_data = WikiDataset(data['train'])
    data_loader = DataLoader(train_data,  batch_size = config.batch_size, sampler = RandomSampler(train_data), pin_memory=True)

    train_loss = 0
    for batch_id, raw_batch in enumerate(tqdm(data_loader), 1):
        opt.zero_grad()
        article_batch = tokenizer(raw_batch, return_tensors='pt', padding=True)['input_ids']
        # manually specify h0
        prev_state = model.h0.repeat(article_batch.size(0), 1, 1)
        if batch_id <= config.warmup:
            for param_group in opt.param_groups:
                param_group['lr'] = param_group['max_lr'] * batch_id / config.warmup
        for i, text in enumerate(long_sequence_splitter(article_batch, config.window_len-1)):
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
            preds, state = model(inputs, qz_mean + (0.5*qz_logvar).exp()*eps)
            # reconstruction loss
            rec_loss = cross_entropy(preds[inputs_mask], targets[inputs_mask])
            # prior matching loss KL(qz||pz)
            if config.state_len > 0:
                pz_mean = prev_state[:, :, :config.d_model]
                pz_logvar = prev_state[:, :, config.d_model:]
                prior_loss = 0.5 * ((pz_logvar-qz_logvar) + (qz_logvar.exp()+(qz_mean - pz_mean)**2)/pz_logvar.exp() - 1).sum(-1).mean()
            else:
                prior_loss = 0.
            prev_state = state
            loss = rec_loss+prior_loss
            #loss.backward(retain_graph=True)
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward(retain_graph=True)
            train_loss += loss.item()
        opt.step()
        if batch_id % 10 == 0:
            ppl = valid(model, posterior_model, data, tokenizer, config)
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
def valid(model, posterior_model, data, tokenizer, config, num_samples=10):
    model.eval()
    posterior_model.eval()
    valid_data = WikiDataset(data['train'])
    data_loader = DataLoader(valid_data,  batch_size=config.batch_size, pin_memory=True)
    valid_loss = 0.
    total_tokens = 0
    for batch_id, raw_batch in enumerate(data_loader):
        article_batch = tokenizer(raw_batch, return_tensors='pt', padding=True)['input_ids']
        # manually specify h0
        prev_state = model.h0.repeat(article_batch.size(0), 1, 1)
        for i, text in enumerate(long_sequence_splitter(article_batch, config.window_len-1)):
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
                preds, state = model(inputs, qz_mean)
            else:
                eps = qz_mean.new(qz_mean.shape).normal_(0, 1)
                preds, state = model(inputs, qz_mean + (0.5*qz_logvar).exp()*eps)
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
            valid_loss += rec_loss.item() + prior_loss.item()
            total_tokens += inputs_mask.sum().item() - inputs_mask.size(0)
        if batch_id == num_samples:
            break
    print(f'Valid PPL: {2.**(valid_loss/total_tokens)}')
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
    