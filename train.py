from datasets import load_dataset
from numpy import broadcast
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler, DataLoader
from torch import nn
from torch.nn.functional import cross_entropy
from torch.optim import Adam
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer
from x_transformers.x_transformers import TokenEmbedding
import wandb

from block_recurrent_transformer import BlockRecurrentAttention, long_sequence_splitter
from block_recurrent_transformer.transformer import VanillaTransformerBlock


class WikiDataset:
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, i: int):
        record = self.data[i]
        title = record['title']
        text = record['text']
        return f'{title}\n\n{text}'
    
    def __len__(self):
        return len(self.data)


class BlockRecurrentDecoder(nn.Module):
    """As simple as I can make the model.
    """

    def __init__(self, config, num_tokens, dim):
        super().__init__()
        self.embed = TokenEmbedding(dim, num_tokens)
        self.attn = BlockRecurrentAttention(config, dim, dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        self.to_gaussian = nn.Linear(dim, 2*dim, bias=False)
        self.h0 = nn.Parameter(torch.zeros(1, config.state_len, 2*dim))
        self.norm = nn.LayerNorm(dim)
        self.dim = dim
    
    def forward(self, x, state=None):
        broadcasted_h0 = self.h0.repeat(x.size(0), 1, 1)
        h0_mean = broadcasted_h0[:, :, :self.dim]
        h0_logvar = broadcasted_h0[:, :, self.dim:]
        eps = h0_mean.new(h0_mean.shape).normal_(0, 1)
        x, state = self.attn(self.embed(x), state if state is not None else \
                                            h0_mean + h0_logvar*eps)
        x = self.to_logits(self.norm(x))
        state = self.to_gaussian(state)
        return x, state

class BiTransformer(nn.Module):
    """As simple as I can make the model.
    """

    def __init__(self, config, num_tokens, dim):
        super().__init__()
        self.embed = TokenEmbedding(dim, num_tokens)
        self.attn = VanillaTransformerBlock(config, dim)
        self.norm = nn.LayerNorm(dim)
        self.to_gaussian = nn.Linear(dim, 2*dim, bias=False)
    
    def forward(self, x):
        x = self.attn(self.embed(x))
        x = self.to_gaussian(x)
        return x


def setup_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


torch.autograd.set_detect_anomaly(True)
def train(data, tokenizer, config):
    model = BlockRecurrentDecoder(config, len(tokenizer), config.d_model)
    model.to(device)
    posterior_model = BiTransformer(config, len(tokenizer), config.d_model)
    posterior_model.to(device)
    opt = Adam(model.parameters(), lr=1e-4)
    train_data = WikiDataset(data['train'])
    data_loader = DataLoader(train_data,  batch_size = config.batch_size, sampler = RandomSampler(train_data), pin_memory=True)

    for raw_batch in tqdm(data_loader):
        state = None
        article_batch = tokenizer(raw_batch, return_tensors='pt', padding=True)['input_ids']
        train_loss = 0
        # manually specify h0
        prev_state = model.h0.repeat(article_batch.size(0), 1, 1)
        for i, text in enumerate(long_sequence_splitter(article_batch, config.window_len-1)):
            opt.zero_grad()
            # add eos token so the low-level model knows when to stop
            text_eos = torch.cat([text, text.new_ones(text.size(0), 1)*tokenizer.eos_token_id], dim=-1)
            inputs = text_eos[:, :-1].cuda()
            targets = text_eos[:, 1:].cuda()
            # add bos token so our BERT can use it to summarize the block
            bos_text = torch.cat([text.new_ones(text.size(0), config.state_len)*tokenizer.bos_token_id, text], dim=-1)
            # run q(z|x)
            state_posterior = posterior_model(bos_text.cuda())[:, :config.state_len]
            qz_mean = state_posterior[:, :, :config.d_model]
            qz_logvar = state_posterior[:, :, :config.d_model]
            # use z ~ q(z|x) to generate p(x|z)
            eps = qz_mean.new(qz_mean.shape).normal_(0, 1)
            preds, state = model(inputs, qz_mean + (0.5*qz_logvar).exp()*eps)
            # reconstruction loss
            rec_loss = cross_entropy(preds.permute(0, 2, 1), targets)
            # prior matching loss KL(qz||pz)
            pz_mean = prev_state[:, :, config.d_model:]
            pz_logvar = prev_state[:, :, config.d_model:]
            prior_loss = ((pz_logvar-qz_logvar) + (qz_logvar.exp()+(qz_mean - pz_mean)**2)/pz_logvar.exp()).mean()

            prev_state = state
            loss = rec_loss+prior_loss
            loss.backward(retain_graph=True)
            train_loss += loss.item()
        opt.step()
        print(f'Train loss: {train_loss / i}')
        train_loss = 0



if __name__ == '__main__':
    device = 'cuda:0'
    data = load_dataset("wikipedia", "20220301.en")
    tokenizer = setup_tokenizer()
    config = OmegaConf.load('configs/base.yaml')
    train(data, tokenizer, config)
    