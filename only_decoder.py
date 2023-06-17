# Simple Decoder Class definition
### (Option) Different split, test data?
from torch.nn.modules.dropout import Dropout
import torch.nn as nn
import torch
from typing import Tuple
import torch.nn.functional as F

class SingleAttentionHead(nn.Module):

    def __init__(self, head_size, config, device):
        super().__init__()
        self.n_embed = config["n_embed"]
        self.context_length = config["context_length"]
        self.dropout = config["dropout"]

        self.query = nn.Linear(self.n_embed, head_size, bias = False, device=device)
        self.key = nn.Linear(self.n_embed, head_size, bias = False, device=device)
        self.value = nn.Linear(self.n_embed, head_size, bias = False, device=device)
        self.triangle_matrix = torch.tril(torch.ones(self.context_length, self.context_length, device=device))
        self.head_dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        B, T, C = x.shape
        keys = self.key(x)
        query = self.query(x)

        affinities = query @ keys.transpose(-2, -1)  # Dot product, with transposition of T and C
        affinities *= C**(-.5)  # Normalization, to prevent softmax for skewing
        affinities = affinities.masked_fill(self.triangle_matrix[:T, :T] == 0, float('-inf'))
        affinities = F.softmax(affinities, dim=-1)
        affinities = self.head_dropout(affinities)

        v = self.value(x)
        return affinities @ v

class TransformerBlock(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        
        self.heads_num = config["att_head_num"]
        self.n_embed = config["n_embed"]
        self.dropout = config["dropout"]

        head_size = self.n_embed // self.heads_num
        self.attention_heads = nn.ModuleList([SingleAttentionHead(head_size, config, device) for _ in range(self.heads_num)])
        self.heads_projection = nn.Linear(self.n_embed, self.n_embed, device=device)
        self.heads_dropout = nn.Dropout(self.dropout)
        self.feed_forward = nn.Sequential(nn.Linear(self.n_embed, 4 * self.n_embed, device=device),
                                          nn.ReLU(),
                                          nn.Linear(4 * self.n_embed, self.n_embed, device=device),
                                          nn.Dropout(self.dropout))
        # '4 times n_embed' comes from the paper 'Attention is all you need' (as the whole transformer)
        self.layer_normalization = nn.LayerNorm(self.n_embed, device=device)
        self.layer_normalization2 = nn.LayerNorm(self.n_embed, device=device)

    def forward(self, x):
        x = self.layer_normalization(x)
        x = x + self.heads_dropout(self.heads_projection(torch.cat([att(x) for att in self.attention_heads], dim=-1)))
        x = self.layer_normalization2(x)
        x = x + self.feed_forward(x)
        return x


class OnlyDecoder(nn.Module):
    def __init__(self, config: dict, device):
        super().__init__()

        self.vocab_size = config["vocab_size"]
        self.n_embed = config["n_embed"]
        self.context_length = config["context_length"]
        self.head_num = config["att_head_num"]
        self.blocks_num = config["blocks_num"]

        self.device = device
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embed, device=device)
        self.pos_embedding_table = nn.Embedding(self.context_length, self.n_embed, device=device)
        self.lin = nn.Linear(self.n_embed, self.vocab_size, device=device)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(config, device) for _ in range(self.blocks_num)],
                                                nn.LayerNorm(self.n_embed, device=device))
        self.layer_normalization = nn.LayerNorm(self.n_embed, device=device)

    def forward(self, token_idx: int, targets=None):
        B, T = token_idx.shape
        token_embedding = self.token_embedding_table(token_idx)
        pos_embedding = self.pos_embedding_table(torch.arange(T, device=self.device))
        x = token_embedding + pos_embedding
        x = self.transformer_blocks(x)
        x = self.layer_normalization(x)
        logits = self.lin(x)

        if targets is None:
          loss = None
        else:
          B, T, C = logits.shape
          logits = logits.view(B*T, C)
          targets = targets.view(B*T)
          targets = targets.type(torch.LongTensor).to(self.device)
          loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate_new_text(self, idx, sym_limit: int) -> torch.Tensor:
        output = []
        for _ in range(sym_limit):
          idx = idx if idx.size(1) <= self.context_length else idx[:, -self.context_length:]
          logits, loss = self(idx)
          logits = logits[:, -1, :]
          probabilities = F.softmax(logits, dim=-1)
          idx_next = torch.multinomial(probabilities, num_samples=1) # Take best
          output.append(idx_next)
          idx = torch.cat((idx, idx_next), dim=1)
        return output