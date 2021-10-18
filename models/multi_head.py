import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F

from .utils import spectral_norm

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=384, num_heads=4, dropout=0, discriminator=False, **kwargs):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.discriminator = discriminator
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        if self.discriminator:
            self.qkv = spectral_norm(self.qkv)
            self.projection = spectral_norm(self.projection)
        
    def forward(self, x, mask=None):
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        if self.discriminator:
            # calculate L2-distances
            energy = torch.cdist(queries.contiguous(), keys.contiguous(), p=2)
        else:
            # sum up over the last axis
            energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out