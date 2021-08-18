import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn import Parameter
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt

import time

import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import make_grid

from torch_ema import ExponentialMovingAverage

from stylegan2_pytorch import stylegan2_pytorch

from utils import *
from diffaugment import *

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=384, num_heads=4, dropout=0, discriminator=False):
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

# Generator

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion=4, drop_p=0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class SLN(nn.Module):
    def __init__(self, input_size, parameter_size=None):
        super().__init__()
        if parameter_size == None:
            parameter_size = input_size
        self.ln = nn.LayerNorm(input_size)
        self.gamma = nn.Linear(input_size, parameter_size)
        self.beta = nn.Linear(input_size, parameter_size)

    def forward(self, hidden, w):
        gamma = self.gamma(w).unsqueeze(1)
        beta = self.beta(w).unsqueeze(1)
        ln = self.ln(hidden)
        return gamma * ln + beta

class GeneratorTransformerEncoderBlock(nn.Module):
    def __init__(self,
                 hidden_size=384,
                 sln_paremeter_size=384,
                 drop_p=0.,
                 forward_expansion=4,
                 forward_drop_p=0.,
                 **kwargs):
        super().__init__()
        self.sln = SLN(hidden_size, parameter_size=sln_paremeter_size)
        self.msa = MultiHeadAttention(hidden_size, **kwargs)
        self.dropout = nn.Dropout(drop_p)
        self.feed_forward = FeedForwardBlock(hidden_size, expansion=forward_expansion, drop_p=forward_drop_p)

    def forward(self, hidden, w):
        res = hidden
        hidden = self.sln(hidden, w)
        hidden = self.msa(hidden)
        hidden = self.dropout(hidden)
        hidden += res

        res = hidden
        hidden = self.sln(hidden, w)
        self.feed_forward(hidden)
        hidden = self.dropout(hidden)
        hidden += res
        return hidden

class GeneratorTransformerEncoder(nn.Module):
    def __init__(self, depth=4, **kwargs):
        self.depth = depth
        self.blocks = [GeneratorTransformerEncoderBlock(**kwargs).cuda() for _ in range(depth)]
        super().__init__()
    
    def forward(self, hidden, w):
        for i in range(self.depth):
            hidden = self.blocks[i](hidden, w)
        return hidden

class GeneratorViT(nn.Module):
    def __init__(self,
                patch_size=4,
                latent_dim=32,
                hidden_size=384,
                sln_paremeter_size=1,
                image_size=32,
                depth=4,
                combine_patch_embeddings=False,
                combined_embedding_size=1024,
                **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 384),
            nn.GELU(),
        )
        num_patches = int(image_size//patch_size)**2
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.image_size = image_size
        self.combine_patch_embeddings = combine_patch_embeddings
        self.combined_embedding_size = combined_embedding_size

        self.pos_emb = nn.Parameter(torch.randn(num_patches, hidden_size))
        self.transformer_encoder = GeneratorTransformerEncoder(depth, hidden_size=hidden_size, **kwargs)
        self.sln = SLN(hidden_size, parameter_size=sln_paremeter_size).cuda()
        self.to_single_emb = nn.Sequential(
            nn.Linear(num_patches*hidden_size, combined_embedding_size),
            nn.GELU(),
        )

    def forward(self, z):
        w = self.mlp(z)
        pos = repeat(torch.sin(self.pos_emb), 'n e -> b n e', b=z.shape[0])
        hidden = self.transformer_encoder(pos, w)

        if self.combine_patch_embeddings:
            # Output [batch_size, combined_embedding_size]
            hidden = self.sln(hidden, w).view((z.shape[0], -1))
            hidden = self.to_single_emb(hidden)
        else:
            # Output [batch_size*num_patches, hidden_size]
            hidden = self.sln(hidden, w).view((-1, self.hidden_size))
        
        return hidden

# SIREN
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

# Discriminator
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, stride_size=4, emb_size=384, image_size=32, batch_size=64):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            spectral_norm(nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=stride_size)).cuda(),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        num_patches = ((image_size-patch_size+stride_size) // stride_size) **2 + 1
        self.positions = nn.Parameter(torch.randn(num_patches, emb_size))
        self.batch_size = batch_size

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += torch.sin(self.positions)
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class DiscriminatorTransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size=384,
                 drop_p=0.,
                 forward_expansion=4,
                 forward_drop_p=0.,
                 **kwargs):
        super().__init__(
                ResidualAdd(nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, **kwargs),
                    nn.Dropout(drop_p)
                )),
                ResidualAdd(nn.Sequential(
                    nn.LayerNorm(emb_size),
                    nn.Sequential(
                        spectral_norm(nn.Linear(emb_size, forward_expansion * emb_size)),
                        nn.GELU(),
                        nn.Dropout(forward_drop_p),
                        spectral_norm(nn.Linear(forward_expansion * emb_size, emb_size)),
                    ),
                    nn.Dropout(drop_p)
                )
            ))

class DiscriminatorTransformerEncoder(nn.Sequential):
    def __init__(self, depth=4, **kwargs):
        super().__init__(*[DiscriminatorTransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=384, class_size_1=4098, class_size_2=1024, class_size_3=512, n_classes=10):
        super().__init__(
            nn.LayerNorm(emb_size),
            spectral_norm(nn.Linear(emb_size, class_size_1)),
            nn.GELU(),
            spectral_norm(nn.Linear(class_size_1, class_size_2)),
            nn.GELU(),
            spectral_norm(nn.Linear(class_size_2, class_size_3)),
            nn.GELU(),
            spectral_norm(nn.Linear(class_size_3, n_classes)),
            nn.GELU(),
        )

    def forward(self, x):
        # Take only the cls token outputs
        x = x[:, 0, :]
        return super().forward(x)

class ViT(nn.Sequential):
    def __init__(self,     
                in_channels=3,
                patch_size=4,
                stride_size=4,
                emb_size=384,
                image_size=32,
                depth=4,
                n_classes=1,
                diffaugment='color,translation,cutout',
                **kwargs):
        self.diffaugment = diffaugment
        super().__init__(
            PatchEmbedding(in_channels, patch_size, stride_size, emb_size, image_size),
            DiscriminatorTransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes=n_classes)
        )
    
    def forward(self, img, do_augment=True):
        if do_augment:
            img = DiffAugment(img, policy=self.diffaugment)
        return super().forward(img)

class CNN(nn.Sequential):
    def __init__(self,
                diffaugment='color,translation,cutout',
                **kwargs):
        self.diffaugment = diffaugment
        super().__init__(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),
            nn.Linear(256*4*4,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,1)
        )
    
    def forward(self, img, do_augment=True):
        if do_augment:
            img = DiffAugment(img, policy=self.diffaugment)
        return super().forward(img)

class StyleGanDiscriminator(stylegan2_pytorch.Discriminator):
    def __init__(self,
                diffaugment='color,translation,cutout',
                **kwargs):
        self.diffaugment = diffaugment
        super().__init__(**kwargs)
    def forward(self, img, do_augment=True):
        if do_augment:
            img = DiffAugment(img, policy=self.diffaugment)
        out, _ = super().forward(img)
        return out