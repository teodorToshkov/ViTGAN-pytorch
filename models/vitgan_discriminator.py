import torch
from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange

from .diffaugment import DiffAugment
from .multi_head import MultiHeadAttention
from .utils import spectral_norm

# Discriminator

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, stride_size=4, emb_size=384, image_size=32, batch_size=64):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            spectral_norm(nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=stride_size)),
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