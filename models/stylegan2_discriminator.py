from stylegan2_pytorch import stylegan2_pytorch
from .diffaugment import DiffAugment

# StyleGAN2 Discriminator

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