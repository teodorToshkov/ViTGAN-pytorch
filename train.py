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

from torch.utils.tensorboard import SummaryWriter

from utils import *
from diffaugment import *
from models import *

import argparse

def get_parser():
    # parse parameters
    parser = argparse.ArgumentParser(description='ViTGAN')
    parser.add_argument("--image_size", type=int, default=32,
                        help="Image Size")
    parser.add_argument("--patch_size", type=int, default=4,
                        help="Patch Size")
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="Dimensions of the seed")
    parser.add_argument("--hidden_features", type=int, default=384,
                        help="Image Size")
    parser.add_argument("--sln_paremeter_size", type=int, default=384,
                        help="Either equal to --hidden_features of 1")
    parser.add_argument("--depth", type=int, default=4,
                        help="Number of Transformer Block layers for both the Generator and the Discriminator")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of Attention heads for every Transformer Block layer")
    parser.add_argument("--combine_patch_embeddings", type=bool, default=False,
                        help="Generate an image from a single SIREN, instead of patch-by-patch")
    parser.add_argument("--combine_patch_embeddings_size", type=int, default=384,
                        help="Size of the combined image embedding")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--discriminator_type", type=str, default="vitgan",
                        help="\"vitgan\", \"cnn\", \"stylegan2\"")
    parser.add_argument("--batch_size_history_discriminator", type=bool, default=True,
                        help="Whether to use a loss, which tracks one sample from last batch_size number of batches")
    parser.add_argument("--lr", type=int, default=128,
                        help="Learning Rate")
    parser.add_argument("--beta1", type=int, default=0,
                        help="Adam beta1 parameter")
    parser.add_argument("--beta2", type=int, default=0.99,
                        help="Adam beta2 parameter")
    parser.add_argument("--epochs", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--lambda_bCR_real", type=int, default=10,
                        help="Batch size")
    parser.add_argument("--lambda_bCR_fake", type=int, default=10,
                        help="Batch size")
    parser.add_argument("--lambda_lossD_noise", type=int, default=0,
                        help="Batch size")
    parser.add_argument("--lambda_lossD_history", type=int, default=0,
                        help="Batch size")
    return parser

parser = get_parser()
params = parser.parse_args()

image_size = params.image_size
patch_size = params.patch_size
latent_dim = params.latent_dim # Size of z
hidden_features = params.hidden_features
depth = params.depth
num_heads = params.num_heads

combine_patch_embeddings = params.combine_patch_embeddings # Generate an image from a single SIREN, instead of patch-by-patch
combine_patch_embeddings_size = params.combine_patch_embeddings_size

sln_paremeter_size = params.sln_paremeter_size # either hidden_features or 1

batch_size = params.batch_size
out_features = 3 # The number of color channels

discriminator_type = params.discriminator_type # "vitgan", "cnn", "stylegan2"

lr = params.lr # Learning rate
beta = (params.beta1, params.beta2) # Adam oprimizer parameters for both the generator and the discriminator
batch_size_history_discriminator = params.batch_size_history_discriminator # Whether to use a loss, which tracks one sample from last batch_size number of batches
epochs = params.epochs # Number of epochs
lambda_bCR_real = params.lambda_bCR_real
lambda_bCR_fake = params.lambda_bCR_fake
lambda_lossD_noise = params.lambda_lossD_noise
lambda_lossD_history = params.lambda_lossD_history

writer = SummaryWriter(log_dir=f'runs/patch_size-{patch_size}_\
hidden_features-{hidden_features}_combine_patch_embeddings-{combine_patch_embeddings}_\
sln_paremeter_size-{sln_paremeter_size}_discriminator_type-{discriminator_type}_\
num_heads-num_heads_\
depth-{depth}')

if combine_patch_embeddings:
    out_patch_size = image_size
    combined_embedding_size = combine_patch_embeddings_size
else:
    out_patch_size = patch_size
    combined_embedding_size = hidden_features

siren_in_features = combined_embedding_size

siren_mapping_size = int(siren_in_features // 2)
B_gauss = torch.randn((siren_mapping_size, 2)) * 10


def fourier_input_mapping(x):
    x_proj = (2. * np.pi * x) @ B_gauss.t()
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

def fourier_pos_embedding():
    # Create input pixel coordinates in the unit square
    coords = np.linspace(-1, 1, out_patch_size, endpoint=True)
    pos = np.stack(np.meshgrid(coords, coords), -1)
    pos = torch.tensor(pos, dtype=torch.float)
    result = fourier_input_mapping(pos).reshape([out_patch_size**2, siren_in_features])
    return result.cuda()

def mix_hidden_and_pos(hidden):
    pos = fourier_pos_embedding()
    hidden = repeat(hidden, 'n h -> n p h', p = out_patch_size**2)

    # Combine the outputs from the Vision Transformer
    # with the Fourier position embeddings
    result = hidden + pos
    
    return result

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# Create the Generator
Generator = GeneratorViT(   patch_size=patch_size,
                            image_size=image_size,
                            latent_dim=latent_dim,
                            combine_patch_embeddings=combine_patch_embeddings,
                            combined_embedding_size=combined_embedding_size,
                            sln_paremeter_size=sln_paremeter_size,
                            num_heads = num_heads,
                            depth = depth,
                         ).cuda()

# Create the SIREN network
img_siren = Siren(in_features=siren_in_features, out_features=out_features, hidden_features=hidden_features,
                  hidden_layers=3, outermost_linear=True).cuda()

# Create the two types of discriminators
Discriminator = ViT(discriminator=True,
                            stride_size=patch_size*2,
                            n_classes=1, 
                            num_heads = num_heads,
                            depth = depth,
                    ).cuda()
cnn_discriminator = CNN().cuda()
stylegan2_discriminator = StyleGanDiscriminator(image_size=32).cuda()

os.makedirs("weights", exist_ok = True)
os.makedirs("samples", exist_ok = True)

# Loss function
criterion = nn.BCEWithLogitsLoss()

if discriminator_type == "cnn": discriminator = cnn_discriminator
elif discriminator_type == "stylegan2": discriminator = stylegan2_discriminator
elif discriminator_type == "vitgan": discriminator = Discriminator

params = list(Generator.parameters()) + list(img_siren.parameters())
optim_g = torch.optim.Adam(lr=lr, params=params, betas=beta)
optim_d = torch.optim.Adam(lr=lr, params=discriminator.parameters(), betas=beta)
ema = ExponentialMovingAverage(params, decay=0.995)

fixed_noise = torch.FloatTensor(np.random.normal(0, 1, (16, latent_dim))).cuda()

discriminator_f_img = torch.zeros([batch_size, 3, image_size, image_size]).cuda()

trainset_len = len(trainloader.dataset)

step = 0
for epoch in range(epochs):
    for batch_id, batch in enumerate(trainloader):
        step += 1

        # Train discriminator

        # Forward + Backward with real images
        r_img = batch[0].cuda()
        r_logit = discriminator(r_img).flatten()
        r_label = torch.ones(r_logit.shape[0]).cuda()

        lossD_real = criterion(r_logit, r_label)
        
        lossD_bCR_real = F.mse_loss(r_logit, discriminator(r_img, do_augment=False))

        # Forward + Backward with fake images
        latent_vector = torch.FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))).cuda()

        hidden = Generator(latent_vector)
        model_input = mix_hidden_and_pos(hidden).cuda()
        model_output, coords = img_siren(model_input)
        # convert model_output from [batch_size*num_patches, patch_size^2, out_features]
        f_img = model_output.view([-1, image_size, image_size, out_features])
        f_img = f_img.permute(0, 3, 1, 2).contiguous()

        f_label = torch.zeros(batch_size).cuda()
        # Save the a single generated image to the discriminator training data
        if batch_size_history_discriminator:
            discriminator_f_img[step % batch_size] = f_img[0].detach()
            f_logit_history = discriminator(discriminator_f_img).flatten()
            lossD_fake_history = criterion(f_logit_history, f_label)
        else: lossD_fake_history = 0
        # Train the discriminator on the images, generated only from this batch
        f_logit = discriminator(f_img).flatten()
        lossD_fake = criterion(f_logit, f_label)
        
        lossD_bCR_fake = F.mse_loss(f_logit, discriminator(f_img, do_augment=False))
        
        f_noise_input = torch.FloatTensor(np.random.rand(*f_img.shape)*2 - 1).cuda()
        f_noise_logit = discriminator(f_noise_input).flatten()
        lossD_noise = criterion(f_noise_logit, f_label)

        lossD = lossD_real * 0.5 +\
                lossD_fake * 0.5 +\
                lossD_fake_history * lambda_lossD_history +\
                lossD_noise * lambda_lossD_noise +\
                lossD_bCR_real * lambda_bCR_real +\
                lossD_bCR_fake * lambda_bCR_fake

        optim_d.zero_grad()
        lossD.backward()
        optim_d.step()
        
        # Train Generator

        latent_vector = torch.FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))).cuda()
        hidden = Generator(latent_vector)
        model_input = mix_hidden_and_pos(hidden).cuda()
        model_output, coords = img_siren(model_input)
        # convert model_output from [batch_size*num_patches, patch_size^2, out_features]
        f_img = model_output.view([-1, image_size, image_size, out_features])
        f_img = f_img.permute(0, 3, 1, 2)

        f_logit = discriminator(f_img).flatten()
        r_label = torch.ones(batch_size).cuda()
        lossG = criterion(f_logit, r_label)
        
        optim_g.zero_grad()
        lossG.backward()
        optim_g.step()
        ema.update()

        writer.add_scalar("Loss/Generator", lossG, step)
        writer.add_scalar("Loss/Dis(real)", lossD_real, step)
        writer.add_scalar("Loss/Dis(fake)", lossD_fake, step)
        writer.add_scalar("Loss/Dis(fake_history)", lossD_fake_history, step)
        writer.add_scalar("Loss/Dis(noise)", lossD_noise, step)
        writer.add_scalar("Loss/Dis(bCR_fake)", lossD_bCR_fake * lambda_bCR_fake, step)
        writer.add_scalar("Loss/Dis(bCR_real)", lossD_bCR_real * lambda_bCR_real, step)
        writer.flush()

        if batch_id%20 == 0:
            print(f'epoch {epoch}/{epochs}; batch {batch_id}/{int(trainset_len/batch_size)}')
            print(f'Generator: {"{:.3f}".format(float(lossG))}, '+\
                  f'Dis(real): {"{:.3f}".format(float(lossD_real))}, '+\
                  f'Dis(fake): {"{:.3f}".format(float(lossD_fake))}, '+\
                  f'Dis(fake_history): {"{:.3f}".format(float(lossD_fake_history))}, '+\
                  f'Dis(noise) {"{:.3f}".format(float(lossD_noise))}, '+\
                  f'Dis(bCR_fake): {"{:.3f}".format(float(lossD_bCR_fake * lambda_bCR_fake))}, '+\
                  f'Dis(bCR_real): {"{:.3f}".format(float(lossD_bCR_real * lambda_bCR_real))}')

            # Plot 8 randomly selected samples
            fig, axes = plt.subplots(1,8, figsize=(24,3))
            for i in range(8):
                j = np.random.randint(0, batch_size-1)
                model_output = model_output.view([-1, image_size**2, out_features])
                img = model_output[j].cpu().view(32,32,3).detach().numpy()
                img -= img.min()
                img /= img.max()
                axes[i].imshow(img)
            plt.show()

    # if step % sample_interval == 0:
    Generator.eval()
    hidden = Generator(fixed_noise)
    model_input = mix_hidden_and_pos(hidden).cuda()
    model_output, coords = img_siren(model_input)
    f_img = model_output.view([-1, image_size, image_size, out_features])
    vis = f_img.permute(0, 3, 1, 2)
    vis.detach().cpu()
    vis = make_grid(vis, nrow = 4, padding = 5, normalize = True)
    
    writer.add_image(f'Generated/epoch_{epoch}', vis)

    vis = T.ToPILImage()(vis)
    vis.save('samples/vis{:03d}.jpg'.format(epoch))
    Generator.train()
    print("Save sample to samples/vis{:03d}.jpg".format(epoch))

    # Save the checkpoints.
    torch.save(Generator.state_dict(), 'weights/Generator.pth')
    torch.save(discriminator.state_dict(), 'weights/discriminator.pth')
    print("Save model state.")

writer.close()