import torch
from torch import nn
import torch.nn.functional as F
import os

import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import make_grid

from torch_ema import ExponentialMovingAverage

from torch.utils.tensorboard import SummaryWriter
import wandb

from models import CNN, ViT, StyleGanDiscriminator, GeneratorViT, CNNGenerator

import argparse

def get_parser():
    # parse parameters
    parser = argparse.ArgumentParser(description='ViTGAN')
    parser.add_argument("--image_size", type=int, default=32,
                        help="Image Size")
    parser.add_argument("--style_mlp_layers", type=int, default=8,
                        help="Style Mapping network depth")
    parser.add_argument("--patch_size", type=int, default=4,
                        help="Patch Size")
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="Dimensions of the seed")
    parser.add_argument("--dropout_p", type=float, default=0.,
                        help="Dropout rate")
    parser.add_argument("--bias", type=bool, default=True,
                        help="Whether to use bias or not")
    parser.add_argument("--weight_modulation", type=bool, default=True,
                        help="Whether to use weight modulation or not")
    parser.add_argument("--demodulation", type=bool, default=False,
                        help="Whether to use weight demodulation or not")
    parser.add_argument("--siren_hidden_layers", type=int, default=1,
                        help="Number of hidden layers for the SIREN network")
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
    parser.add_argument("--generator_type", type=str, default="vitgan",
                        help="\"vitgan\", \"cnn\"")
    parser.add_argument("--discriminator_type", type=str, default="vitgan",
                        help="\"vitgan\", \"cnn\", \"stylegan2\"")
    parser.add_argument("--batch_size_history_discriminator", type=bool, default=True,
                        help="Whether to use a loss, which tracks one sample from last batch_size number of batches")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning Rate for the Generator")
    parser.add_argument("--lr_dis", type=float, default=0.001,
                        help="Learning Rate for the Discriminator")
    parser.add_argument("--beta1", type=float, default=0,
                        help="Adam beta1 parameter")
    parser.add_argument("--beta2", type=float, default=0.99,
                        help="Adam beta2 parameter")
    parser.add_argument("--epochs", type=int, default=400,
                        help="Number of epocks")
    parser.add_argument("--lambda_bCR_real", type=int, default=10,
                        help="lambda_bCR_real")
    parser.add_argument("--lambda_bCR_fake", type=int, default=10,
                        help="lambda_bCR_fake")
    parser.add_argument("--lambda_lossD_noise", type=int, default=0,
                        help="lambda_lossD_noise")
    parser.add_argument("--lambda_lossD_history", type=int, default=0,
                        help="lambda_lossD_history")
    parser.add_argument("--lambda_diversity_penalty", type=int, default=0,
                        help="lambda_diversity_penalty")
    return parser

parser = get_parser()
params = parser.parse_args()

image_size = params.image_size
style_mlp_layers = params.style_mlp_layers
patch_size = params.patch_size
latent_dim = params.latent_dim # Size of z
hidden_size = params.hidden_size
depth = params.depth
num_heads = params.num_heads

dropout_p = params.dropout_p
bias = params.bias
weight_modulation = params.weight_modulation
demodulation = params.demodulation
siren_hidden_layers = params.siren_hidden_layers

combine_patch_embeddings = params.combine_patch_embeddings # Generate an image from a single SIREN, instead of patch-by-patch
combine_patch_embeddings_size = params.combine_patch_embeddings_size

sln_paremeter_size = params.sln_paremeter_size # either hidden_size or 1

batch_size = params.batch_size
device = params.device
out_features = 3 # The number of color channels

generator_type = params.generator_type # "vitgan", "cnn"
discriminator_type = params.discriminator_type # "vitgan", "cnn", "stylegan2"

lr = params.lr # Learning rate
lr_dis = params.lr_dis # Learning rate
beta = (params.beta1, params.beta2) # Adam oprimizer parameters for both the generator and the discriminator
batch_size_history_discriminator = params.batch_size_history_discriminator # Whether to use a loss, which tracks one sample from last batch_size number of batches
epochs = params.epochs # Number of epochs
lambda_bCR_real = params.lambda_bCR_real
lambda_bCR_fake = params.lambda_bCR_fake
lambda_lossD_noise = params.lambda_lossD_noise
lambda_lossD_history = params.lambda_lossD_history
lambda_diversity_penalty = params.lambda_diversity_penalty

experiment_folder_name = f'runs/lr-{lr}_\
lr_dis-{lr_dis}_\
bias-{bias}_\
demod-{demodulation}_\
sir_n_layer-{siren_hidden_layers}_\
w_mod-{weight_modulation}_\
patch_s-{patch_size}_\
st_mlp_l-{style_mlp_layers}_\
hid_size-{hidden_size}_\
comb_patch_emb-{combine_patch_embeddings}_\
sln_par_s-{sln_paremeter_size}_\
dis_type-{discriminator_type}_\
gen_type-{generator_type}_\
n_head-{num_heads}_\
depth-{depth}_\
drop_p-{dropout_p}_\
l_bCR_r-{lambda_bCR_real}_\
l_bCR_f-{lambda_bCR_fake}_\
l_D_noise-{lambda_lossD_noise}_\
l_D_his-{lambda_lossD_history}\
'
writer = SummaryWriter(log_dir=experiment_folder_name)

wandb.init(project='ViTGAN-pytorch')
config = wandb.config
config.image_size = image_size
config.bias = bias
config.demodulation = demodulation
config.siren_hidden_layers = siren_hidden_layers
config.weight_modulation = weight_modulation
config.style_mlp_layers = style_mlp_layers
config.patch_size = patch_size
config.latent_dim = latent_dim
config.hidden_size = hidden_size
config.depth = depth
config.num_heads = num_heads

config.dropout_p = dropout_p

config.combine_patch_embeddings = combine_patch_embeddings
config.combine_patch_embeddings_size = combine_patch_embeddings_size

config.sln_paremeter_size = sln_paremeter_size

config.batch_size = batch_size
config.device = device
config.out_features = out_features

config.generator_type = generator_type
config.discriminator_type = discriminator_type

config.lr = lr
config.lr_dis = lr_dis
config.beta1 = beta[0]
config.beta2 = beta[1]
config.batch_size_history_discriminator = batch_size_history_discriminator
config.epochs = epochs
config.lambda_bCR_real = lambda_bCR_real
config.lambda_bCR_fake = lambda_bCR_fake
config.lambda_lossD_noise = lambda_lossD_noise
config.lambda_lossD_history = lambda_lossD_history
config.lambda_diversity_penalty = lambda_diversity_penalty

if combine_patch_embeddings:
    out_patch_size = image_size
    combined_embedding_size = combine_patch_embeddings_size
else:
    out_patch_size = patch_size
    combined_embedding_size = hidden_size

siren_in_features = combined_embedding_size

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0., 0., 0.), (1., 1., 1.))
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# Diversity Loss

def diversity_loss(images):
    num_images_to_calculate_on = 10
    num_pairs = num_images_to_calculate_on * (num_images_to_calculate_on - 1) // 2

    scale_factor = 5

    loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=True)
    i = 0
    for a_id in range(num_images_to_calculate_on):
        for b_id in range(a_id+1, num_images_to_calculate_on):
            img_a = images[a_id]
            img_b = images[b_id]
            img_a_l2 = torch.norm(img_a)
            img_b_l2 = torch.norm(img_b)
            img_a, img_b = img_a.flatten(), img_b.flatten()

            # print(img_a_l2, img_b_l2, img_a.shape, img_b.shape)

            a_b_loss = scale_factor * (img_a.t() @ img_b) / (img_a_l2 * img_b_l2)
            # print(a_b_loss)
            loss = loss + torch.sigmoid(a_b_loss)
            i += 1
    loss = loss.sum() / num_pairs
    return loss

# Normal distribution init weight

def init_normal(m):
    if type(m) == nn.Linear:
        # y = m.in_features
        # m.weight.data.normal_(0.0,1/np.sqrt(y))
        if 'weight' in m.__dict__.keys():
            m.weight.data.normal_(0.0,1)
        # m.bias.data.fill_(0)

# Experiments

if generator_type == "vitgan":
    # Create the Generator
    Generator = GeneratorViT(   patch_size=patch_size,
                                image_size=image_size,
                                style_mlp_layers=style_mlp_layers,
                                latent_dim=latent_dim,
                                hidden_size=hidden_size,
                                combine_patch_embeddings=combine_patch_embeddings,
                                combined_embedding_size=combined_embedding_size,
                                sln_paremeter_size=sln_paremeter_size,
                                num_heads=num_heads,
                                depth=depth,
                                forward_drop_p=dropout_p,
                                bias=bias,
                                weight_modulation=weight_modulation,
                                siren_hidden_layers=siren_hidden_layers,
                                demodulation=demodulation,
                            ).to(device)
                            
    # use the modules apply function to recursively apply the initialization
    Generator.apply(init_normal)

    num_patches_x = int(image_size//out_patch_size)

    if os.path.exists(f'{experiment_folder_name}/weights/Generator.pth'):
        Generator = torch.load(f'{experiment_folder_name}/weights/Generator.pth')

    wandb.watch(Generator)

elif generator_type == "cnn":
    cnn_generator = CNNGenerator(hidden_size=hidden_size, latent_dim=latent_dim).to(device)

    cnn_generator.apply(init_normal)

    if os.path.exists(f'{experiment_folder_name}/weights/cnn_generator.pth'):
        cnn_generator = torch.load(f'{experiment_folder_name}/weights/cnn_generator.pth')

    wandb.watch(cnn_generator)

# Create the three types of discriminators
if discriminator_type == "vitgan":
    Discriminator = ViT(discriminator=True,
                            patch_size=patch_size*2,
                            stride_size=patch_size,
                            n_classes=1, 
                            num_heads=num_heads,
                            depth=depth,
                            forward_drop_p=dropout_p,
                    ).to(device)
            
    Discriminator.apply(init_normal)
    
    if os.path.exists(f'{experiment_folder_name}/weights/discriminator.pth'):
        Discriminator = torch.load(f'{experiment_folder_name}/weights/discriminator.pth')

    wandb.watch(Discriminator)

elif discriminator_type == "cnn":
    cnn_discriminator = CNN().to(device)

    cnn_discriminator.apply(init_normal)

    if os.path.exists(f'{experiment_folder_name}/weights/discriminator.pth'):
        cnn_discriminator = torch.load(f'{experiment_folder_name}/weights/discriminator.pth')

    wandb.watch(cnn_discriminator)

elif discriminator_type == "stylegan2":
    stylegan2_discriminator = StyleGanDiscriminator(image_size=32).to(device)

    # stylegan2_discriminator.apply(init_normal)

    if os.path.exists(f'{experiment_folder_name}/weights/discriminator.pth'):
        stylegan2_discriminator = torch.load(f'{experiment_folder_name}/weights/discriminator.pth')

    wandb.watch(stylegan2_discriminator)

# Training

os.makedirs(f"{experiment_folder_name}/weights", exist_ok = True)
os.makedirs(f"{experiment_folder_name}/samples", exist_ok = True)

# Loss function
criterion = nn.BCEWithLogitsLoss()

if discriminator_type == "cnn": discriminator = cnn_discriminator
elif discriminator_type == "stylegan2": discriminator = stylegan2_discriminator
elif discriminator_type == "vitgan": discriminator = Discriminator

if generator_type == "cnn":
    params = cnn_generator.parameters()
else:
    params = Generator.parameters()
optim_g = torch.optim.Adam(lr=lr, params=params, betas=beta)
optim_d = torch.optim.Adam(lr=lr_dis, params=discriminator.parameters(), betas=beta)
ema = ExponentialMovingAverage(params, decay=0.995)

fixed_noise = torch.FloatTensor(np.random.normal(0, 1, (16, latent_dim))).to(device)

discriminator_f_img = torch.zeros([batch_size, 3, image_size, image_size]).to(device)

trainset_len = len(trainloader.dataset)

step = 0
for epoch in range(epochs):
    for batch_id, batch in enumerate(trainloader):
        step += 1

        # Train discriminator

        # Forward + Backward with real images
        r_img = batch[0].to(device)
        r_logit = discriminator(r_img).flatten()
        r_label = torch.ones(r_logit.shape[0]).to(device)

        lossD_real = criterion(r_logit, r_label)
        
        lossD_bCR_real = F.mse_loss(r_logit, discriminator(r_img, do_augment=False))

        # Forward + Backward with fake images
        latent_vector = torch.FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))).to(device)

        if generator_type == "vitgan":
            f_img = Generator(latent_vector)
            f_img = f_img.reshape([-1, image_size, image_size, out_features])
            f_img = f_img.permute(0, 3, 1, 2)
        else:
            model_output = cnn_generator(latent_vector)
            f_img = model_output
            
        assert f_img.size(0) == batch_size, f_img.shape
        assert f_img.size(1) == out_features, f_img.shape
        assert f_img.size(2) == image_size, f_img.shape
        assert f_img.size(3) == image_size, f_img.shape

        f_label = torch.zeros(batch_size).to(device)
        # Save the a single generated image to the discriminator training data
        if batch_size_history_discriminator:
            discriminator_f_img[step % batch_size] = f_img[0].detach()
            f_logit_history = discriminator(discriminator_f_img).flatten()
            lossD_fake_history = criterion(f_logit_history, f_label)
        else: lossD_fake_history = 0
        # Train the discriminator on the images, generated only from this batch
        f_logit = discriminator(f_img.detach()).flatten()
        lossD_fake = criterion(f_logit, f_label)
        
        lossD_bCR_fake = F.mse_loss(f_logit, discriminator(f_img, do_augment=False))
        
        f_noise_input = torch.FloatTensor(np.random.rand(*f_img.shape)*2 - 1).to(device)
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

        if generator_type == "vitgan":
            f_img = Generator(latent_vector)
            f_img = f_img.reshape([-1, image_size, image_size, out_features])
            f_img = f_img.permute(0, 3, 1, 2)
        else:
            model_output = cnn_generator(latent_vector)
            f_img = model_output
        
        assert f_img.size(0) == batch_size
        assert f_img.size(1) == out_features
        assert f_img.size(2) == image_size
        assert f_img.size(3) == image_size

        f_logit = discriminator(f_img).flatten()
        r_label = torch.ones(batch_size).to(device)
        lossG_main = criterion(f_logit, r_label)
        
        lossG_diversity = diversity_loss(f_img) * lambda_diversity_penalty
        lossG = lossG_main + lossG_diversity
        
        optim_g.zero_grad()
        lossG.backward()
        optim_g.step()
        ema.update()

        writer.add_scalar("Loss/Generator", lossG_main, step)
        writer.add_scalar("Loss/Gen(diversity)", lossG_diversity, step)
        writer.add_scalar("Loss/Dis(real)", lossD_real, step)
        writer.add_scalar("Loss/Dis(fake)", lossD_fake, step)
        writer.add_scalar("Loss/Dis(fake_history)", lossD_fake_history, step)
        writer.add_scalar("Loss/Dis(noise)", lossD_noise, step)
        writer.add_scalar("Loss/Dis(bCR_fake)", lossD_bCR_fake * lambda_bCR_fake, step)
        writer.add_scalar("Loss/Dis(bCR_real)", lossD_bCR_real * lambda_bCR_real, step)
        writer.flush()

        wandb.log({
            'Generator': lossG_main,
            'Gen(diversity)': lossG_diversity,
            'Dis(real)': lossD_real,
            'Dis(fake)': lossD_fake,
            'Dis(fake_history)': lossD_fake_history,
            'Dis(noise)': lossD_noise,
            'Dis(bCR_fake)': lossD_bCR_fake * lambda_bCR_fake,
            'Dis(bCR_real)': lossD_bCR_real * lambda_bCR_real
        })

        if batch_id%20 == 0:
            print(f'epoch {epoch}/{epochs}; batch {batch_id}/{int(trainset_len/batch_size)}')
            print(f'Generator: {"{:.3f}".format(float(lossG_main))}, '+\
                  f'Gen(diversity): {"{:.3f}".format(float(lossG_diversity))}, '+\
                  f'Dis(real): {"{:.3f}".format(float(lossD_real))}, '+\
                  f'Dis(fake): {"{:.3f}".format(float(lossD_fake))}, '+\
                  f'Dis(fake_history): {"{:.3f}".format(float(lossD_fake_history))}, '+\
                  f'Dis(noise) {"{:.3f}".format(float(lossD_noise))}, '+\
                  f'Dis(bCR_fake): {"{:.3f}".format(float(lossD_bCR_fake * lambda_bCR_fake))}, '+\
                  f'Dis(bCR_real): {"{:.3f}".format(float(lossD_bCR_real * lambda_bCR_real))}')

            # Plot 8 randomly selected samples
            fig, axes = plt.subplots(1,8, figsize=(24,3))
            output = f_img.permute(0, 2, 3, 1)
            for i in range(8):
                j = np.random.randint(0, batch_size-1)
                img = output[j].cpu().view(32,32,3).detach().numpy()
                img -= img.min()
                img /= img.max()
                axes[i].imshow(img)
            plt.show()

    # if step % sample_interval == 0:
    if generator_type == "vitgan":
        Generator.eval()
        # img_siren.eval()
        vis = Generator(fixed_noise)
        vis = vis.reshape([-1, image_size, image_size, out_features])
        vis = vis.permute(0, 3, 1, 2)
    else:
        model_output = cnn_generator(fixed_noise)
        vis = model_output

    assert vis.shape[0] == fixed_noise.shape[0], f'vis.shape[0] is {vis.shape[0]}, but should be {fixed_noise.shape[0]}'
    assert vis.shape[1] == out_features, f'vis.shape[1] is {vis.shape[1]}, but should be {out_features}'
    assert vis.shape[2] == image_size, f'vis.shape[2] is {vis.shape[2]}, but should be {image_size}'
    assert vis.shape[3] == image_size, f'vis.shape[3] is {vis.shape[3]}, but should be {image_size}'
    
    vis.detach().cpu()
    vis = make_grid(vis, nrow = 4, padding = 5, normalize = True)
    writer.add_image(f'Generated/epoch_{epoch}', vis)
    wandb.log({'examples': wandb.Image(vis)})

    vis = T.ToPILImage()(vis)
    vis.save(f'{experiment_folder_name}/samples/vis{epoch}.jpg')
    if generator_type == "vitgan":
        Generator.train()
        # img_siren.train()
    else:
        cnn_generator.train()
    print(f"Save sample to {experiment_folder_name}/samples/vis{epoch}.jpg")

    # Save the checkpoints.
    if generator_type == "vitgan":
        torch.save(Generator, f'{experiment_folder_name}/weights/Generator.pth')
        # torch.save(img_siren, f'{experiment_folder_name}/weights/img_siren.pth')
    elif generator_type == "cnn":
        torch.save(cnn_generator, f'{experiment_folder_name}/weights/cnn_generator.pth')
    torch.save(discriminator, f'{experiment_folder_name}/weights/discriminator.pth')
    print("Save model state.")

writer.close()
