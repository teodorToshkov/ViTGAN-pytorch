# ViTGAN-pytorch
A PyTorch implementation of [VITGAN: Training GANs with Vision Transformers](https://arxiv.org/pdf/2107.04589v1.pdf)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kJJw6BYW01HgooCZ2zUDt54e1mXqITXH?usp=sharing)

## TODO:
1.   [x] Use vectorized L2 distance in attention for **Discriminator**
2.   [x] Overlapping Image Patches
2.   [x] DiffAugment
3.   [x] Self-modulated LayerNorm (SLN)
4.   [x] Implicit Neural Representation for Patch Generation
5.   [x] ExponentialMovingAverage (EMA)
6.   [x] Balanced Consistency Regularization (bCR)
7.   [x] Improved Spectral Normalization (ISN)
8.   [x] Equalized Learning Rate
9.   [x] Weight Modulation

## Dependencies

- Python3
- einops
- pytorch_ema
- stylegan2-pytorch
- tensorboard
- wandb

``` bash
pip install einops
pip install git+https://github.com/fadel/pytorch_ema
pip install stylegan2-pytorch
pip install tensorboard
pip install wandb
```

## **TLDR:**

Train the model with the proposed parameters:

``` bash
python train.py
```

Tensorboard

``` bash
tensorboard --logdir runs/
```

***

The following parameters are the parameters, proposed in the paper for the CIFAR-10 dataset:

``` bash
python train.py \
--image_size 32 \
--patch_size 4 \
--latent_dim 32 \
--hidden_features 384 \
--sln_paremeter_size 384 \
--depth 4 \
--num_heads 4 \
--combine_patch_embeddings false \
--batch_size 128 \
--discriminator_type "stylegan2" \
--batch_size_history_discriminator false \
--lr 0.002 \
--epochs 200 \
--lambda_bCR_real 10 \
--lambda_bCR_fake 10 \
--lambda_lossD_noise 0 \
--lambda_lossD_history 0 \
```

## Implementation Details

### Generator

The Generator follows the following architecture:

![ViTGAN Generator architecture](https://drive.google.com/uc?export=view&id=1XaCVOLq8Bvg-I3qM-bugNZcjIW5L7XTO)

For debugging purposes, the Generator is separated into a Vision Transformer (ViT) model and a SIREN model.

Given a seed, the dimensionality of which is controlled by ```latent_dim```, the ViT model creates an embedding for each of the patches of the final image. Those embeddings are fed to a SIREN network, combined with a Fourier Position Encoding \([Jupyter Notebook](https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb)\). It outputs the patches of the image, which are stitched together.

The ViT part of the Generator differs from a standard Vision Transformer in the following ways:
- The input to the Transformer consists only of the position embeddings
- Self-Modulated Layer Norm (SLN) is used in place of LayerNorm
- There is no classification head

SLN is the only place, where the seed is inputted to the network. <br/>
SLN consists of a regular LayerNorm, the result of which is multiplied by ```gamma``` and added to ```beta```. <br/>
Both ```gamma``` and ```beta``` are calculated using a fully connected layer, different for each place, SLN is applied. <br/>
The input dimension to each of those fully connected is equal to ```hidden_dimension``` and the output dimension is equal to ```hidden_dimension```.

#### SIREN

A description of SIREN:
\[[Blog Post](https://tech.fusic.co.jp/posts/2021-08-03-what-are-sirens/)\] \[[Paper](https://arxiv.org/pdf/2006.09661.pdf)\] \[[Colab Notebook](https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb)\]

In contrast to regular SIREN, the desired output is not a single image. For this purpose, the patch embedding is combined to a position embedding.

The positional encoding, used in ViTGAN is the Fourier Position Encoding, the code for which was taken from here: \([Jupyter Notebook](https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb)\)

In my implementation, the input to the SIREN is the sum of a patch embedding and a position embedding.

#### Weight Modulation

Weight Modulation usually consists of a modulation and a demodulation module. After testing the network, I concluded that **demodulation is not used in ViTGAN**.

My implementation of the weight modulation is heavily based on [CIPS](https://github.com/saic-mdal/CIPS/blob/main/model/blocks.py#L173). I have adjusted it to work for a fully-connected network, using a 1D convolution. The reason for using 1D convolution, instead of a linear layer is the groups term, which optimizes the performance by a factor of batch_size.

Each SIREN layer consists of a sinsin activation, applied to a weight modulation layer. The size of the input, the hidden and the output layers in a SIREN network could vary. Thus, in case the input size differs from the size of the patch embedding, I define an additional fully-connected layer, which converts the patch embedding to the appropriate size.

### Discriminator

The Discriminator follows the following architecture:

![ViTGAN Discriminator architecture](https://drive.google.com/uc?export=view&id=1LK-WLwNGXqAhJ44MAexSHOyPkyiGapys)

The ViTGAN Discriminator is mostly a standard Vision Transformer network, with the following modifications:
- DiffAugment
- Overlapping Image Patches
- Use vectorized L2 distance in attention for **Discriminator**
- Improved Spectral Normalization (ISN)
- Balanced Consistency Regularization (bCR)

#### DiffAugment

For implementating DiffAugment, I used the code below: <br/>
\[[GitHub](https://github.com/mit-han-lab/data-efficient-gans/blob/master/DiffAugment-stylegan2-pytorch/DiffAugment_pytorch.py)\] \[[Paper](https://arxiv.org/pdf/2006.10738.pdf)\]

#### Overlapping Image Patches

Creation of the overlapping image patches is implemented with the use of a convolution layer.

#### Use vectorized L2 distance in attention for **Discriminator**

\[[Paper](https://arxiv.org/pdf/2006.04710.pdf)\]

#### Improved Spectral Normalization (ISN)

The ISN implementation is based on the following implementation of Spectral Normalization: <br/>
\[[GitHub](https://github.com/koshian2/SNGAN/blob/117fbb19ac79bbc561c3ccfe285d6890ea0971f9/models/core_layers.py#L9)\]
\[[Paper](https://arxiv.org/abs/1802.05957)\]

#### Balanced Consistency Regularization (bCR)

Zhengli Zhao, Sameer Singh, Honglak Lee, Zizhao Zhang, Augustus Odena, Han Zhang; Improved Consistency Regularization for GANs; AAAI 2021
\[[Paper](https://arxiv.org/pdf/2002.04724.pdf)\]

## References
SIREN: [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/pdf/2006.09661.pdf) <br/>
Vision Transformer: \[[Blog Post](https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632)\] <br/>
L2 distance attention: [The Lipschitz Constant of Self-Attention](https://arxiv.org/pdf/2006.04710.pdf) <br/>
Spectral Normalization reference code: \[[GitHub](https://github.com/koshian2/SNGAN/blob/117fbb19ac79bbc561c3ccfe285d6890ea0971f9/models/core_layers.py#L9)\] \[[Paper](https://arxiv.org/abs/1802.05957)\] <br/>
Diff Augment: \[[GitHub](https://github.com/mit-han-lab/data-efficient-gans/blob/master/DiffAugment-stylegan2-pytorch/DiffAugment_pytorch.py)\] \[[Paper](https://arxiv.org/pdf/2006.10738.pdf)\] <br/>
Fourier Position Embedding: \[[Jupyter Notebook](https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb)\] <br/>
Exponential Moving Average: \[[GitHub](https://github.com/fadel/pytorch_ema)\] <br/>
Balanced Concictancy Regularization (bCR): \[[Paper](https://arxiv.org/pdf/2002.04724.pdf)\] <br/>
SyleGAN2 Discriminator: \[[GitHub](https://github.com/lucidrains/stylegan2-pytorch/blob/1a789d186b9697571bd6bbfa8bb1b9735bb42a0c/stylegan2_pytorch/stylegan2_pytorch.py#L627)\] <br/>
