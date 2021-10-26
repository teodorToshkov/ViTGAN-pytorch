import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import repeat

from .multi_head import MultiHeadAttention

# ViTGAN Generator

class FullyConnectedLayer(nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
        **kwargs
    ):
        super().__init__()
        self.activation = activation
        if activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        self.weight = nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            if b is not None:
                x = x + b
            if self.activation != 'linear':
                x = self.activation(x)
        return x

def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

class MappingNetwork(nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws          = None,     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
        **kwargs
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c=None, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                assert z.shape[1] == self.z_dim
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                assert c.shape[1] == self.c_dim
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion=4, drop_p=0., bias=False):
        super().__init__(
            FullyConnectedLayer(emb_size, expansion * emb_size, activation='gelu', bias=False),
            nn.Dropout(drop_p),
            FullyConnectedLayer(expansion * emb_size, emb_size, bias=False),
        )

# Self-Modulated LayerNorm

class SLN(nn.Module):
    def __init__(self, input_size, parameter_size=None, **kwargs):
        super().__init__()
        if parameter_size == None:
            parameter_size = input_size
        assert(input_size == parameter_size or parameter_size == 1)
        self.input_size = input_size
        self.parameter_size = parameter_size
        self.ln = nn.LayerNorm(input_size)
        self.gamma = FullyConnectedLayer(input_size, parameter_size, bias=False)
        self.beta = FullyConnectedLayer(input_size, parameter_size, bias=False)
        # self.gamma = nn.Linear(input_size, parameter_size, bias=False)
        # self.beta = nn.Linear(input_size, parameter_size, bias=False)

    def forward(self, hidden, w):
        assert(hidden.size(-1) == self.parameter_size and w.size(-1) == self.parameter_size)
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
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList([GeneratorTransformerEncoderBlock(**kwargs) for _ in range(depth)])
    
    def forward(self, hidden, w):
        for i in range(self.depth):
            hidden = self.blocks[i](hidden, w)
        return hidden

# SIREN

#Code for SIREN is taken from https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb

class ModulatedLinear(nn.Module):
    def __init__(self, in_channels, out_channels, style_size, bias=False, demodulation=True, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.style_size = style_size
        self.scale = 1 / np.sqrt(in_channels)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, 1)
        )
        self.modulation = None
        if self.style_size != self.in_channels:
            self.modulation = FullyConnectedLayer(style_size, in_channels, bias=False)
        self.demodulation = demodulation

    def forward(self, input, style):
        batch_size = input.shape[0]

        if self.style_size != self.in_channels:
            style = self.modulation(style)
        style = style.view(batch_size, 1, self.in_channels, 1)
        weight = self.scale * self.weight * style

        if self.demodulation:
            demod = torch.rsqrt(weight.pow(2).sum([2]) + 1e-8)
            weight = weight * demod.view(batch_size, self.out_channels, 1, 1)

        weight = weight.view(
            batch_size * self.out_channels, self.in_channels, 1
        )

        img_size = input.size(1)
        input = input.reshape(1, batch_size * self.in_channels, img_size)
        out = F.conv1d(input, weight, groups=batch_size)
        out = out.view(batch_size, img_size, self.out_channels)

        return out

class ResLinear(nn.Module):
    def __init__(self, in_channels, out_channels, style_size, bias=False, **kwargs):
        super().__init__()
        self.linear = FullyConnectedLayer(in_channels, out_channels, bias=False)
        self.style = FullyConnectedLayer(style_size, in_channels, bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.style_size = style_size

    def forward(self, input, style):
        x = input + self.style(style).unsqueeze(1)
        x = self.linear(x)
        return x

class ConLinear(nn.Module):
    def __init__(self, ch_in, ch_out, is_first=False, bias=True, **kwargs):
        super().__init__()
        self.conv = nn.Linear(ch_in, ch_out, bias=bias)
        if is_first:
            nn.init.uniform_(self.conv.weight, -np.sqrt(9 / ch_in), np.sqrt(9 / ch_in))
        else:
            nn.init.uniform_(self.conv.weight, -np.sqrt(3 / ch_in), np.sqrt(3 / ch_in))

    def forward(self, x):
        return self.conv(x)

class SinActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

class LFF(nn.Module):
    def __init__(self, hidden_size, **kwargs):
        super().__init__()
        self.ffm = ConLinear(2, hidden_size, is_first=True)
        self.activation = SinActivation()

    def forward(self, x):
        x = x
        x = self.ffm(x)
        x = self.activation(x)
        return x

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, style_size, bias=False,
                 is_first=False, omega_0=30, weight_modulation=True, **kwargs):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.weight_modulation = weight_modulation
        if weight_modulation:
            self.linear = ModulatedLinear(in_features, out_features, style_size=style_size, bias=bias, **kwargs)
        else:
            self.linear = ResLinear(in_features, out_features, style_size=style_size, bias=bias, **kwargs)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                if self.weight_modulation:
                    self.linear.weight.uniform_(-1 / self.in_features, 
                                                1 / self.in_features)
                else:
                    self.linear.linear.weight.uniform_(-1 / self.in_features, 
                                                        1 / self.in_features) 
            else:
                if self.weight_modulation:
                    self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                                np.sqrt(6 / self.in_features) / self.omega_0)
                else:
                    self.linear.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                                        np.sqrt(6 / self.in_features) / self.omega_0) 
        
    def forward(self, input, style):
        return torch.sin(self.omega_0 * self.linear(input, style))
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_size, hidden_layers, out_features, style_size, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30., weight_modulation=True, bias=False, **kwargs):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_size, style_size,
                                  is_first=True, omega_0=first_omega_0,
                                  weight_modulation=weight_modulation, **kwargs))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_size, hidden_size, style_size,
                                      is_first=False, omega_0=hidden_omega_0,
                                      weight_modulation=weight_modulation, **kwargs))

        if outermost_linear:
            if weight_modulation:
                final_linear = ModulatedLinear(hidden_size, out_features,
                                               style_size=style_size, bias=bias, **kwargs)
            else:
                final_linear = ResLinear(hidden_size, out_features, style_size=style_size, bias=bias, **kwargs)
            
            with torch.no_grad():
                if weight_modulation:
                    final_linear.weight.uniform_(-np.sqrt(6 / hidden_size) / hidden_omega_0, 
                                                np.sqrt(6 / hidden_size) / hidden_omega_0)
                else:
                    final_linear.linear.weight.uniform_(-np.sqrt(6 / hidden_size) / hidden_omega_0, 
                                                np.sqrt(6 / hidden_size) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_size, out_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      weight_modulation=weight_modulation, **kwargs))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords, style):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        # output = self.net(coords, style)
        output = coords
        for layer in self.net:
            output = layer(output, style)
        return output

class GeneratorViT(nn.Module):
    def __init__(self,
                style_mlp_layers=8,
                patch_size=4,
                latent_dim=32,
                hidden_size=384,
                sln_paremeter_size=1,
                image_size=32,
                depth=4,
                combine_patch_embeddings=False,
                combined_embedding_size=1024,
                forward_drop_p=0.,
                bias=False,
                out_features=3,
                out_patch_size=4,
                weight_modulation=True,
                siren_hidden_layers=1,
                **kwargs):
        super().__init__()
        self.hidden_size = hidden_size

        self.mlp = MappingNetwork(z_dim=latent_dim, c_dim=0, w_dim=hidden_size, num_layers=style_mlp_layers, w_avg_beta=None)

        num_patches = int(image_size//patch_size)**2
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.image_size = image_size
        self.combine_patch_embeddings = combine_patch_embeddings
        self.combined_embedding_size = combined_embedding_size
        self.out_patch_size = out_patch_size
        self.out_features = out_features

        self.pos_emb = nn.Parameter(torch.randn(num_patches, hidden_size))
        self.transformer_encoder = GeneratorTransformerEncoder(depth,
                                                               hidden_size=hidden_size,
                                                               sln_paremeter_size=sln_paremeter_size,
                                                               drop_p=forward_drop_p,
                                                               forward_drop_p=forward_drop_p,
                                                               **kwargs)
        self.sln = SLN(hidden_size, parameter_size=sln_paremeter_size)
        if combine_patch_embeddings:
            self.to_single_emb = nn.Sequential(
                FullyConnectedLayer(num_patches*hidden_size, combined_embedding_size, bias=bias, activation='gelu'),
                nn.Dropout(forward_drop_p),
            )

        self.lff = LFF(self.hidden_size)

        self.siren_in_features = combined_embedding_size if combine_patch_embeddings else self.hidden_size
        self.siren = Siren(in_features=self.siren_in_features, out_features=out_features,
                           style_size=self.siren_in_features, hidden_size=self.hidden_size, bias=bias,
                           hidden_layers=siren_hidden_layers, outermost_linear=True, weight_modulation=weight_modulation, **kwargs)

        self.num_patches_x = int(image_size//self.out_patch_size)

    def fourier_input_mapping(self, x):
        return self.lff(x)

    def fourier_pos_embedding(self, device):
        # Create input pixel coordinates in the unit square
        coords = np.linspace(-1, 1, self.out_patch_size, endpoint=True)
        pos = np.stack(np.meshgrid(coords, coords), -1)
        pos = torch.tensor(pos, dtype=torch.float, device=device)
        result = self.fourier_input_mapping(pos).reshape([self.out_patch_size**2, self.hidden_size])
        return result.to(device)

    def repeat_pos(self, hidden):
        pos = self.fourier_pos_embedding(hidden.device)
        result = repeat(pos, 'p h -> n p h', n = hidden.shape[0])
        
        return result

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
        
        pos = self.repeat_pos(hidden)

        result = self.siren(pos, hidden)

        model_output_1 = result.view([-1, self.num_patches_x, self.num_patches_x, self.out_patch_size, self.out_patch_size, self.out_features])
        model_output_2 = model_output_1.permute([0, 1, 3, 2, 4, 5])
        model_output = model_output_2.reshape([-1, self.image_size**2, self.out_features])
        
        return model_output
