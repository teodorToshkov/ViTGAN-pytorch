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

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def l2normalize(v, eps=1e-4):
	return v / (v.norm() + eps)

class spectral_norm(nn.Module):
	def __init__(self, module, name='weight', power_iterations=1):
		super().__init__()
		self.module = module
		self.name = name
		self.power_iterations = power_iterations
		if not self._made_params():
			self._make_params()
		self.w_init_sigma = None

	def _update_u_v(self):
		u = getattr(self.module, self.name + "_u")
		v = getattr(self.module, self.name + "_v")
		w = getattr(self.module, self.name + "_bar")

		height = w.data.shape[0]
		_w = w.view(height, -1)
		for _ in range(self.power_iterations):
			v = l2normalize(torch.matmul(_w.t(), u))
			u = l2normalize(torch.matmul(_w, v))

		sigma = u.dot((_w).mv(v))
		if type(self.w_init_sigma) != np.ndarray:
			self.w_init_sigma = np.array(sigma.expand_as(w).detach().cpu())
		setattr(self.module, self.name, torch.tensor(self.w_init_sigma).cuda() * w / sigma.expand_as(w))

	def _made_params(self):
		try:
			getattr(self.module, self.name + "_u")
			getattr(self.module, self.name + "_v")
			getattr(self.module, self.name + "_bar")
			return True
		except AttributeError:
			return False

	def _make_params(self):
		w = getattr(self.module, self.name)

		height = w.data.shape[0]
		width = w.view(height, -1).data.shape[1]

		u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
		v = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
		u.data = l2normalize(u.data)
		v.data = l2normalize(v.data)
		w_bar = Parameter(w.data)

		del self.module._parameters[self.name]
		self.module.register_parameter(self.name + "_u", u)
		self.module.register_parameter(self.name + "_v", v)
		self.module.register_parameter(self.name + "_bar", w_bar)

	def forward(self, *args):
		self._update_u_v()
		return self.module.forward(*args)