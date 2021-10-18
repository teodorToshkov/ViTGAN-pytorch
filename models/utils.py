import torch
from torch import nn
from torch.nn import Parameter
import numpy as np

# Improved Spectral Normalization (ISN)
# Reference code: https://github.com/koshian2/SNGAN
# When updating the weights, normalize the weights' norm to its norm at initialization.

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
		self.w_initalized = False

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
		if not self.w_initalized:
			self.w_init_sigma = np.array(sigma.expand_as(w).detach().cpu())
			self.w_initalized = True
		setattr(self.module, self.name, torch.tensor(self.w_init_sigma).to(sigma.device) * w / sigma.expand_as(w))

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