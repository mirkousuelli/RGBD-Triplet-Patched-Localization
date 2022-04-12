import torch
from torch import nn, Tensor

from neuralnetwork.Encoder import Encoder


class Siamese(nn.Module):
	"""

	"""

	def __init__(self):
		super(Siamese, self).__init__()
		self.encoder = Encoder()

	def forward_encoder(self, x) -> Tensor:
		x = self.encoder(x)
		return x

	def forward(self, x1, x2):
		f1 = self.forward_encoder(x1)
		f2 = self.forward_encoder(x2)
		out = torch.cosine_similarity(f1, f2)
		return out
