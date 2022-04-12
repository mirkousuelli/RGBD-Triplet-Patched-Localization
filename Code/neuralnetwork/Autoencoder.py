from torch import nn
from neuralnetwork.Encoder import Encoder
from neuralnetwork.Decoder import Decoder


class Autoencoder(nn.Module):

	def __init__(self):
		super().__init__()
		# Convolutional section
		self.encoder = Encoder()
		self.decoder = Decoder()

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x
