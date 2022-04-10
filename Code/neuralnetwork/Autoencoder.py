from torch import nn
from neuralnetwork.Encoder import Encoder
from neuralnetwork.Decoder import Decoder


class Autoencoder(nn.Module):

	def __init__(self):
		super().__init__()

		# Convolutional section
		self.encoder = Encoder()
		self.latent = nn.Conv2d(64, 64, 3, stride=1, padding=1)
		self.decoder = Decoder()

	def forward(self, x):
		x = self.encoder(x)
		x = self.latent(x)
		x = self.decoder(x)
		return x
