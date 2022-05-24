from torch import nn


class RGBD_PatchEncoder(nn.Module):
	"""
	RGB-D Patch Encoder
	Convolutional Neural Network which takes as input a 4@16x16 path from
	an RGB-D image and encodes an embedding of the input batch expressed as
	a latent vector.
	"""

	def __init__(
		self
	):
		"""
		Constructor composed of a CNN-Encoder and a Latent layer.
		"""
		super(RGBD_PatchEncoder, self).__init__()

		# 16 x 16 @ 4  (input shape)
		self.encoder_cnn = nn.Sequential(

			# 8 x 8 @ 8
			nn.Conv2d(3, 8, 3, stride=2, padding=0),
			nn.BatchNorm2d(8),
			nn.ReLU(True),

			# 4 x 4 @ 16
			nn.Conv2d(8, 16, 3, stride=2, padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU(True),

			# 2 x 2 @ 32
			nn.Conv2d(16, 32, 3, stride=2, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(True),

			# 1 x 1 @ 64  (output shape)
			nn.Conv2d(32, 64, 3, stride=2, padding=1)
		)

		# latent layer vector
		self.latent = nn.Sequential(
			nn.Linear(in_features=64, out_features=64), nn.ReLU(True)
		)

	def forward(
		self,
		x
	):
		"""
		:param x:
			Patch RGB-D image of size [4 @ 17 x 17] as input

		:return:
			Latent embedded representation
		"""
		x = self.encoder_cnn(x)
		x = x.view(x.size()[0], -1)
		x = self.latent(x)

		# 1 x 64
		return x
