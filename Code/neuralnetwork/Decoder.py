from torch import nn


class Decoder(nn.Module):

	def __init__(self):
		super().__init__()

		# 1 x 1 @ 64  (input shape)
		self.decoder_cnn = nn.Sequential(
			# 2 x 2 @ 32
			nn.ConvTranspose2d(64, 32, 2, stride=2, output_padding=0),
			nn.BatchNorm2d(32),
			nn.ReLU(True),

			# 4 x 4 @ 16
			nn.ConvTranspose2d(32, 16, 3, stride=1, output_padding=0),
			nn.BatchNorm2d(16),
			nn.ReLU(True),

			# 8 x 8 @ 8
			nn.ConvTranspose2d(16, 8, 5, stride=1, output_padding=0),
			nn.BatchNorm2d(8),
			nn.ReLU(True),

			# 16 x 16 @ 4  (output shape)
			nn.ConvTranspose2d(8, 4, 3, stride=2, output_padding=0)
		)

	def forward(self, x):
		x = self.decoder_cnn(x)
		return x
