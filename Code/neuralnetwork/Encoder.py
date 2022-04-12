from torch import nn


class Encoder(nn.Module):

	def __init__(self):
		super().__init__()

		# 16 x 16 @ 4  (input shape)
		self.encoder_cnn = nn.Sequential(
			# 8 x 8 @ 8
			nn.Conv2d(4, 8, 3, stride=2, padding=0),
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
			nn.Conv2d(32, 64, 3, stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
		)

	def forward(self, x):
		x = self.encoder_cnn(x)
		return x
