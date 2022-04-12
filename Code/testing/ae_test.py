from neuralnetwork.Encoder import Encoder
from neuralnetwork.Siamese import Siamese
from torchsummary import summary

summary(Siamese(), [(4, 17, 17), (4, 17, 17)])
