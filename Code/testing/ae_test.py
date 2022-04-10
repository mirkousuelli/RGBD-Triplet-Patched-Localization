from neuralnetwork.Autoencoder import Autoencoder
from torchsummary import summary

summary(Autoencoder(), (4, 17, 17))
