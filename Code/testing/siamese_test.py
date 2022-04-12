# from neuralnetwork.Siamese import Siamese
from neuralnetwork.Autoencoder import Autoencoder
from torchsummary import summary
from torchviz import make_dot

model = Autoencoder()
summary(model, (4, 17, 17))
make_dot(model)
