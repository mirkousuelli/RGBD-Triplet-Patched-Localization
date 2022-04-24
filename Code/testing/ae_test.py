from Code.neuralnetwork.RGBD_PatchEncoder import RGBD_PatchEncoder
from torchsummary import summary

summary(RGBD_PatchEncoder(), (4, 17, 17))
