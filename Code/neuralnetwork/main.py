import torch.optim as optim
import torch.utils.data
from torch.optim import lr_scheduler

from Code.neuralnetwork.RGBD_PatchEncoder import *
from Code.neuralnetwork.RGBD_TripletLoss import *
from Code.neuralnetwork.RGBD_TripletNetwork import *
from Code.neuralnetwork.RGBD_TripletTrainer import *
from Code.neuralnetwork.RGBD_TripletLocalizationDataset import *

cuda = torch.cuda.is_available()
root = "../../Dataset"
triplet_train_dataset = RGBD_TripletLocalizationDataset(
	root,
	mode=RGBD_TripletLocalizationDataset.TRAINING,
	scale=0.8,
	shift=100,
	random_dist=30,
	num_features=3,
	detector_method="ORB"
)
triplet_valid_dataset = RGBD_TripletLocalizationDataset(
	root,
	mode=RGBD_TripletLocalizationDataset.VALIDATION,
	scale=0.8,
	shift=100,
	random_dist=30,
	num_features=3,
	detector_method="ORB"
)
batch_size = 64

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
triplet_train_loader = torch.utils.data.DataLoader(
	triplet_train_dataset, batch_size=batch_size, shuffle=False, **kwargs
)
triplet_valid_loader = torch.utils.data.DataLoader(
	triplet_valid_dataset, batch_size=batch_size, shuffle=False, **kwargs
)

embedding_net = RGBD_PatchEncoder()
model = RGBD_TripletNetwork(embedding_net)

if cuda:
	model.cuda()

alpha = 1.
loss_fn = RGBD_TripletLoss(alpha)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 100

fit(
	triplet_train_loader, triplet_valid_loader, model, loss_fn,
	optimizer, scheduler, n_epochs, cuda, log_interval
)

torch.save(model, "Code/neuralnetwork/model")
