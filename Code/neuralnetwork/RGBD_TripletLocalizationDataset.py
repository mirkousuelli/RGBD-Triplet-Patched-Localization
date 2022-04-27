import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from camera.Frame import Frame
from camera.Action import Action
from tools.Detector import Detector
from utils.utils import *


class RGBD_TripletLocalizationDataset(Dataset):
	"""
	RGB-D Triplet Localization Dataset as pytorch object being able to train
	a NN on it.
	"""

	# constants
	TRAINING = "Training"
	VALIDATION = "Validation"
	TESTING = "Testing"
	SUPPORT = [TRAINING, VALIDATION]
	MODES = [TRAINING, VALIDATION, TESTING]

	def __init__(
			self,
			root,
			mode=TRAINING,
			scale=1.0,
			shift=10,
			random_dist=60,
			num_features=32,
			detector_method="ORB"
	):
		"""
		RGB-D Triplet Localization Dataset constructor.

		:param root:
			Dataset path string root
		:type root: str

		:param scale:
			In order to maintain a stratified sampling training it is assumed
			to bound the image scenes with the minimum populated ones. This
			parameter allows to further limit this amount by imposing a
			percentage on it (1.0 == 100% : use the maximum stratified sampling
			capability)
		:type scale: float

		:param shift:
			Since the training could take a lot of time and closer images are
			identical, this parameter allows to skip through the scenes folders
			with a certain shift rate among frames.
		:type shift: int

		:param random_dist:
			In order to randomize the reference-support frame pairs choice, we
			set a symmetrical bounds over the current frame which allows to pick
			a random number around it.
		:type random_dist: int
		"""
		# pre-conditions
		assert mode in self.MODES, "Error dataset mode! Choose among [%s]" %\
		                           ', '.join(map(str, self.MODES))
		super(RGBD_TripletLocalizationDataset).__init__()

		# hyper-parameters
		self.root = root
		self.mode = mode
		self.scale = scale
		self.shift = shift
		self.random_dist = random_dist
		self.num_features = num_features
		self.folder_len = 0

		# Detector initialization
		self.detector = Detector(self.num_features, detector_method)

		# initialize two source dictionary both for reference and support frames
		self.reference_dict = {}
		self.support_dict = {}

		# the reference has the mode folder (i.e. Training, Validation, Testing)
		self.reference_dict[self.mode] = {}

		# the support does not have Testing as Triplet training definition
		if self.mode in self.SUPPORT:
			self.support_dict[self.mode] = {}

		# list used as reference to store the training scene keys order
		self.order_keys = []

		for scene in os.listdir(self.root + '/' + self.mode):
			# the reference has both Training, Validation, Testing
			self.reference_dict[self.mode][scene] = {}

			# the support does not have Testing as Triplet training
			# definition
			if self.mode in self.SUPPORT:
				self.support_dict[self.mode][scene] = {}

				# keep track of the training keys for the scenes
				self.order_keys.append(scene)

		# setting the first order index key
		self.curr_scene_idx = self.order_keys[0]
		self.batch_scene_idx = self.curr_scene_idx

		# in order to maintain a stratified sampling procedure at training time
		# we set up a common maximum bound about the size of the frames per
		# scenes to be processed while learning in training
		self.max_common_size = 5000  # 5000 is an arbitrary high number
		for scene in os.listdir(self.root + '/' + self.mode):
			# update the max common size based on the minimum one
			self.max_common_size = min(
				self.max_common_size,
				len(os.listdir(self.root + '/' + self.mode + '/' + scene + '/Colors'))
			)

		# random pair matching reference-support
		for scene in os.listdir(self.root + '/' + self.mode):
			# reference list initialization for each scene
			self.reference_dict[self.mode][scene] = []

			# doing the same for the support
			if self.mode in self.SUPPORT:
				self.support_dict[self.mode][scene] = []

			# adding the images of the scene based on the shift and scale
			# limits
			for num in range(1, int(self.max_common_size * self.scale),
			                 self.shift):
				# reference standard append done in straight order
				self.reference_dict[self.mode][scene].append(num)

				# reference non-standard append done in random order
				if self.mode in self.SUPPORT:
					# random lower bound
					lb = num - self.random_dist \
						if num - self.random_dist > 0 else 0

					# random upper bound
					ub = num + self.random_dist \
						if num + self.random_dist < self.max_common_size \
						else self.max_common_size

					# finally append by picking an image between the two
					# bounds
					self.support_dict[self.mode][scene].append(
						np.random.randint(lb, ub)
					)

		self.folder_len = len(self.reference_dict[self.mode][self.curr_scene_idx])

	def __len__(self):
		"""
		Gets the dimension of the dataset.
		
		:return:
			The length in terms of number of items of the dataset.
		"""
		return self.folder_len

	def __getitem__(self, index):
		"""
		Gets the item of the dataset at the specified index.
		
		:param index:
			Index of the item to get.
		 
		:return:
			The item at index idx of the dataset and its target.
		"""
		# pre-conditions
		assert 0 <= index < self.__len__(), "Index out of bounds in the dataset!"

		# updated afterwards to be correctly visualized
		# NB. this index is just used for batch information
		self.batch_scene_idx = self.curr_scene_idx

		# load the corresponding reference-support pair based on the index
		action = self.__load_item(index)

		# update the scene
		self.__scene_update(index)

		# fetch and return triplet patches
		return self.__get_triplet(action), []

	def get_num_scenes(
		self
	):
		return len(self.order_keys)

	def __scene_update(
			self,
			index
	):
		"""
		Scene update during training at the end of the scrolling;
		The scene will be updated based on the current random order.

		:param index:
			current index of the scene frame
		"""
		# if the image index has reach the end
		if index >= self.__len__() - 1:

			# if the scene index correspond to the last one
			if self.curr_scene_idx == self.order_keys[-1]:

				# if we are in training mode
				if self.mode == self.TRAINING:

					# then shuffle the training order keys
					random.shuffle(self.order_keys)

				# set the current scene as the first element
				self.curr_scene_idx = self.order_keys[0]

			# otherwise if the scene index is not the last one
			else:

				# scroll the current index upwards
				self.curr_scene_idx = self.order_keys[
					self.order_keys.index(self.curr_scene_idx) + 1
				]

	def __load_item(self, index: int):
		"""
		Gets the action from the dataset based on the index.
		
		:param index:
			The index of the first frame to be considered.
			
		:return:
			The action at the specified index.
		"""
		# index retrieval through the dictionaries
		reference = self.reference_dict[self.mode][self.curr_scene_idx][index]
		support = self.support_dict[self.mode][self.curr_scene_idx][index]

		# frames initializations
		frame_reference = Frame(
			get_rgb_triplet_dataset_path(
				self.root, self.mode, self.curr_scene_idx, reference
			),
			get_depth_triplet_dataset_path(
				self.root, self.mode, self.curr_scene_idx, reference
			),
			get_pose_triplet_dataset_path(
				self.root, self.mode, self.curr_scene_idx
			),
			reference
		)
		frame_support = Frame(
			get_rgb_triplet_dataset_path(
				self.root, self.mode, int(self.curr_scene_idx), support
			),
			get_depth_triplet_dataset_path(
				self.root, self.mode, int(self.curr_scene_idx), support
			),
			get_pose_triplet_dataset_path(
				self.root, self.mode, int(self.curr_scene_idx)
			),
			support
		)

		# detect frames' key-points
		self.detector.detect_and_compute(frame_reference)
		self.detector.detect_and_compute(frame_support)

		# action pose difference estimation
		action = Action(frame_reference, frame_support)
		action.pose_difference()

		# from pose to roto-translation
		action.from_pose_to_rototrasl()

		# from roto-translation to fundamental matrix
		action.from_rototrasl_to_f_matrix()

		# return the final ready-to-be-used action
		return action

	def __get_triplet(
			self,
			action: Action
	):
		"""
		Gets the coordinates of the triplet patches.
		
		:param action:
			The action from which the coordinates of the patches must be
			retrieved.
		
		:return:
			A list of triplets containing the coordinates of Anchor, Positive
			and Negative in this order.
		"""
		# I get all the key-points of the first image and of the second image
		first_keys = [key.pt
		              for key in action.first.key_points]
		first_keys = [np.array([pt[0], pt[1], 1])
		              for pt in first_keys.copy()]
		first_keys = np.array(first_keys)

		second_keys = [key.pt
		               for key in action.second.key_points]
		second_keys = [np.array([pt[0], pt[1], 1])
		               for pt in second_keys.copy()]
		second_keys = np.array(second_keys)

		# I find all the transformed points using the fundamental matrix
		triplets = []
		for key_point in first_keys:
			x1Fx2 = [key_point @ action.f_matrix @ np.transpose(key2)
			         for key2 in second_keys]
			x1Fx2 = np.absolute(x1Fx2)
			pos_idx = np.argmin(x1Fx2)
			neg_idx = np.argmax(x1Fx2)

			# I add the triplet anchor-positive-negative to the triplets list
			triplets.append(
				np.array(
					[key_point,
					 second_keys[pos_idx],
					 second_keys[neg_idx]],
					dtype=np.int32
				)
			)

		# I convert the triplet list to a numpy array
		triplets = np.array(triplets)

		return self.__extract_triplet_patches(action, 8, triplets)

	@staticmethod
	def __extract_triplet_patches(
		action: Action,
		patch_side: int,
		triplets: np.ndarray
	):
		"""
		Extracts the patches given the triplet.
		
		:param action:
			The action from which the patches must be extracted.
		
		:param patch_side:
			The dimension of one patch side.
		
		:param triplets:
			The list of all the triplets from which patches must be extracted.
			Triplets must be ordered as Anchor-Positive-Negative.
		
		:return:
			The patches relative to the Anchor-Positive-Negative elements
			returned as a numpy array of shape (n_samples, 3, 2*patch_side + 1,
			2*patch_side + 1, 4)
		"""

		# I extract all the patches
		anchor_patches, positive_patches, negative_patches = [], [], []
		for triplet in triplets:
			# Get rgb and depth for both images
			first_rgbd = action.first.get_rgbd_image()
			first_color = np.asarray(first_rgbd.color)
			first_depth = np.asarray(first_rgbd.depth)
			second_rgbd = action.second.get_rgbd_image()
			second_color = np.asarray(second_rgbd.color)
			second_depth = np.asarray(second_rgbd.depth)
			w = first_color.shape[1]
			h = first_color.shape[0]

			for idx, coord in enumerate(triplet):
				patch = np.zeros((4, 1 + 2 * patch_side, 1 + 2 * patch_side))
				xc = coord[0]
				yc = coord[1]

				# Iterate taking care of border cases
				for x_off in range(2 * patch_side + 1):
					for y_off in range(2 * patch_side + 1):
						xo = int(max(0, min(xc - patch_side,
						                    w - 1 - 2 * patch_side)) + x_off)
						yo = int(max(0, min(yc - patch_side,
						                    h - 1 - 2 * patch_side)) + y_off)
						if idx == 0:
							color_img = first_color
							depth_img = first_depth
						else:
							color_img = second_color
							depth_img = second_depth

						patch[0, y_off, x_off] = color_img[yo, xo, 0]
						patch[1, y_off, x_off] = color_img[yo, xo, 1]
						patch[2, y_off, x_off] = color_img[yo, xo, 2]
						patch[3, y_off, x_off] = depth_img[yo, xo]

				if idx == 0:
					anchor_patches.append(torch.from_numpy(patch))
				elif idx == 1:
					positive_patches.append(torch.from_numpy(patch))
				elif idx == 2:
					negative_patches.append(torch.from_numpy(patch))

		return anchor_patches, positive_patches, negative_patches
