from typing import Tuple, List

import os
import random
import numpy as np
from cv2 import KeyPoint
from torch.utils.data import Dataset

from Code.camera.Frame import Frame
from Code.camera.Action import Action
from Code.tools.Detector import Detector
from Code.utils.utils import *


class RGBDTripletLocalizationDataset(Dataset):
	"""
	RGB-D Triplet Localization Dataset as pytorch object being able to train
	a NN on it.
	"""

	# constants
	TRAINING = "Training"
	VALIDATION = "Validation"
	TESTING = "Testing"
	SUPPORT = [TRAINING, VALIDATION]

	def __init__(
		self,
		root,
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
		:type scale: int

		:param shift:
			Since the training could take a lot of time and closer images are
			identical, this parameter allows to skip through the scenes folders
			with a certain shift rate among frames.
		:type shift:

		:param random_dist:
			In order to randomize the reference-support frame pairs choice, we
			set a symmetrical bounds over the current frame which allows to pick
			a random number around it.
		:type random_dist: int
		"""
		super().__init__()

		# hyper-parameters
		self.root = root
		self.scale = scale
		self.shift = shift
		self.random_dist = random_dist
		self.num_features = num_features
		self.curr_mode = self.TRAINING

		# Detector initialization
		self.detector = Detector(self.num_features, detector_method)

		# initialize two source dictionary both for reference and support frames
		self.reference_dict = {}
		self.support_dict = {}

		# subdirectories populations for the dataset main folders
		for sub in os.listdir(self.root):
			# the reference has both Training, Validation, Testing
			self.reference_dict[sub] = {}

			# the support does not have Testing as Triplet training definition
			if sub in self.SUPPORT:
				self.support_dict[sub] = {}

		# list used as reference to store the training scene keys order
		self.training_order_keys = []
		self.curr_scene = 0

		# subdirectories populations for the dataset main folders scenes
		for sub in list(iter(self.reference_dict)):
			for scene in os.listdir(self.root + sub):
				# the reference has both Training, Validation, Testing
				self.reference_dict[sub][scene] = {}

				# the support does not have Testing as Triplet training
				# definition
				if sub in self.SUPPORT:
					self.support_dict[sub][scene] = {}

					# keep track of the training keys for the scenes
					if sub == self.TRAINING:
						self.training_order_keys.append(scene)

		# in order to maintain a stratified sampling procedure at training time
		# we set up a common maximum bound about the size of the frames per
		# scenes to be processed while learning in training
		self.max_common_size = 5000
		for sub in list(iter(self.reference_dict)):
			for scene in os.listdir(self.root + sub):
				# update the max common size based on the minimum one
				self.max_common_size = min(
					self.max_common_size,
					len(os.listdir(self.root + sub + '/' + scene + '/Colors'))
				)

		# populate all the subdirectories with scene indexes
		for sub in list(iter(self.reference_dict)):
			for scene in os.listdir(self.root + sub):
				# reference list initialization for each scene
				self.reference_dict[sub][scene] = []

				# doing the same for the support
				if sub in self.SUPPORT:
					self.support_dict[sub][scene] = []

				# adding the images of the scene based on the shift and scale
				# limits
				for num in range(1, int(self.max_common_size * self.scale),
				                 self.shift):
					# reference standard append done in straight order
					self.reference_dict[sub][scene].append(num)

					# reference non-standard append done in random order
					if sub in self.SUPPORT:
						# random lower bound
						lb = num - self.random_dist \
							if num - self.random_dist > 0 else 0

						# random upper bound
						ub = num + self.random_dist \
							if num + self.random_dist < self.max_common_size \
							else self.max_common_size

						# finally append by picking an image between the two
						# bounds
						self.support_dict[sub][scene].append(
							np.random.randint(lb, ub)
						)

	def __len__(self):
		"""Gets the dimension of the dataset.
		
		:return:
			The length in terms of number of items of the dataset.
		"""
		return self.max_common_size

	def __getitem__(self, index):
		"""Gets the item of the dataset at the specified index.
		
		:param index:
			Index of the item to get.
		 
		:return:
			The item at index idx of the dataset and its target.
		"""
		# pre-conditions
		assert 0 < index < self.__len__(), "Index out of bounds in the dataset!"

		# load the corresponding reference-support pair based on the index
		action = self.__load_item(index)

		# fetch the triplet patches
		patch_anchor, patch_pos, patch_neg = self.__get_triplet_coords(action)

		# update the scene
		self.__scene_update(index)

		# return triplet patches
		return (patch_anchor, patch_pos, patch_neg), []

	def set_mode(
		self,
		new_mode
	):
		"""
		Change the dataset mode between Training, Validation and Testing.

		:param new_mode:
			Must be 'Training', 'Validation', 'Testing'
		"""
		assert new_mode in [self.TRAINING, self.VALIDATION, self.TESTING], \
			"No existent mode selected!"
		self.curr_mode = new_mode

	def __scene_update(
		self,
		index
	):
		"""
		Scene update during training at the end of the scrolling;
		The scene will be updated based on the current random order.

		:param index:
			current index
		"""
		if index == self.__len__() - 1 and self.curr_mode == self.TRAINING:
			if self.curr_scene == self.training_order_keys[-1]:
				random.shuffle(self.training_order_keys)
				self.curr_scene = self.training_order_keys[0]
			else:
				self.curr_scene = self.training_order_keys[
					self.training_order_keys.index(self.curr_scene) + 1
				]

	def __load_item(self, index: int):
		"""Gets the action from the dataset based on the index.
		
		:param index:
			The index of the first frame to be considered.
			
		:return:
			The action at the specified index.
		"""
		# index retrieval through the dictionaries
		reference = self.reference_dict[self.curr_mode][self.curr_scene][index]
		support = self.support_dict[self.curr_mode][self.curr_scene][index]

		# frames initializations
		frame_reference = Frame(
			get_rgb_triplet_dataset_path(
				self.root, self.curr_mode, self.curr_scene, reference
			),
			get_depth_triplet_dataset_path(
				self.root, self.curr_mode, self.curr_scene, reference
			),
			get_pose_triplet_dataset_path(
				self.root, self.curr_mode, self.curr_scene
			),
			reference
		)
		frame_support = Frame(
			get_rgb_triplet_dataset_path(
				self.root, self.curr_mode, self.curr_scene, support
			),
			get_depth_triplet_dataset_path(
				self.root, self.curr_mode, self.curr_scene, support
			),
			get_pose_triplet_dataset_path(
				self.root, self.curr_mode, self.curr_scene
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

	def __get_triplet_coords(
		self,
		action: Action
	):
		"""Gets the coordinates of the triplet patches.
		
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
			triplets.append(np.array([key_point,
			                          second_keys[pos_idx],
			                          second_keys[neg_idx]]))

		# I convert the triplet list to a numpy array
		triplets = np.array(triplets)

		return self.__extract_patches(action, 8, triplets)

	def __extract_patches(self, action: Action,
	                      patch_side: int,
	                      triplets: np.ndarray):
		"""Extracts the patches given the triplet.
		
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
		patches = []
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

			triplet_patch = []
			for idx, coord in enumerate(triplet):
				patch = np.zeros((1 + 2 * patch_side, 1 + 2 * patch_side, 4))
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

						patch[y_off, x_off, 0] = color_img[yo, xo, 0]
						patch[y_off, x_off, 1] = color_img[yo, xo, 1]
						patch[y_off, x_off, 2] = color_img[yo, xo, 2]
						patch[y_off, x_off, 3] = depth_img[yo, xo]

				triplet_patch.append(patch)

			triplet_patch = np.array(triplet_patch)
			patches.append(triplet_patch)

		patches = np.array(patches)

		return patches
