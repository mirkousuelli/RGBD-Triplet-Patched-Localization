from typing import Tuple, List

import os
import numpy as np
from cv2 import KeyPoint
from torch.utils.data import Dataset

from Code.camera import Action


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

		# root reference
		self.root = root
		self.scale = scale
		self.shift = shift
		self.random_dist = random_dist

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

				# adding the images of the scene based on the shift and scale limits
				for num in range(1, int(self.max_common_size * self.scale), self.shift):
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

						# finally append by picking an image between the two bounds
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
		patch_anchor, patch_pos, patch_neg = [], [], []  # self.__extract_patch()

		# return triplet patches
		return (patch_anchor, patch_pos, patch_neg), []

	def __load_item(self, idx: int):
		"""Gets the action from the dataset based on the index.
		
		:param idx:
			The index of the first frame to be considered.
			
		:return:
			The action at the specified index.
		"""
		pass

	def __get_fundamental_matrix(self, action: Action) -> np.ndarray:
		"""Gets the fundamental matrix from the given action.
		
		:param action:
			The action on which we must compute the fundamental matrix.
		
		:return:
			The fundamental matrix as ndarray of shape (3,3).
		"""
		pass

	def __feature_detection(self, action: Action,
	                        num_samples: int):
		"""Select the features to be used for training.
		
		To extract the features, a pseudo-random number generator with seed is
		used such that the experiments over this dataset are reproducible.
		
		:param action:
			The action from which the features are extracted.
		
		:param num_samples:
			The number of features to extract from the action.
		
		:return:
			The feature extracted for which we need to get the triplet patches.
		"""
		if num_samples < 1:
			raise ValueError("The number of samples must be greater than 0.")

		rng = np.random.default_rng(22)
		matches = action.links_inliers
		num_samples = num_samples if num_samples < len(matches) else len(
			matches)
		selected_features = rng.choice(len(matches), num_samples, replace=False)
		features = [action.first.key_points[matches[x].queryIdx] for x in
		            selected_features]

		return self.__get_triplet_coords(action,
		                                 features)

	def __get_triplet_coords(self, action: Action,
	                         features: List[KeyPoint]):
		"""Gets the coordinates of the triplet patches.
		
		:param action:
			The action from which the coordinates of the patches must be
			retrieved.
		
		:param features:
			The features from which we want to compute the positive and negative
			examples for the triplet loss.
		
		:return:
			A list of triplets containing the coordinates of Anchor, Positive
			and Negative in this order.
		"""
		# I get all the keypoints of the first image and of the second image
		first_keys = [key.pt
		              for key in features]
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

		return self.__extract_patch(action,
		                            8,
		                            triplets)

	def __extract_patch(self, action: Action,
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
