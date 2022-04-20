from typing import Tuple

import numpy as np
from cv2 import DMatch, KeyPoint
from torch.utils.data import Dataset

from camera.Action import Action


class RGBDScenesDatasetV2(Dataset):
	"""RGBDScenesDatasetV2 pytorch object to be able to train a NN on it."""
	
	def __init__(self):
		super().__init__()

	def __len__(self):
		"""Gets the dimension of the dataset.
		
		:return:
			The length in terms of number of items of the dataset.
		"""
		pass
	
	def __getitem__(self, idx):
		"""Gets the item of the dataset at the specified index.
		
		:param idx:
			Index of the item to get.
		 
		:return:
			The item at index idx of the dataset and its target.
		"""
		pass

	def __load_item(self, idx: int,
					shift_window: int = 60,
					num_shift: float = 0.8):
		"""Gets the action from the dataset based on the index.
		
		:param idx:
			The index of the first frame to be considered.
		
		:param shift_window:
			The dimension of the shift window to use to draw a random frame to
			be used as second image.
		
		# TODO: penso che non sia un nome molto intuitivo, vedi tu
		:param num_shift:
			The fraction of shifts to perform on a scene set.
			
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
		num_samples = num_samples if num_samples < len(matches) else len(matches)
		selected_features = rng.choice(len(matches), num_samples, replace=False)
		features = [action.first.key_points[matches[x].queryIdx] for x in selected_features]
		
		return self.__get_triplet_coords(action,
										 features)
	
	def __get_triplet_coords(self, action: Action,
							 features: list[KeyPoint]):
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
