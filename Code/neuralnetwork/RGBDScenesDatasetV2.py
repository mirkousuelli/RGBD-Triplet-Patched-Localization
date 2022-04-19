from typing import Tuple

import numpy as np
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
		pass
	
	def __get_triplet_coords(self, action: Action,
							 feature) -> Tuple:
		"""Gets the coordinates of the triplet patches.
		
		:param action:
			The action from which the coordinates of the patches must be
			retrieved.
		
		:param feature:
			The feature from which we want to compute the positive and negative
			examples for the triplet loss.
		
		:return:
			A tuple containing the coordinates of Anchor, Positive and Negative
			in this order.
		"""
		pass
	
	def __extract_patch(self, action: Action,
						patch_side: int,
						anchor: np.ndarray,
						positive: np.ndarray,
						negative: np.ndarray) -> Tuple:
		"""Extracts the patches given the triplet.
		
		:param action:
			The action from which the patches must be extracted.
		
		:param patch_side:
			The dimension of one patch side.
		
		:param anchor:
			Coordinates of the anchor patch.
		
		:param positive:
			Coordinates of the positive patch.
		
		:param negative:
			Coordinates of the negative patch.
		
		:return:
			The patches relative
		"""
		pass
