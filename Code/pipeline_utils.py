from typing import Tuple

import numpy as np
import torch
from scipy import spatial
from torch import Tensor
from torch.autograd import Variable

from camera.Action import Action
from neuralnetwork.RGBD_TripletNetwork import RGBD_TripletNetwork


class LatentMatchScore(object):
	"""Class used to enclose matches score between latent vectors.
	"""
	SCORING_METHODS = ["euclidean", "cosine"]
	
	def __init__(self, first_latents: np.ndarray,
				 second_latents: np.ndarray,
				 first_idx: int,
				 second_idx: int,
				 eps: float = 0.0001):
		"""Initialize class.

		:param first_latents:
			Array of latent vectors of first patches.

		:param second_latents:
			Array of latent vectors of second patches.

		:param first_idx:
			Index of the first latent vector.

		:param second_idx:
			Index of the second latent vector.

		:param eps:
			Minimum amount of distance between latent vectors.
		"""
		if eps == 0:
			raise ValueError("Eps cannot be zero.")
		
		self.score = -np.inf
		self.first_idx = first_idx
		self.second_idx = second_idx
		self.eps = eps
		self.__first_latents = first_latents.copy()
		self.__second_latents = second_latents.copy()
	
	def compute_score(self, method: str = "euclidean") -> None:
		"""Computes the score between the two latent vectors.

		:param method:
			Method used to compute the score of the match.

		:return:
			None
		"""
		if method not in self.SCORING_METHODS:
			raise ValueError(
				"The method must be one of %s" % self.SCORING_METHODS)
		
		if method == "euclidean":
			vector_1 = self.__first_latents[self.first_idx]
			vector_2 = self.__second_latents[self.second_idx]
			self.score = np.linalg.norm(vector_2 - vector_1)
			if self.score == 0:
				self.score = self.eps
		elif method == "cosine":
			vector_1 = self.__first_latents[self.first_idx]
			vector_2 = self.__second_latents[self.second_idx]
			self.score = spatial.distance.cosine(vector_1, vector_2)
			if self.score == 0:
				self.score = self.eps

def get_features_from_merged(action: Action) -> Tuple[list, list]:
	"""From a merged action, it retrieves the features of first and second frame.
	
	:param action:
		The action from which features must be extracted.
	
	:return:
		The lists of the features of the first and second frames, in this order.
	"""
	matches = action.links
	features_1 = [action.first.key_points[matches[x].queryIdx]
				  for x in range(len(matches)) ]
	features_2 = [action.second.key_points[matches[x].trainIdx]
				  for x in range(len(matches)) ]
	return features_1, features_2

def get_key_points_from_features(features_1: list,
								 features_2: list) -> Tuple[np.ndarray, np.ndarray]:
	"""Extract the key points from the features.
	
	:param features_1:
		The list of the first features.
		
	:param features_2:
		The list of the second features.
	
	:return:
		The lists of the key points of the first and second frames, in this order.
	"""
	first_key_points = np.array(
		[np.array([key.pt[0], key.pt[1], 1]) for key in features_1]
	)
	second_key_points = np.array(
		[np.array([key.pt[0], key.pt[1], 1]) for key in features_2]
	)
	return first_key_points, second_key_points

def get_semantic_patches(action: Action,
						 key_points_1: np.ndarray,
						 key_points_2: np.ndarray,
						 patch_side: int) -> Tuple[np.ndarray, np.ndarray]:
	"""Get patches from the key points of an image.
	
	:param action:
		The action from which patches will be extracted. It must be merged.
		
	:param key_points_1:
		The array of the key points of the first frame.
	 
	:param key_points_2:
		The array of the key points of the second frame.
	
	:param patch_side:
		The patch side.
	
	:return:
		The arrays of the patches relative to the first and the second frame.
	"""
	first_patches = []
	second_patches = []
	
	first_rgbd = action.first.get_rgbd_image()
	first_color = np.asarray(first_rgbd.color)
	first_depth = np.asarray(first_rgbd.depth)
	second_rgbd = action.second.get_rgbd_image()
	second_color = np.asarray(second_rgbd.color)
	second_depth = np.asarray(second_rgbd.depth)
	
	for key_point in key_points_1:
		first_patches.append(__get_patch(key_point,
										 first_color,
										 first_depth,
										 patch_side))
	for key_point in key_points_2:
		second_patches.append(__get_patch(key_point,
										  second_color,
										  second_depth,
										  patch_side))
	
	first_patches = np.array(first_patches)
	second_patches = np.array(second_patches)
	return first_patches, second_patches

def __get_patch(patch_key_point, color_img, depth_img, patch_side):
	"""Extract the patch from the key point and the images.
	
	:param patch_key_point:
		The key point from which the patch must be extracted.
	
	:param color_img:
		The rgb image from which the patch must be extracted.
	
	:param depth_img:
		The depth image from which the patch must be extracted.
	
	:param patch_side:
		The side of the patch.
	
	:return:
		The extracted patch.
	"""
	w = color_img.shape[1]
	h = color_img.shape[0]
	patch = np.zeros((4, 1 + 2 * patch_side, 1 + 2 * patch_side))
	xc = patch_key_point[0]
	yc = patch_key_point[1]
	
	# Iterate taking care of border cases
	for x_off in range(2 * patch_side + 1):
		for y_off in range(2 * patch_side + 1):
			xo = int(max(0, min(xc - patch_side,
								w - 1 - 2 * patch_side)) + x_off)
			yo = int(max(0, min(yc - patch_side,
								h - 1 - 2 * patch_side)) + y_off)
			
			patch[0, y_off, x_off] = color_img[yo, xo, 0]
			patch[1, y_off, x_off] = color_img[yo, xo, 1]
			patch[2, y_off, x_off] = color_img[yo, xo, 2]
			patch[3, y_off, x_off] = depth_img[yo, xo]
	
	return patch

def get_latent_vectors(network_path: str,
					   first_patches: np.ndarray,
					   second_patches: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""From the patches, it extracts the latent vectors.
	
	:param network_path:
		The path where the trained network is stored.
	
	:param first_patches:
		The first patches.
	
	:param second_patches:
		The second patches.
	 
	:return:
		The latent vectors relative to the first and second patches.
	"""
	model: RGBD_TripletNetwork = torch.load(network_path)
	model.eval()
	
	patches_input = Variable(Tensor(first_patches).float())
	first_latent_vectors = model(patches_input).detach().numpy()
	
	patches_input = Variable(Tensor(second_patches).float())
	second_latent_vectors = model(patches_input).detach().numpy()
	
	return first_latent_vectors, second_latent_vectors

def get_semantic_scores(first_latent_vectors: np.ndarray,
						second_latent_vectors: np.ndarray) -> list:
	"""Extract the semantic score from the latent vectors.
	
	:param first_latent_vectors:
		The latent vectors relative to the first image.
	
	:param second_latent_vectors:
		The latent vectors relative to the second image.
	
	:return:
		The semantic scores of each match.
	"""
	# Even though a different structure is used, element at index i of action.links
	# is the same match represented at index i of matches_semantic_scores
	semantic_scores = []
	for idx in range(len(first_latent_vectors)):
		match_score = LatentMatchScore(
			first_latent_vectors,
			second_latent_vectors,
			idx,
			idx
		)
		match_score.compute_score(method="euclidean")
		semantic_scores.append(match_score.score)
	return semantic_scores

def compute_probabilities(semantic_scores: list) -> np.ndarray:
	"""Compute the probability of a given match.
	
	:param semantic_scores:
		The semantic scores of the matches.
	
	:return:
		The probabilities of the matches.
	"""
	semantic_scores = np.array(semantic_scores)
	semantic_scores -= semantic_scores.mean()
	semantic_scores /= semantic_scores.std()
	semantic_scores = torch.from_numpy(semantic_scores).float()
	semantic_probs = torch.nn.Softmax(dim=0)(semantic_scores)
	ones_probs = torch.ones(semantic_probs.size(dim=0))
	semantic_probs = torch.sub(ones_probs, semantic_probs)
	semantic_probs = torch.nn.Softmax(dim=0)(semantic_probs)
	probs = semantic_probs.tolist()
	probs = np.asarray(probs)
	probs[probs < np.percentile(probs, 60)] = 0
	probs /= np.sum(probs)
	return probs

def get_match_probabilities(action: Action,
							network_path: str,
							patch_side: int) -> np.ndarray:
	"""From a merged action, it retrieves the probabilities of the matches.
	
	:param action:
		The action from which features must be extracted.
		
	:param network_path:
		The path where the trained network is stored.
	
	:param patch_side:
		The side of the patch.
		
	:return:
		The probabilities of the matches.
	"""
	features_1, features_2 = get_features_from_merged(action)
	first_key_points, second_key_points = get_key_points_from_features(features_1, features_2)
	first_patches, second_patches = get_semantic_patches(action, first_key_points, second_key_points, patch_side)
	first_latent_vectors, second_latent_vectors = get_latent_vectors(network_path, first_patches, second_patches)
	semantic_scores = get_semantic_scores(first_latent_vectors, second_latent_vectors)
	probs = compute_probabilities(semantic_scores)
	return probs
