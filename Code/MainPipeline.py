import cv2
import numpy as np
import scipy.special
import torch
from torch import Tensor
from torch.autograd import Variable
from scipy import spatial

from camera.Action import Action
from camera.Frame import Frame
from neuralnetwork.RGBD_TripletNetwork import RGBD_TripletNetwork
from tools.Merger import Merger
from tools.SemanticSampling import SemanticSampling
from tools.Visualizer import Visualizer
from utils.utils import *

DETECTION_METHOD = "ORB"
first = 10
second = 50

# PIPELINE TO PERFORM SEMANTIC SAMPLING USING DEEP NEURAL NETWORKS
# PHASE 1: DETECTION
# Description: Using SIFT or ORB, features are detected using classical computer
# vision techniques.
print("# Phase 1: performing detection")
frame1 = Frame(
	get_rgb_triplet_dataset_path("../Dataset", "Testing", 2, first),
	get_depth_triplet_dataset_path("../Dataset", "Testing", 2, first),
	get_pose_triplet_dataset_path("../Dataset", "Testing", 2),
	first
)
frame2 = Frame(
	get_rgb_triplet_dataset_path("../Dataset", "Testing", 2, second),
	get_depth_triplet_dataset_path("../Dataset", "Testing", 2, second),
	get_pose_triplet_dataset_path("../Dataset", "Testing", 2),
	second
)
action = Action(frame1, frame2)
# Detection is embedded into the Merger class

# PHASE 2: MATCHING
# Description: Using an extractor, from the detected features we compose matches
# between points in images.
print("# Phase 2: performing matching")
merger = Merger(
	num_features=5000,
	detector_method=DETECTION_METHOD,
	matcher_method="FLANN"
)
merge_image = merger.merge_action(action)
cv2.imshow("Matches without DNN-RANSAC", merge_image)
matches = action.links
features_1 = [
	action.first.key_points[matches[x].queryIdx] for x in range(len(matches))
]
features_2 = [
	action.second.key_points[matches[x].trainIdx] for x in range(len(matches))
]

# I get all the key points relative to a match
first_key_points = np.array(
	[np.array([key.pt[0], key.pt[1], 1]) for key in features_1]
)
second_key_points = np.array(
	[np.array([key.pt[0], key.pt[1], 1]) for key in features_2]
)

# PHASE 3: PATCH EXTRACTION
# Description: Patches are extracted from the matches computed from features.
# I extract all the patches
print("# Phase 3: performing patch extraction")
first_patches = []
second_patches = []

first_rgbd = action.first.get_rgbd_image()
first_color = np.asarray(first_rgbd.color)
first_depth = np.asarray(first_rgbd.depth)
second_rgbd = action.second.get_rgbd_image()
second_color = np.asarray(second_rgbd.color)
second_depth = np.asarray(second_rgbd.depth)

for key_point in first_key_points:
	first_patches.append(
		SemanticSampling.get_patch(key_point, first_color, first_depth)
	)
for key_point in second_key_points:
	second_patches.append(
		SemanticSampling.get_patch(key_point, second_color, second_depth)
	)

first_patches = np.array(first_patches)
second_patches = np.array(second_patches)

# PHASE 4: LATENT VECTOR
# Description: We use the extracted patches to extract latent vectors using the
# DNN (Deep Neural Network).
print("# Phase 4: computing latent vectors")
file_path = "neuralnetwork/model/rgbd_triplet_patch_encoder_model_no_code.pt"
model: RGBD_TripletNetwork = torch.load(file_path)
model.eval()

patches_input = Variable(Tensor(first_patches).float())
first_latent_vectors = model(patches_input).detach().numpy()

patches_input = Variable(Tensor(second_patches).float())
second_latent_vectors = model(patches_input).detach().numpy()

# PHASE 5: SEMANTIC SCORE
# Description: A semantic score is computed between each latent vector to
# resemble a similarity function.
print("# Phase 5: computing semantic score of matches")


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
			raise ValueError("The method must be one of %s" % self.SCORING_METHODS)
		
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
	match_score.compute_score(method="cosine")
	semantic_scores.append(match_score.score)

# PHASE 6: SOFTMAX
# Description: Softmax is applied over all semantic scores to be able to extract
# a probability distribution over which weighted RANSAC will be executed.
print("# Phase 6: computing probabilities of being chosen")
semantic_scores = np.array(semantic_scores)
semantic_scores = np.ones(len(semantic_scores)) - \
                  (semantic_scores - min(semantic_scores)) / \
                  (max(semantic_scores) - min(semantic_scores))
semantic_scores = torch.from_numpy(semantic_scores).float()
semantic_probs = torch.nn.Softmax(dim=0)(semantic_scores)
probs = semantic_probs.tolist()
probs = np.asarray(probs)
probs /= np.sum(probs)
# PHASE 7: WEIGHTED RANSAC
# Description: RANSAC over the patches with weights to represent the probability
# of being chosen.
print("# Phase 7: executing weighted ransac")
semantic_ransac = SemanticSampling()
dnn_best_f, dnn_best_mask = semantic_ransac.ransac_fundamental_matrix(
	action,
	iterations=10,
	semantic=probs
)
cv_best_f, cv_best_mask = semantic_ransac.ransac_fundamental_matrix(
	action,
	iterations=10
)

# PHASE 8: LOCALIZATION
# Description: Given the results of RANSAC, we perform localization.
print("# Phase 8: doing localization")
action.set_fundamental_matrix(dnn_best_f, dnn_best_mask)
action.compute_essential_matrix()
action.roto_translation()
print("F = %s" % dnn_best_f)
print("E = %s" % action.e_matrix)
print("R, t = %s, %s" % (action.R, action.t))
print("Q = %s" % action.from_rot_to_quat(normalize_em=False))

# PHASE 9: VISUALIZATION
# Description: Given the results of the pipeline, point clouds are visualized.
print("# Phase 9: printing visualization")
action.set_inliers(dnn_best_mask)
dnn_inliers_image = merger.merge_inliers(action)

#visualizer = Visualizer(action=action)
#visualizer.plot_action_point_cloud(registration_method="standard")

cv2.imshow("DNN-RANSAC Inliers", dnn_inliers_image)

action.set_fundamental_matrix(cv_best_f, cv_best_mask)
action.compute_essential_matrix()
action.roto_translation()
print("F = %s" % cv_best_f)
print("E = %s" % action.e_matrix)
print("R, t = %s, %s" % (action.R, action.t))
print("Q = %s" % action.from_rot_to_quat(normalize_em=False))

action.set_inliers(cv_best_mask)
cv_inliers_image = merger.merge_inliers(action)

#visualizer = Visualizer(action=action)
#visualizer.plot_action_point_cloud(registration_method="standard")

cv2.imshow("RANSAC Inliers", cv_inliers_image)
cv2.waitKey(0)
