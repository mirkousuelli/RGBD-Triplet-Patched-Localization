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
from pipeline_utils import get_features_from_merged, \
	get_key_points_from_features, get_semantic_patches, get_latent_vectors, \
	get_semantic_scores, compute_probabilities
from tools.Merger import Merger
from tools.SemanticSampling import SemanticSampling
from tools.Visualizer import Visualizer
from utils.utils import *

ITERATIONS = 5000
DETECTION_METHOD = "ORB"
first = 300
second = 380
PATCH_SIDE = 8

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

features_1, features_2 = get_features_from_merged(action)
first_key_points, second_key_points = get_key_points_from_features(features_1, features_2)

# PHASE 3: PATCH EXTRACTION
# Description: Patches are extracted from the matches computed from features.
# I extract all the patches
print("# Phase 3: performing patch extraction")
first_patches, second_patches = get_semantic_patches(action,
													 first_key_points,
													 second_key_points,
													 PATCH_SIDE)

# PHASE 4: LATENT VECTOR
# Description: We use the extracted patches to extract latent vectors using the
# DNN (Deep Neural Network).
print("# Phase 4: computing latent vectors")
file_path = "neuralnetwork/model/rgbd_triplet_patch_encoder_model_euclidean.pt"
first_latent_vectors, second_latent_vectors = get_latent_vectors(file_path, first_patches, second_patches)

# PHASE 5: SEMANTIC SCORE
# Description: A semantic score is computed between each latent vector to
# resemble a similarity function.
print("# Phase 5: computing semantic score of matches")
semantic_scores = get_semantic_scores(first_latent_vectors, second_latent_vectors)

# PHASE 6: SOFTMAX
# Description: Softmax is applied over all semantic scores to be able to extract
# a probability distribution over which weighted RANSAC will be executed.
print("# Phase 6: computing probabilities of being chosen")
probs = compute_probabilities(semantic_scores)

# PHASE 7: WEIGHTED RANSAC
# Description: RANSAC over the patches with weights to represent the probability
# of being chosen.
print("# Phase 7: executing weighted ransac")
semantic_ransac = SemanticSampling()
dnn_best_f, dnn_best_mask = semantic_ransac.ransac_fundamental_matrix(
	action,
	iterations=ITERATIONS,
	semantic=probs
)
cv_best_f, cv_best_mask = semantic_ransac.ransac_fundamental_matrix(
	action,
	iterations=ITERATIONS
)

# PHASE 8: LOCALIZATION
# Description: Given the results of RANSAC, we perform localization.
print("# Phase 8: doing localization")
action.set_fundamental_matrix(dnn_best_f, dnn_best_mask)
action.compute_essential_matrix()
action.roto_translation()
print("DNN RANSAC results")
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
print("Classical Computer Vision RANSAC results")
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
