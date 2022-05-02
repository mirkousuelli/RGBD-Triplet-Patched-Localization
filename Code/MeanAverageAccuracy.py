from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from camera.Action import Action
from camera.Frame import Frame
from pipeline_utils import get_match_probabilities
from tools.Merger import Merger
from tools.SemanticSampling import SemanticSampling
from utils.utils import get_rgb_triplet_dataset_path, \
	get_pose_triplet_dataset_path, get_depth_triplet_dataset_path


def meanAverageAccuracy(threshold: float,
						last_image: int,
						merge_features: int,
						detect_method: str,
						match_method: str,
						network_path: str,
						ransac_iterations: int,
						patch_side: int,
						verbose: bool = True) -> Tuple[float, float]:
	"""Computes the mean average accuracy of classical ransac and DNN ransac.
	
	:param threshold:
		The maximum angle of the rotation between estimated rotation and the
		ground truth rotation to consider the computation correct. It is
		measured in degrees.
		
	:param last_image:
		The index of the last image to consider in test.
		
	:param merge_features:
		The number of iterations while merging the action.
	
	:param detect_method:
		The method to be used for detection.
	
	:param match_method:
		The method to be used for matching.
	
	:param network_path:
		The path of the trained network.
	
	:param ransac_iterations:
		The iterations of ransac.
	
	:param patch_side:
		The dimension of patch's side.
		
	:param verbose:
		If true, detailed printing will be shown.
		
	:return:
		The mean average accuracy of the DNN and classical computer vision
		ransac.
	"""
	dnn_correct = 0
	cv_correct = 0
	total = 0
	
	# I iterate over all the images of the test set to compute the
	# rototranslation of the dnn ransac and classical ransac.
	for i in range(last_image):
		if verbose:
			print("Analysing images 0 and %s" % (i+1))
		
		frame1 = Frame(
			get_rgb_triplet_dataset_path("../Dataset", "Testing", 2, 0),
			get_depth_triplet_dataset_path("../Dataset", "Testing", 2, 0),
			get_pose_triplet_dataset_path("../Dataset", "Testing", 2),
			0
		)
		frame2 = Frame(
			get_rgb_triplet_dataset_path("../Dataset", "Testing", 2, i+1),
			get_depth_triplet_dataset_path("../Dataset", "Testing", 2, i+1),
			get_pose_triplet_dataset_path("../Dataset", "Testing", 2),
			i+1
		)
		action = Action(frame1, frame2)
		dnn_f, dnn_mask, cv_f, cv_mask = __get_dnn_cv_matrices(action,
															   merge_features,
															   detect_method,
															   match_method,
															   network_path,
															   ransac_iterations,
															   patch_side)
		
		# Compute rototranslation of dnn
		action.set_fundamental_matrix(dnn_f, dnn_mask)
		action.compute_essential_matrix()
		action.roto_translation()
		dnn_R, dnn_t = action.R, action.t
		
		# Compute rototranslation of classical computer vision
		action.set_fundamental_matrix(cv_f, cv_mask)
		action.compute_essential_matrix()
		action.roto_translation()
		cv_R, cv_t = action.R, action.t
		
		# Compute ground truth rototranslation
		t_from_2_to_0, t_from_1_to_0 = action.roto_translation_pose()
		truth_R = t_from_2_to_0[0:3, 0:3]
		truth_t = t_from_2_to_0[0:3, 3]
		
		# Compute angle between estimations and ground truth
		dnn_unit_t = dnn_t / np.linalg.norm(dnn_t)
		cv_unit_t = cv_t / np.linalg.norm(cv_t)
		truth_unit_t = truth_t / np.linalg.norm(truth_t)
		dnn_t_angle = np.arccos(np.dot(dnn_unit_t, truth_unit_t))
		cv_t_angle = np.arccos(np.dot(cv_unit_t, truth_unit_t))
		
		# Compute angles of the rotation matrix
		dnn_rot_angles = Rotation.from_matrix(dnn_R).as_euler('xyz', degrees=True)
		cv_rot_angles = Rotation.from_matrix(cv_R).as_euler('xyz', degrees=True)
		truth_rot_angles = Rotation.from_matrix(truth_R).as_euler('xyz', degrees=True)
		dnn_R_angles = np.abs(np.array(truth_rot_angles) - np.array(dnn_rot_angles))
		cv_R_angles = np.abs(np.array(truth_rot_angles) - np.array(cv_rot_angles))
		
		# I check that every angle is less than the threshold. If so, the
		# transformation is considered correct
		total += 1
		if (dnn_t_angle < threshold and
			dnn_R_angles[0] < threshold and
			dnn_R_angles[1] < threshold and
			dnn_R_angles[2] < threshold):
			dnn_correct += 1
		elif (cv_t_angle < threshold and
			  cv_R_angles[0] < threshold and
			  cv_R_angles[1] < threshold and
			  cv_R_angles[2] < threshold):
			cv_correct += 1
			
	return dnn_correct / total, cv_correct / total

def __get_dnn_cv_matrices(action: Action,
						  merge_features: int,
						  detect_method: str,
						  match_method: str,
						  network_path: str,
						  ransac_iterations: int,
						  patch_side: int) -> Tuple:
	"""Get fundamental matrix of DNN and classical computer vision RANSAC.
	
	:param action:
		The action over which the fundamental matrix must be computed.
	
	:param merge_features:
		The number of iterations while merging the action.
	
	:param detect_method:
		The method to be used for detection.
	
	:param match_method:
		The method to be used for matching.
	
	:param network_path:
		The path of the trained network.
	
	:param ransac_iterations:
		The iterations of ransac.
	
	:param patch_side:
		The dimension of patch's side.
	
	:return:
		A tuple returning fundamental matrices and the masks in order: dnn
		fundamental matrix, dnn mask, cv fundamental matrix cv mask.
	"""
	merger = Merger(
		num_features=merge_features,
		detector_method=detect_method,
		matcher_method=match_method
	)
	merge_image = merger.merge_action(action)
	probs = get_match_probabilities(action, network_path, patch_side)
	semantic_ransac = SemanticSampling()
	dnn_best_f, dnn_best_mask = semantic_ransac.ransac_fundamental_matrix(
		action,
		iterations=ransac_iterations,
		semantic=probs
	)
	cv_best_f, cv_best_mask = semantic_ransac.ransac_fundamental_matrix(
		action,
		iterations=ransac_iterations
	)
	return dnn_best_f, dnn_best_mask, cv_best_f, cv_best_mask
