import json
import linecache
import math
import os
from os.path import exists
from typing import Tuple, Union

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from camera.Action import Action
from camera.Frame import Frame
from pipeline_utils import get_match_probabilities, get_features_from_merged, \
	get_key_points_from_features, get_semantic_patches, get_latent_vectors, \
	get_semantic_scores, compute_probabilities
from tools.Merger import Merger
from tools.SemanticSampling import SemanticSampling
from utils.utils import get_rgb_triplet_dataset_path, \
	get_pose_triplet_dataset_path, get_depth_triplet_dataset_path


class MeanAverageAccuracy(object):
	"""It is a class computing mean average accuracy of a roto-translation computation model.
	"""
	
	def __init__(self, threshold: float,
				 frames_distance: int,
				 last_image: int,
				 merge_features: int,
				 detect_method: str,
				 match_method: str,
				 network_path: str,
				 ransac_iterations: int,
				 patch_side: int,
				 dataset: str = "washington",
				 method: str = "rgbd"):
		"""Initializes the MeanAverageAccuracy object.
		
		:param threshold:
			The maximum angle of the rotation between estimated rotation and the
			ground truth rotation to consider the computation correct. It is
			measured in degrees.
			
		:param frames_distance:
			The distance between two consecutive frames.
			
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
			
		:param dataset:
			It can be either "washington" or "notre_dame".
			
		:param method:
			States if the network works on rgb or rbgd images. It can be rgb or
			rgbd.
		"""
		super().__init__()
		
		self.threshold = threshold
		self.frames_distance = frames_distance
		self.last_image = last_image
		self.merge_features = merge_features
		self.detect_method = detect_method
		self.match_method = match_method
		self.network_path = network_path
		self.ransac_iterations = ransac_iterations
		self.patch_side = patch_side
		self.dataset = dataset
		self.method = method
		
		self.dnn_fails = 0
		self.cv_fails = 0
		
	def compute_metric(self, verbose: bool = True) -> Tuple[float, float]:
		"""Computes the mean average accuracy of classical ransac and DNN ransac.
		
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
		for i in range(self.last_image - self.frames_distance):
			img1 = i
			img2 = i + self.frames_distance
			
			if verbose:
				print("Analysing images %s and %s" % (img1, img2))
			
			total += 1
			frame1, frame2 = self._get_frames(img1, img2)
			action = Action(frame1, frame2)
			dnn_f, dnn_mask, cv_f, cv_mask = self.__get_dnn_cv_matrices(action)
			
			# Compute ground truth roto-translation
			truth_R, truth_t = self._get_ground_truth(action)
			
			# Compute roto-translation of dnn
			action.set_fundamental_matrix(dnn_f, dnn_mask)
			if dnn_f is not None:
				action.compute_essential_matrix()
				action.roto_translation()
				dnn_R, dnn_t = action.R, action.t
				
				# Compute angle between estimations and ground truth
				dnn_unit_t = dnn_t / np.linalg.norm(dnn_t)
				truth_unit_t = truth_t / np.linalg.norm(truth_t)
				dnn_t_angle = np.arccos(np.dot(dnn_unit_t, truth_unit_t))
			
				# Compute angles of the rotation matrix
				dnn_rot_angles = Rotation.from_matrix(dnn_R).as_euler('xyz', degrees=True)
				truth_rot_angles = Rotation.from_matrix(truth_R).as_euler('xyz', degrees=True)
				dnn_R_angles = np.abs(np.array(truth_rot_angles) - np.array(dnn_rot_angles))
				
				# I check that every angle is less than the threshold. If so, the
				# transformation is considered correct
				if (dnn_t_angle < self.threshold and
						dnn_R_angles[0] < self.threshold and
						dnn_R_angles[1] < self.threshold and
						dnn_R_angles[2] < self.threshold):
					dnn_correct += 1
			else:
				print("WARNING: DNN Fundamental matrix not found")
				self.dnn_fails += 1
			
			# Compute roto-translation of classical computer vision
			action.set_fundamental_matrix(cv_f, cv_mask)
			if cv_f is not None:
				action.compute_essential_matrix()
				action.roto_translation()
				cv_R, cv_t = action.R, action.t
				
				# Compute angle between estimations and ground truth
				cv_unit_t = cv_t / np.linalg.norm(cv_t)
				truth_unit_t = truth_t / np.linalg.norm(truth_t)
				cv_t_angle = np.arccos(np.dot(cv_unit_t, truth_unit_t))
			
				# Compute angles of the rotation matrix
				cv_rot_angles = Rotation.from_matrix(cv_R).as_euler('xyz', degrees=True)
				truth_rot_angles = Rotation.from_matrix(truth_R).as_euler('xyz', degrees=True)
				cv_R_angles = np.abs(np.array(truth_rot_angles) - np.array(cv_rot_angles))
				
				# I check that every angle is less than the threshold. If so, the
				# transformation is considered correct
				if (cv_t_angle < self.threshold and
					  cv_R_angles[0] < self.threshold and
					  cv_R_angles[1] < self.threshold and
					  cv_R_angles[2] < self.threshold):
					cv_correct += 1
			else:
				print("WARNING: Classical Fundamental matrix not found")
				self.cv_fails += 1
		
		return dnn_correct / total, cv_correct / total
	
	def _get_frames(self, img1: int,
					img2: int) -> Tuple[Frame, Frame]:
		"""Gets the two frame from the dataset.
		
		:param img1:
			The index of the first image to consider.
		
		:param img2:
			The index of the second image to consider.
		
		:return:
			The frames corresponding to the first image (first) and to the
			second image (second).
		"""
		if self.dataset == "washington":
			frame1 = Frame(get_rgb_triplet_dataset_path("../Dataset", "Testing", 2, img1),
						   get_depth_triplet_dataset_path("../Dataset", "Testing", 2, img1),
						   get_pose_triplet_dataset_path("../Dataset", "Testing", 2),
						   img1)
			frame2 = Frame(get_rgb_triplet_dataset_path("../Dataset", "Testing", 2, img2),
						   get_depth_triplet_dataset_path("../Dataset", "Testing", 2,img2),
						   get_pose_triplet_dataset_path("../Dataset", "Testing", 2),
						   img2)
		else:
			img1_name = linecache.getline("../Dataset/NotreDame/list.txt", img1 + 1)
			img2_name = linecache.getline("../Dataset/NotreDame/list.txt", img2 + 1)
			
			frame1 = Frame("../Dataset/NotreDame/" + img1_name[:-1],
						   "not_present_in_notre_dame",
						   "not_present_in_notre_dame",
						   img1)
			frame2 = Frame("../Dataset/NotreDame/" + img2_name[:-1],
						   "not_present_in_notre_dame",
						   "not_present_in_notre_dame",
						   img2)
		
		return frame1, frame2
	
	def _get_ground_truth(self, action: Action):
		"""Gets the ground truth rotation and translation.
		
		:param action:
			The action representing the couple of images.
			
		:return:
			A tuple composed of rotation (first) and translation (second)
		"""
		if self.dataset == "washington":
			t_from_2_to_0, t_from_1_to_0 = action.roto_translation_pose()
			roto_translation = np.linalg.inv(t_from_1_to_0) @ t_from_2_to_0
			truth_R = roto_translation[0:3, 0:3]
			truth_t = roto_translation[0:3, 3]
		else:
			img1 = action.first.index
			img2 = action.second.index
			
			img1_f_k1_k2 = linecache.getline("../Dataset/NotreDame/notredame.out", img1 + 3)
			img1_row1_R = linecache.getline("../Dataset/NotreDame/notredame.out", img1 + 1 + 3)
			img1_row2_R = linecache.getline("../Dataset/NotreDame/notredame.out", img1 + 2 + 3)
			img1_row3_R = linecache.getline("../Dataset/NotreDame/notredame.out", img1 + 3 + 3)
			img1_translation = linecache.getline("../Dataset/NotreDame/notredame.out", img1 + 4 + 3)
			
			img2_f_k1_k2 = linecache.getline("../Dataset/NotreDame/notredame.out", img2 + 3)
			img2_row1_R = linecache.getline("../Dataset/NotreDame/notredame.out", img2 + 1 + 3)
			img2_row2_R = linecache.getline("../Dataset/NotreDame/notredame.out", img2 + 2 + 3)
			img2_row3_R = linecache.getline("../Dataset/NotreDame/notredame.out", img2 + 3 + 3)
			img2_translation = linecache.getline("../Dataset/NotreDame/notredame.out", img2 + 4 + 3)
			
			img1_f_k1_k2 = img1_f_k1_k2.split(" ")
			img1_f_k1_k2 = np.array(img1_f_k1_k2, dtype=float)
			f, k1, k2 = img1_f_k1_k2[0], img1_f_k1_k2[1], img1_f_k1_k2[2]
			action.first.fx, action.first.fy = f, f
			action.first.Cx = action.first.get_width() / 2
			action.first.Cy = action.first.get_height() / 2
			
			img2_f_k1_k2 = img2_f_k1_k2.split(" ")
			img2_f_k1_k2 = np.array(img2_f_k1_k2, dtype=float)
			f, k1, k2 = img2_f_k1_k2[0], img2_f_k1_k2[1], img2_f_k1_k2[2]
			action.second.fx, action.second.fy = f, f
			action.second.Cx = action.second.get_width() / 2
			action.second.Cy = action.second.get_height() / 2
			
			img1_row1_R = img1_row1_R.split(" ")
			img1_row1_R = np.array(img1_row1_R, dtype=float)
			img1_row2_R = img1_row2_R.split(" ")
			img1_row2_R = np.array(img1_row2_R, dtype=float)
			img1_row3_R = img1_row3_R.split(" ")
			img1_row3_R = np.array(img1_row3_R, dtype=float)
			img1_translation = img1_translation.split(" ")
			img1_translation = np.array(img1_translation, dtype=float)
			
			img2_row1_R = img2_row1_R.split(" ")
			img2_row1_R = np.array(img2_row1_R, dtype=float)
			img2_row2_R = img2_row2_R.split(" ")
			img2_row2_R = np.array(img2_row2_R, dtype=float)
			img2_row3_R = img2_row3_R.split(" ")
			img2_row3_R = np.array(img2_row3_R, dtype=float)
			img2_translation = img2_translation.split(" ")
			img2_translation = np.array(img2_translation, dtype=float)
			
			# here images are from 0 to i
			img1_R = np.array([[img1_row1_R[0], img1_row1_R[1], img1_row1_R[2]],
							   [img1_row2_R[0], img1_row2_R[1], img1_row2_R[2]],
							   [img1_row3_R[0], img1_row3_R[1], img1_row3_R[2]]], dtype=float)
			img2_R = np.array([[img2_row1_R[0], img2_row1_R[1], img2_row1_R[2]],
							   [img2_row2_R[0], img2_row2_R[1], img2_row2_R[2]],
							   [img2_row3_R[0], img2_row3_R[1], img2_row3_R[2]]], dtype=float)
			
			truth_R = img1_R @ np.linalg.inv(img2_R)
			truth_t = img1_translation - img2_translation
		
		return truth_R, truth_t
	
	def save_to_file(self, file_path: str,
					 dnn_mAA: float,
					 cv_mAA: float) -> None:
		"""Save to file the accuracy and the parameters.
		
		:param file_path:
			Where to save the file.
			
		:param dnn_mAA:
			Mean average accuracy of DNN.
		 
		:param cv_mAA:
			Mean average accuracy of classical CV.
		
		:return:
			None
		"""
		accuracy_setup = {"threshold": self.threshold,
						  "frames_distance": self.frames_distance,
						  "last_image": self.last_image,
						  "merge_features": self.merge_features,
						  "detect_method": self.detect_method,
						  "match_method": self.match_method,
						  "network_path": self.network_path,
						  "ransac_iterations": self.ransac_iterations,
						  "patch_side": self.patch_side,
						  "dnn_maa": dnn_mAA,
						  "cv_maa": cv_mAA,
						  "dnn_fails": self.dnn_fails,
						  "cv_fails": self.cv_fails}
		json_string = json.JSONEncoder().encode(accuracy_setup)
		
		with open(file_path, mode="w") as file_:
			json.dump(json_string, file_)
			
	@staticmethod
	def load_accuracy_computation(file_path: str) -> Union[dict, None]:
		"""Loads a previously stored mean average accuracy computation.
		
		:param file_path:
			Where to save the file.
			
		:return:
			The accuracy setup and the computed accuracy.
		"""
		if exists(file_path):
			with open(file_path) as file_:
				json_string = json.load(file_)
			
			return json.JSONDecoder().decode(json_string)
		else:
			return None
		
	def print_accuracy_computation(self, file_path: str) -> None:
		"""Loads the accuracy computation and prints it.
		
		:param file_path:
			Where to save the file.
			
		:return:
			None.
		"""
		accuracy_setup = self.load_accuracy_computation(file_path)
		
		if accuracy_setup is None:
			print("There is no search at the specified path")
		else:
			for key, item in accuracy_setup.items():
				print("%s = %s" % (key, item))
	
	def __get_dnn_cv_matrices(self, action: Action) -> Tuple:
		"""Get fundamental matrix of DNN and classical computer vision RANSAC.

		:param action:
			The action over which the fundamental matrix must be computed.

		:return:
			A tuple returning fundamental matrices and the masks in order: dnn
			fundamental matrix, dnn mask, cv fundamental matrix cv mask.
		"""
		merger = Merger(
			num_features=self.merge_features,
			detector_method=self.detect_method,
			matcher_method=self.match_method
		)
		merge_image = merger.merge_action(action, return_draw=False)
		
		features_1, features_2 = get_features_from_merged(action)
		first_key_points, second_key_points = get_key_points_from_features(features_1, features_2)
		first_patches, second_patches = get_semantic_patches(action, first_key_points, second_key_points, self.patch_side, self.method)
		
		if first_patches.shape[0] != 0:
			first_latent_vectors, second_latent_vectors = get_latent_vectors(self.network_path, first_patches, second_patches)
			semantic_scores = get_semantic_scores(first_latent_vectors, second_latent_vectors)
			probs = compute_probabilities(semantic_scores)
			
			# Eliminate NaN value and re-normalize the values if necessary
			for i in range(probs.shape[0]):
				if probs[i] == np.nan or math.isnan(probs[i]):
					probs[i] = 0.0
			if np.sum(probs) == 0:
				probs[0] = 1.0
			
			semantic_ransac = SemanticSampling()
			dnn_best_f, dnn_best_mask = semantic_ransac.ransac_fundamental_matrix(
				action,
				iterations=self.ransac_iterations,
				semantic=probs
			)
			cv_best_f, cv_best_mask = semantic_ransac.ransac_fundamental_matrix(
				action,
				iterations=self.ransac_iterations
			)
		else:
			dnn_best_f, dnn_best_mask, cv_best_f, cv_best_mask = None, None, None, None
			
		return dnn_best_f, dnn_best_mask, cv_best_f, cv_best_mask

