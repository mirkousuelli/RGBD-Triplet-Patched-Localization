import json
from os.path import exists
from typing import Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation

from camera.Action import Action
from camera.Frame import Frame
from pipeline_utils import get_match_probabilities
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
				 patch_side: int):
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
			
			frame1 = Frame(
				get_rgb_triplet_dataset_path("../Dataset", "Testing", 2, img1),
				get_depth_triplet_dataset_path("../Dataset", "Testing", 2, img1),
				get_pose_triplet_dataset_path("../Dataset", "Testing", 2),
				img1
			)
			frame2 = Frame(
				get_rgb_triplet_dataset_path("../Dataset", "Testing", 2, img2),
				get_depth_triplet_dataset_path("../Dataset", "Testing", 2, img2),
				get_pose_triplet_dataset_path("../Dataset", "Testing", 2),
				img2
			)
			action = Action(frame1, frame2)
			dnn_f, dnn_mask, cv_f, cv_mask = self.__get_dnn_cv_matrices(action)
			
			# Compute roto-translation of dnn
			action.set_fundamental_matrix(dnn_f, dnn_mask)
			action.compute_essential_matrix()
			action.roto_translation()
			dnn_R, dnn_t = action.R, action.t
			
			# Compute roto-translation of classical computer vision
			action.set_fundamental_matrix(cv_f, cv_mask)
			action.compute_essential_matrix()
			action.roto_translation()
			cv_R, cv_t = action.R, action.t
			
			# Compute ground truth roto-translation
			t_from_2_to_0, t_from_1_to_0 = action.roto_translation_pose()
			roto_translation = t_from_2_to_0 @ np.linalg.inv(t_from_1_to_0)
			truth_R = roto_translation[0:3, 0:3]
			truth_t = roto_translation[0:3, 3]
			
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
			if (dnn_t_angle < self.threshold and
					dnn_R_angles[0] < self.threshold and
					dnn_R_angles[1] < self.threshold and
					dnn_R_angles[2] < self.threshold):
				dnn_correct += 1
			elif (cv_t_angle < self.threshold and
				  cv_R_angles[0] < self.threshold and
				  cv_R_angles[1] < self.threshold and
				  cv_R_angles[2] < self.threshold):
				cv_correct += 1
		
		return dnn_correct / total, cv_correct / total
	
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
						  "cv_maa": cv_mAA}
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
		merge_image = merger.merge_action(action)
		probs = get_match_probabilities(action, self.network_path, self.patch_side)
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
		return dnn_best_f, dnn_best_mask, cv_best_f, cv_best_mask

