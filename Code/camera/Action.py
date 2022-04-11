# Python imports

# External imports

# In-project imports
from typing import Tuple, Union

import cv2
import numpy as np
from open3d.cpu.pybind.geometry import KDTreeSearchParamHybrid
from open3d.cpu.pybind.pipelines.registration import registration_icp, \
	TransformationEstimationPointToPoint, evaluate_registration, \
	ICPConvergenceCriteria, TransformationEstimationPointToPlane

from camera.Frame import Frame
from ProjectObject import ProjectObject


# TODO: wow, implement me please
from tools.Merger import Merger
from utils.transformation_utils import get_4x4_transform_from_quaternion, \
	get_4x4_transform_from_translation


class Action(ProjectObject):
	ERROR_KEY = ProjectObject.ERROR_KEY + ["action"]
	# RANSAC hyper-parameters
	RANSAC_THRESHOLD_PIXEL = 0.1
	RANSAC_PROB = 0.999
	RANSAC_ITER = 10000
	
	def __init__(self,
	             first: Frame,
	             second: Frame):
		super().__init__()

		# Frames
		self.first = first
		self.second = second

		# Matches
		self.matches = None
		self.links = None
		self.links_inliers = []

		# Fundamental Matrix
		self.f_matrix = None

		# Inliers mask
		self.f_mask = None

		# Essential Matrix
		self.e_matrix = None

		# Roto-Translation
		self.R = None
		self.t = None

	def normalize_essential_matrix(self):
		"""Normalizes the essential matrix"""
		# pre-conditions
		assert self.e_matrix is not None
		
		self.e_matrix = self.e_matrix / self.e_matrix[2, 2]

	def compute_fundamental_matrix(
		self,
		inplace=True
	):
		"""
		Compute the Fundamental matrix between the two frames inside the action.

		:param inplace:
            If the operation must happen inplace
        :type inplace: bool

        :returns:
            Fundamental matrix and inliers mask
		"""
		# pre-conditions
		assert len(self.first.points) == len(self.second.points), \
			"Frames features array have different size!"
		assert len(self.first.points) > 0, "Features array is empty!"

		# Fundamental Matrix + RANSAC
		F, mask = cv2.findFundamentalMat(
			self.first.points,
			self.second.points,
			cv2.FM_RANSAC,
			ransacReprojThreshold=self.RANSAC_THRESHOLD_PIXEL,
			confidence=self.RANSAC_PROB,
			maxIters=self.RANSAC_ITER
		)

		if F is None or F.shape == (1, 1):
			# no fundamental matrix found
			raise Exception('No fundamental matrix found')
		elif F.shape[0] > 3:
			# more than one matrix found, just pick the first
			F = F[0:3, 0:3]

		# returning
		if inplace:
			self.f_matrix = np.mat(F)
			self.f_mask = mask
		else:
			return np.mat(F), mask

	def compute_inliers(self):
		"""
		This static method computes the inlier through the previously computed
		mask through RANSAC and afterwards selects all the matches between
		inliers.
		"""
		# pre-conditions
		assert len(self.first.points) == len(self.second.points), \
			"Frames features array have different size!"
		assert len(self.first.points) > 0, "Features array is empty!"
		assert self.f_mask is not None, "You must compute the Fundamental " \
		                                  "Matrix before!"

		# We select only inlier points
		self.first.inliers = self.first.points[self.f_mask.ravel() == 1]
		self.second.inliers = self.second.points[self.f_mask.ravel() == 1]

		# Looking for links through the previous selected inliers
		for i in range(len(self.links)):
			if self.f_mask[i].ravel() == 1:
				self.links_inliers.append(self.links[i])

	def compute_essential_matrix(
		self,
		inplace=True
	):
		"""
		Computes the essential matrix from the Action fundamental matrix and the
		frames' calibration matrices.

		:param action:
			The action containing the fundamental matrix.
		:type action: Action

		:param inplace:
			If the operation must happen inplace.
		:type inplace: bool

		:return:
			Essential Matrix.
		"""
		# pre-conditions
		if self.f_matrix is None:
			self.compute_fundamental_matrix(inplace)

		# essential matrix computation
		if inplace:
			self.e_matrix = self.second.calibration_matrix().T \
			                  @ self.f_matrix @ self.first.calibration_matrix()
		else:
			return self.second.calibration_matrix().T @ self.f_matrix \
			       @ self.first.calibration_matrix()

	def compute_epipolar_lines(self):
		"""
		Compute the correspondent epipolar lines for both frames involved
		within the action
		"""
		# pre-conditions
		assert self.f_matrix is not None, "You must compute the " \
		                                    "Fundamental Matrix before!"

		# first frame
		self.first.epi_lines = cv2.computeCorrespondEpilines(
			self.second.inliers.reshape(-1, 1, 2), 2, self.f_matrix)
		self.first.epi_lines = self.first.epi_lines.reshape(-1, 3)

		# second frame
		self.second.epi_lines = cv2.computeCorrespondEpilines(
			self.first.inliers.reshape(-1, 1, 2), 1, self.f_matrix)
		self.second.epi_lines = self.second.epi_lines.reshape(-1, 3)

	def __draw_epipolar_lines(self,
							  frame_1: Frame,
	                          frame_2: Frame):
		"""
		Private static method used to print out the epipolar lines on the image.
		Note that the order of the two frames is important because the method
		computes just the epipolar lines drawing of the first image contained
		in the first frame.

		:param frame_1:
			First frame.
		:type frame_1: Frame

		:param frame_2:
			Second frame.
		:type frame_2: Frame

		:return:
			Image of the first frame with the epipolar lines printed out
		"""
		img = frame_1.get_cv2_images(ret="rgb")
		_, c = frame_1.get_size()

		# fixed the seed to match the same color in the other frame
		np.random.seed(42)

		# for each epipolar lines and inlier point relying on both frames
		for lines, pt1, pt2 in zip(frame_1.epi_lines,
		                           frame_1.inliers, frame_2.inliers):
			# choose random color
			color = np.random.randint(0, 255, 3).tolist()

			# select two point for the epipolar lines
			x0, y0 = map(int, [0, -lines[2] / lines[1]])
			x1, y1 = map(int, [c, -(lines[2] + lines[0] * c) / lines[1]])

			# print lines for the epipolar lines
			img = cv2.line(img, (x0, y0), (x1, y1), color, 1)

			# print circles fo the inliner key points
			img = cv2.circle(img, tuple(pt1), 5, color, -1)

		return img

	def show_epipolar_lines(self):
		"""
		Show the epipolar lines on both frames within the Action.

		:param action:
			Action consisting of two frames.
		:type action: Action

		:return:
			Final image of the two frames with the epipolar lines drawn
		"""
		# pre-conditions
		assert self.first.epi_lines is not None \
		       and self.second.epi_lines is not None, "You must compute the " \
		                                                "Epipolar Lines before!"

		# draw epipolar lines for each of the two frames in the action object
		img_1 = self.__draw_epipolar_lines(self.first, self.second)
		img_2 = self.__draw_epipolar_lines(self.second, self.first)

		# return the final image
		return np.concatenate((img_1, img_2), axis=1)

	def roto_translation(
		self,
		inplace=True,
		normalize_em=False
	) -> Union[None, Tuple[np.mat, np.ndarray]]:
		"""
		Compute the roto-translation components, i.e. the rotation matrix and
		the translation vector from the essential matrix contained in the object
		Action.

		:param action:
			Action containing the Essential Matrix of the two frames.
		:type action:

		:param inplace:
		:type inplace:
		
		:param normalize_em:
			States whether the essential matrix must be normalized or not to fit
			the roto-translation. It can be used only with inplace = True.
		:type normalize_em:

		:return:
			Rotation Matrix and Translation Vector w.r.t. the first
			reference frame
		"""
		# pre-conditions
		if self.e_matrix is None:
			if inplace:
				self.compute_essential_matrix(inplace)
			else:
				e_matrix = self.compute_essential_matrix(inplace)

		if normalize_em and inplace:
			self.normalize_essential_matrix()
		
		# SVD decomposition of the essential matrix
		w, u, vt = cv2.SVDecomp(self.e_matrix if inplace else e_matrix)

		# determinant adjustments
		if np.linalg.det(u) < 0:
			u *= -1.0
		if np.linalg.det(vt) < 0:
			vt *= -1.0

		W = np.mat([[0, -1, 0],
		            [1, 0, 0],
		            [0, 0, 1]], dtype=float)
		
		if normalize_em and inplace:
			self.compute_essential_matrix(inplace)

		# return the roto-translation components
		if inplace:
			self.R = np.mat(u) * W * np.mat(vt)
			self.t = u[:, 2]
		else:
			return np.mat(u) * W * np.mat(vt), u[:, 2]

	def roto_translation_with_icp(self, threshold,
								  verbose: bool = True,
								  estimation_method: str = "point-plane") -> np.ndarray:
		"""ICP used to register the images of the two point clouds.
		
		The method uses ICP on the two images composing the action to align the
		two point clouds.
		
		:param threshold:
			It is the maximum point-pair distance.
			
		:param verbose:
			States if detailed printing must be shown.
			
		:param estimation_method:
			The estimation method to be used in ICP to register the point clouds.
		
		:return:
			The matrix representing the transformation of the rototranslation.
		"""
		ACCEPTED_ESTIMATION_METHODS = ["point-plane", "point-point"]
		VOXEL_DOWN_SAMPLE = 0.01
		if estimation_method not in ACCEPTED_ESTIMATION_METHODS:
			raise ValueError("Estimation method must be one of the following "
							 "%s" % ACCEPTED_ESTIMATION_METHODS)
		
		# Get a rough transformation of the second frame into the first
		pose_2 = self.second.extract_pose()
		quat_2 = pose_2[0:4]
		pos_2 = pose_2[4:7]
		pose_1 = self.first.extract_pose()
		quat_1 = pose_1[0:4]
		pos_1 = pose_1[4:7]

		translation_1 = get_4x4_transform_from_translation(pos_1)
		rotation_1 = get_4x4_transform_from_quaternion(quat_1)
		translation_2 = get_4x4_transform_from_translation(pos_2)
		rotation_2 = get_4x4_transform_from_quaternion(quat_2)
		transformation_from_0_to_1 = np.matmul(translation_1, rotation_1)
		transformation_from_0_to_2 = np.matmul(translation_2, rotation_2)
		transformation_from_1_to_0 = np.linalg.inv(transformation_from_0_to_1)
		transformation_from_2_to_0 = np.linalg.inv(transformation_from_0_to_2)
		transformation_from_2_to_1 = np.matmul(transformation_from_0_to_1,
											   transformation_from_2_to_0)
		
		# Perform the registration using ICP
		first_cloud = self.first.get_point_cloud()
		second_cloud = self.second.get_point_cloud()
		
		if verbose:
			print("The first cloud has %s points and the second cloud has %s "
				  "points before downsampling" % (len(first_cloud.points),
												  len(second_cloud.points)))
		
		# Down sample point clouds to simplify the task of finding the transformation
		first_cloud = first_cloud.voxel_down_sample(VOXEL_DOWN_SAMPLE)
		second_cloud = second_cloud.voxel_down_sample(VOXEL_DOWN_SAMPLE)
		
		if verbose:
			print("The first cloud has %s points and the second cloud has %s "
				  "points after downsampling" % (len(first_cloud.points),
												 len(second_cloud.points)))
		
		if estimation_method == "point-point":
			estimation = TransformationEstimationPointToPoint()
		else:
			first_cloud.estimate_normals(KDTreeSearchParamHybrid(radius=0.1,
																 max_nn=30))
			second_cloud.estimate_normals(KDTreeSearchParamHybrid(radius=0.1,
																  max_nn=30))
			estimation = TransformationEstimationPointToPlane()
		icp_reg = registration_icp(second_cloud,
								   first_cloud,
								   threshold,
								   transformation_from_2_to_1,
								   estimation_method=estimation,
								   criteria=ICPConvergenceCriteria(max_iteration=2000))
		
		if verbose:
			evaluation = evaluate_registration(second_cloud,
											   first_cloud,
											   threshold,
											   icp_reg.transformation)
			print("The computed transformation is:\n %s" % icp_reg.transformation)
			print("The registration evaluation is %s" % evaluation)
		
		return icp_reg.transformation

	def from_rot_to_quat(
		self,
		normalize_em=True
	) -> np.ndarray:
		"""
		From Rotation Matrix to Quaternions.

		:param action:
			Action containing the two frames.
		:type action: Action
		
		:param normalize_em:
			States whether the essential matrix must be normalized or not to fit
			the roto-translation.
		:type normalize_em:

		:return:
			Quaternions
		"""
		# pre-conditions
		if self.R is None:
			self.roto_translation(normalize_em=normalize_em)

		# storing locally the rotation matrix for the sake of simplicity
		R = self.R

		# First row of the quaterion matrix
		q00 = R[0, 0] - R[1, 1] - R[2, 2]
		q01 = R[1, 0] + R[0, 1]
		q02 = R[2, 0] + R[0, 2]
		q03 = R[2, 1] - R[1, 2]

		# Second row of the quaterion matrix
		q10 = q01
		q11 = R[1, 1] - R[0, 0] - R[2, 2]
		q12 = R[2, 1] + R[1, 2]
		q13 = R[0, 2] - R[2, 0]

		# Third row of the quaterion matrix
		q20 = q02
		q21 = q12
		q22 = R[2, 2] - R[0, 0] - R[1, 1]
		q23 = R[1, 0] - R[0, 1]

		# Fourth row of the quaterion matrix
		q30 = q03
		q31 = q13
		q32 = q23
		q33 = R[0, 0] + R[1, 1] + R[2, 2]

		# Quaternion Matrix
		Q = np.mat([[q00, q01, q02, q03],
		            [q10, q11, q12, q13],
		            [q20, q21, q22, q23],
		            [q30, q31, q32, q33]], dtype=float) / 3

		# computing the eigenvectors of the symmetric matrix Q
		_, eig_vec = np.linalg.eigh(Q)

		# saving the last one eigenvector because since the matrix is symmetric
		# and the eigh() method returns the vectors ordered based on the+
		# eigenvalue (ascending order), we know that the last vector has eigen-
		# -value equals to 1 (the largest value) and hence the best approximation
		# for the choice of quaternions
		q = eig_vec[:, 3]

		# returning the quaternions in the correct order
		return np.array(np.concatenate((q[3].flatten().squeeze(),
		                                q[0].flatten().squeeze(),
		                                q[1].flatten().squeeze(),
		                                q[2].flatten().squeeze()))).squeeze()
