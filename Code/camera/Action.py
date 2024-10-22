from typing import Tuple, Union

import cv2
import numpy as np
from cv2 import DMatch
from open3d.cpu.pybind.geometry import KDTreeSearchParamHybrid
from open3d.cpu.pybind.pipelines.registration import registration_icp, \
	TransformationEstimationPointToPoint, evaluate_registration, \
	ICPConvergenceCriteria, TransformationEstimationPointToPlane

from camera.Frame import Frame
from ProjectObject import ProjectObject
from utils.transformation_utils import get_4x4_transform_from_quaternion, \
	get_4x4_transform_from_translation, find_3d_affine


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
		self.links: list[DMatch] = None
		"""The queryIdx is about the first frame and trainIdx is about the second"""
		self.links_inliers: list[DMatch] = []
		"""The queryIdx is about the first frame and trainIdx is about the second"""
		
		# Fundamental Matrix
		self.f_matrix = None
		
		# Inliers mask
		self.f_mask = None
		
		# Essential Matrix
		self.e_matrix = None
		
		# Roto-Translation
		self.R = None
		self.t = None
		self.pose = None
	
	def set_fundamental_matrix(self, matrix,
							   mask) -> None:
		"""Sets the fundamental matrix.
		
		:param matrix:
			The fundamental matrix to set.
			
		:param mask:
			The mask to set.
		
		:return:
			None.
		"""
		self.f_matrix = np.array(matrix)
		self.f_mask = np.array(mask)
	
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

	def set_inliers(
		self,
		new_mask=None
	):
		if new_mask is not None:
			self.f_mask = new_mask

		for i in range(len(self.links)):
			if self.f_mask[i] == 1:
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
			self.second.inliers.reshape(-1, 1, 2), 2, self.f_matrix
		)
		self.first.epi_lines = self.first.epi_lines.reshape(-1, 3)
		
		# second frame
		self.second.epi_lines = cv2.computeCorrespondEpilines(
			self.first.inliers.reshape(-1, 1, 2), 1, self.f_matrix
		)
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
	
	def roto_translation_pose(self, verbose: bool = True) -> Tuple[
		np.ndarray, np.ndarray]:
		"""Return the dataset pose transformations.

		:param verbose:
			States if detailed printing must be shown.

		:return:
			A tuple of transformations. The first transformation is the
			transformation of the first point cloud to the reference image and
			the second transformation is the transformation of the second point
			cloud to the reference image.
		"""
		pos = 1
		neg = -1
		quat_sign = np.array([pos, pos, pos, pos])
		pos_sign = np.array([pos, pos, pos])
		
		pose_2 = self.second.extract_pose()
		quat_2 = pose_2[0:4] * quat_sign
		pos_2 = pose_2[4:7] * pos_sign
		y = pos_2[1]
		z = pos_2[2]
		pos_2[1] = z
		pos_2[2] = y
		pose_1 = self.first.extract_pose()
		quat_1 = pose_1[0:4] * quat_sign
		pos_1 = pose_1[4:7] * pos_sign
		y = pos_1[1]
		z = pos_1[2]
		pos_1[1] = z
		pos_1[2] = y
		
		translation_1 = get_4x4_transform_from_translation(pos_1)
		rotation_1 = get_4x4_transform_from_quaternion(quat_1)
		translation_2 = get_4x4_transform_from_translation(pos_2)
		rotation_2 = get_4x4_transform_from_quaternion(quat_2)
		t_from_0_to_1 = np.matmul(translation_1, rotation_1)
		t_from_0_to_2 = np.matmul(translation_2, rotation_2)
		return np.linalg.inv(t_from_0_to_2), np.linalg.inv(t_from_0_to_1)
	
	def roto_translation_estimation_3d(self,
									   verbose: bool = True) -> np.ndarray:
		"""Find the rototranslation between point clouds by directly estimating it.

		:param verbose:
			States if detailed printing must be shown.

		:return:
			The matrix representing the transformation of the rototranslation
			from 2 to 1.
		"""
		self.compute_fundamental_matrix()
		self.compute_inliers()
		self.compute_epipolar_lines()
		
		# Retrieve all the points related to the couples of inlier points
		first_rgbd = self.first.get_rgbd_image()
		second_rgbd = self.second.get_rgbd_image()
		first_points = []
		second_points = []
		for inlier_match in self.links_inliers:
			# Descriptors and key points have the same indexes. Which means
			# that a descriptor inlier at 0 brings to a key point at 0
			first_keypoint = self.first.key_points[inlier_match.queryIdx]
			second_keypoint = self.second.key_points[inlier_match.trainIdx]
			
			# Get the points on the image of the key points of inliers
			first_img_point = [int(first_keypoint.pt[0]),
							   int(first_keypoint.pt[1])]
			second_img_point = [int(second_keypoint.pt[0]),
								int(second_keypoint.pt[1])]
			
			# Find for both image depth and depth scale, images are stored by
			# height x width and not width x height.
			first_d_scale = np.max(np.array(first_rgbd.depth))
			first_d = np.array(first_rgbd.depth)[first_img_point[1],
												 first_img_point[0]]
			second_d_scale = np.max(np.array(second_rgbd.depth))
			second_d = np.array(second_rgbd.depth)[second_img_point[1],
												   second_img_point[0]]
			
			# Black points are not detected by the camera
			if first_d != 0 and second_d != 0:
				# Project the point on the point cloud 3D space as specified by the
				# open3d documentation
				first_z = first_d * first_d_scale
				first_x = (first_img_point[
							   0] - self.first.Cx) * first_z / self.first.fx
				first_y = (first_img_point[
							   1] - self.first.Cy) * first_z / self.first.fy
				second_z = second_d * second_d_scale
				second_x = (second_img_point[
								0] - self.second.Cx) * first_z / self.second.fx
				second_y = (second_img_point[
								1] - self.second.Cy) * first_z / self.second.fy
				
				first_points.append([first_x, first_y, first_z])
				second_points.append([second_x, second_y, second_z])
		
		pcd_1 = np.asarray(self.first.get_point_cloud().points)
		pcd_2 = np.asarray(self.second.get_point_cloud().points)
		first_centroid = np.linalg.norm(pcd_1, axis=0).reshape(3)
		second_centroid = np.linalg.norm(pcd_2, axis=0).reshape(3)
		first_points = np.array(first_points)
		second_points = np.array(second_points)
		transformation = find_3d_affine(second_points,
										first_points,
										centroid1=first_centroid,
										centroid2=second_centroid,
										max_iter=300)
		print("ROT MATRIX NORM %s" % np.linalg.det(transformation[:3, :3]))
		if verbose:
			print("The computed transformation is %s" % transformation)
		
		return transformation
	
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
		t_from_2_to_0, t_from_1_to_0 = self.roto_translation_pose(verbose)
		t_from_0_to_1 = np.linalg.inv(t_from_1_to_0)
		t_from_0_to_2 = np.linalg.inv(t_from_2_to_0)
		t_from_2_to_1 = np.matmul(t_from_0_to_1, t_from_2_to_0)
		
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
								   t_from_2_to_1,
								   estimation_method=estimation,
								   criteria=ICPConvergenceCriteria(
									   max_iteration=2000))
		
		if verbose:
			evaluation = evaluate_registration(second_cloud,
											   first_cloud,
											   threshold,
											   icp_reg.transformation)
			print(
				"The computed transformation is:\n %s" % icp_reg.transformation)
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

	def pose_difference(
		self
	):
		"""
		Compute the new pose with respect to the two frames' poses expressed
		in quaternions.

		:return:
			The pose with respect to the two frames
		"""
		diff = [0.0] * 7
		for i in range(0, 4):
			diff[i] = self.second.pose[i] * self.first.pose[i] * (-1 if i > 0 else 1)
		for i in range(4, 7):
			diff[i] = self.second.pose[i] - self.first.pose[i]

		self.pose = np.array(diff)

		return self.pose

	def from_pose_to_rototrasl(
		self
	):
		q = np.ndarray([])
		q = np.append(q, -self.pose[1])
		q = np.append(q, -self.pose[2])
		q = np.append(q, -self.pose[3])
		q = np.append(q, -self.pose[0])
		self.t = self.pose[4:]

		s = sum(i ** 2 for i in q) ** (-2)

		# First row of the rotation matrix
		r00 = 1 - 2 * s * (q[2] ** 2 + q[3] ** 2)
		r01 = 2 * s * (q[1] * q[2] - q[0] * q[3])
		r02 = 2 * s * (q[1] * q[3] + q[0] * q[2])

		# Second row of the rotation matrix
		r10 = 2 * s * (q[1] * q[2] + q[0] * q[3])
		r11 = 1 - 2 * s * (q[1] ** 2 + q[3] ** 2)
		r12 = 2 * s * (q[2] * q[3] - q[0] * q[1])

		# Third row of the rotation matrix
		r20 = 2 * s * (q[1] * q[3] - q[0] * q[2])
		r21 = 2 * s * (q[2] * q[3] + q[0] * q[1])
		r22 = 1 - 2 * s * (q[1] ** 2 + q[2] ** 2)

		self.R = np.mat(
			[[r00, r01, r02],
			 [r10, r11, r12],
			 [r20, r21, r22]], dtype=float
		)
		self.R /= self.R[2, 2]

		# 3x3 rotation matrix
		return self.R, self.t

	def from_rototrasl_to_f_matrix(
		self
	):
		K = self.first.calibration_matrix()
		A = K @ self.R.T @ self.t
		C = np.mat(
			[[0.0, -A[0, 2], +A[0, 1]],
			 [+A[0, 2], 0.0, -A[0, 0]],
			 [-A[0, 1], +A[0, 0], 0.0]], dtype=float
		)
		self.f_matrix = np.linalg.inv(K).T @ self.R @ K.T @ C
		self.f_matrix /= self.f_matrix[2, 2]
		return self.f_matrix
