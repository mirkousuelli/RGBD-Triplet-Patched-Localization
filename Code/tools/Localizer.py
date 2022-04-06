"""
Project : RGB-D Semantic Sampling
Authors : Marco Petri and Mirko Usuelli
-----------------------------------------------
Degree : M.Sc. Computer Science and Engineering
Course : Image Analysis and Computer Vision
Professor : Vincenzo Caglioti
Advisors : Giacomo Boracchi, Luca Magri
University : Politecnico di Milano - A.Y. 2021/2022
"""
from typing import Union, Tuple

import cv2
import numpy as np

from camera.Action import Action
from camera.Frame import Frame


class Localizer:
	"""
	Class implementing the tool 'Localizer' to apply stereo-vision operations
	such as epipolar geometry computation, fundamental matrix,
	camera localization, ... etc, w.r.t an Action (i.e. two frames, where the
	former is the reference one).
	"""
	# RANSAC hyper-parameters
	RANSAC_THRESHOLD_PIXEL = 0.1
	RANSAC_PROB = 0.999
	RANSAC_ITER = 10000

	def __init__(
		self
	):
		"""
		Constructor is empty because all the methods are static.
		"""
		return

	@staticmethod
	def compute_fundamental_matrix(
		action: Action,
		inplace=True
	):
		"""
		Compute the Fundamental matrix between the two frames inside the action.

		:param action:
			Couple of frame.
		:return: Action

		:param inplace:
            If the operation must happen inplace
        :type inplace: bool

        :returns:
            Fundamental matrix and inliers mask
		"""
		# pre-conditions
		assert len(action.first.points) == len(action.second.points), \
			"Frames features array have different size!"
		assert len(action.first.points) > 0, "Features array is empty!"

		# Fundamental Matrix + RANSAC
		F, mask = cv2.findFundamentalMat(
			action.first.points,
			action.second.points,
			cv2.FM_RANSAC,
			ransacReprojThreshold=Localizer.RANSAC_THRESHOLD_PIXEL,
			confidence=Localizer.RANSAC_PROB,
			maxIters=Localizer.RANSAC_ITER
		)

		if F is None or F.shape == (1, 1):
			# no fundamental matrix found
			raise Exception('No fundamental matrix found')
		elif F.shape[0] > 3:
			# more than one matrix found, just pick the first
			F = F[0:3, 0:3]

		# returning
		if inplace:
			action.f_matrix = np.mat(F)
			action.f_mask = mask
		else:
			return np.mat(F), mask

	@staticmethod
	def compute_inliers(
		action: Action
	):
		"""
		This static method computes the inlier through the previously computed
		mask through RANSAC and afterwards selects all the matches between
		inliers.

		:param action:
			The action which needs to be used for the inliers' extraction
		:type action: Action
		"""
		# pre-conditions
		assert len(action.first.points) == len(action.second.points), \
			"Frames features array have different size!"
		assert len(action.first.points) > 0, "Features array is empty!"
		assert action.f_mask is not None, "You must compute the Fundamental " \
		                                  "Matrix before!"

		# We select only inlier points
		action.first.inliers = action.first.points[action.f_mask.ravel() == 1]
		action.second.inliers = action.second.points[action.f_mask.ravel() == 1]

		# Looking for links through the previous selected inliers
		for i in range(len(action.links)):
			if action.f_mask[i].ravel() == 1:
				action.links_inliers.append(action.links[i])

	@staticmethod
	def compute_essential_matrix(
		action: Action,
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
		if action.f_matrix is None:
			Localizer.compute_fundamental_matrix(action, inplace)

		# essential matrix computation
		if inplace:
			action.e_matrix = action.second.calibration_matrix().T \
			                  @ action.f_matrix @ action.first.calibration_matrix()
		else:
			return action.second.calibration_matrix().T @ action.f_matrix \
			       @ action.first.calibration_matrix()

	@staticmethod
	def compute_epipolar_lines(
		action: Action
	):
		"""
		Compute the correspondent epipolar lines for both frames involved
		within the action

		:param action:
			Action consisting of two frames.
		:type action:
		"""
		# pre-conditions
		assert action.f_matrix is not None, "You must compute the " \
		                                    "Fundamental Matrix before!"

		# first frame
		action.first.epi_lines = cv2.computeCorrespondEpilines(
			action.second.inliers.reshape(-1, 1, 2), 2, action.f_matrix)
		action.first.epi_lines = action.first.epi_lines.reshape(-1, 3)

		# second frame
		action.second.epi_lines = cv2.computeCorrespondEpilines(
			action.first.inliers.reshape(-1, 1, 2), 1, action.f_matrix)
		action.second.epi_lines = action.second.epi_lines.reshape(-1, 3)

	@staticmethod
	def __draw_epipolar_lines(frame_1: Frame,
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

	@staticmethod
	def show_epipolar_lines(action: Action):
		"""
		Show the epipolar lines on both frames within the Action.

		:param action:
			Action consisting of two frames.
		:type action: Action

		:return:
			Final image of the two frames with the epipolar lines drawn
		"""
		# pre-conditions
		assert action.first.epi_lines is not None \
		       and action.second.epi_lines is not None, "You must compute the " \
		                                                "Epipolar Lines before!"

		# draw epipolar lines for each of the two frames in the action object
		img_1 = Localizer.__draw_epipolar_lines(action.first, action.second)
		img_2 = Localizer.__draw_epipolar_lines(action.second, action.first)

		# return the final image
		return np.concatenate((img_1, img_2), axis=1)

	@staticmethod
	def roto_translation(
		action: Action,
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
		if action.e_matrix is None:
			if inplace:
				Localizer.compute_essential_matrix(action, inplace)
			else:
				e_matrix = Localizer.compute_essential_matrix(action, inplace)

		if normalize_em and inplace:
			action.normalize_essential_matrix()
		
		# SVD decomposition of the essential matrix
		w, u, vt = cv2.SVDecomp(action.e_matrix if inplace else e_matrix)

		# determinant adjustments
		if np.linalg.det(u) < 0:
			u *= -1.0
		if np.linalg.det(vt) < 0:
			vt *= -1.0

		W = np.mat([[0, -1, 0],
		            [1, 0, 0],
		            [0, 0, 1]], dtype=float)
		
		if normalize_em and inplace:
			Localizer.compute_essential_matrix(action, inplace)

		# return the roto-translation components
		if inplace:
			action.R = np.mat(u) * W * np.mat(vt)
			action.t = u[:, 2]
		else:
			return np.mat(u) * W * np.mat(vt), u[:, 2]

	@staticmethod
	def from_rot_to_quat(
		action: Action,
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
		if action.R is None:
			Localizer.roto_translation(action, normalize_em=normalize_em)

		# storing locally the rotation matrix for the sake of simplicity
		R = action.R

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

	@staticmethod
	def from_quat_to_rot(q):
		"""
		From Quaternions to Rotation Matrix.

		:param q:
			Quaternions.

		:return:
			Rotation Matrix.
		"""
		# scale term
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

		# 3x3 rotation matrix
		return np.mat([[r00, r01, r02],
		               [r10, r11, r12],
		               [r20, r21, r22]], dtype=float)
