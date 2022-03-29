"""
Project : RGB-D Semantic Sampling
Authors : Marco Petri and Mirko Usuelli
--------------------------------------------------------------------------------
Degree : M.Sc. Computer Science and Engineering
Course : Image Analysis and Computer Vision
Professor : Vincenzo Caglioti
Advisors : Giacomo Boracchi, Luca Magri
University : Politecnico di Milano - A.Y. 2021/2022
"""
import numpy as np
import cv2

from camera.Action import Action
from camera.Frame import Frame
from tools.Matcher import Matcher


class Localizer:
	"""

	"""
	RANSAC_THRESHOLD_PIXEL = 0.1
	RANSAC_PROB = 0.999
	RANSAC_ITER = 10000

	def __init__(self):
		return

	@staticmethod
	def compute_fundamental_matrix(action: Action):
		"""

		:param action:
		:return:
		"""
		assert len(action.first.points) == len(action.second.points), \
			"Frames features array have different size!"
		assert len(action.first.points) > 0, "Features array is empty!"

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

		action.f_matrix = np.mat(F)
		action.f_mask = mask

		return np.matrix(F)

	@staticmethod
	def compute_inliers(action: Action):
		assert len(action.first.points) == len(action.second.points), \
			"Frames features array have different size!"
		assert len(action.first.points) > 0, "Features array is empty!"
		assert action.f_mask is not None, "You must compute the Fundamental " \
		                                  "Matrix before!"

		# We select only inlier points
		action.first.inliers = action.first.points[action.f_mask.ravel() == 1]
		for inlier in action.first.inliers:
			new_key_point = cv2.KeyPoint(int(inlier[0]), int(inlier[1]), 5)
			action.first.key_points_inliers.append(new_key_point)
			index = -1
			for i in range(len(action.first.key_points)):
				if action.first.key_points[i].pt == new_key_point.pt:
					index = i
					break
			action.first.descriptors_inliers = np.append(
				action.first.descriptors_inliers,
				action.first.descriptors[index].copy()
			).reshape(
				(-1, action.first.descriptors.shape[1])
			)
		action.first.key_points_inliers = tuple(
			action.first.key_points_inliers
		)

		# We select only inlier points
		action.second.inliers = action.second.points[action.f_mask.ravel() == 1]
		for inlier in action.second.inliers:
			new_key_point = cv2.KeyPoint(int(inlier[0]), int(inlier[1]), 5)
			action.second.key_points_inliers.append(new_key_point)
			index = -1
			for i in range(len(action.second.key_points)):
				if action.second.key_points[i].pt == new_key_point.pt:
					index = i
					break
			action.second.descriptors_inliers = np.append(
				action.second.descriptors_inliers,
				action.second.descriptors[index].copy()
			).reshape(
				(-1, action.second.descriptors.shape[1])
			)
		action.second.key_points_inliers = tuple(
			action.second.key_points_inliers
		)

		for i in range(len(action.links)):
			if action.f_mask[i].ravel() == 1:
				action.links_inliers.append(action.links[i])

		#action.links_inliers = Matcher.just_match(
		#	action.first.descriptors_inliers,
		#	action.second.descriptors_inliers
		#)

	@staticmethod
	def compute_essential_matrix(action: Action):
		assert action.f_matrix is not None, "You must compute the " \
		                                    "Fundamental Matrix before!"

		action.e_matrix = action.second.calibration_matrix().T \
		                  @ action.f_matrix @ action.first.calibration_matrix()
		return action.e_matrix

	@staticmethod
	def compute_epipolar_lines(action: Action):
		"""

		:param action:
		:return:
		"""
		assert action.f_matrix is not None, "You must compute the " \
		                                    "Fundamental Matrix before!"

		action.first.epi_lines = cv2.computeCorrespondEpilines(
			action.second.inliers.reshape(-1, 1, 2), 2, action.f_matrix)
		action.first.epi_lines = action.first.epi_lines.reshape(-1, 3)

		action.second.epi_lines = cv2.computeCorrespondEpilines(
			action.first.inliers.reshape(-1, 1, 2), 1, action.f_matrix)
		action.second.epi_lines = action.second.epi_lines.reshape(-1, 3)

	@staticmethod
	def __draw_epipolar_lines(frame_1: Frame,
	                          frame_2: Frame):
		"""

		:param frame_1:
		:param frame_2:
		:return:
		"""
		img = frame_1.get_cv2_images(ret="rgb")
		_, c = frame_1.get_size()

		np.random.seed(42)

		for lines, pt1, pt2 in zip(frame_1.epi_lines,
		                           frame_1.inliers,
		                           frame_2.inliers):
			color = np.random.randint(0, 255, 3).tolist()

			x0, y0 = map(int, [0, -lines[2] / lines[1]])
			x1, y1 = map(int, [c, -(lines[2] + lines[0] * c) / lines[1]])
			img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
			img = cv2.circle(img, tuple(pt1), 5, color, -1)

		return img

	@staticmethod
	def show_epipolar_lines(action: Action):
		"""

		:param action:
		:return:
		"""
		assert action.first.epi_lines is not None \
		       and action.second.epi_lines is not None, "You must compute the " \
		                                                "Epipolar Lines before!"

		img_1 = Localizer.__draw_epipolar_lines(action.first, action.second)
		img_2 = Localizer.__draw_epipolar_lines(action.second, action.first)

		final_img = np.concatenate((img_1, img_2), axis=1)
		return final_img

	@staticmethod
	def roto_translation(action: Action):
		w, u, vt = cv2.SVDecomp(action.e_matrix)
		if np.linalg.det(u) < 0:
			u *= -1.0
		if np.linalg.det(vt) < 0:
			vt *= -1.0
		W = np.mat([[0, -1, 0],
		            [1, 0, 0],
		            [0, 0, 1]], dtype=float)
		action.R = np.mat(u) * W * np.mat(vt)
		action.t = u[:, 2]

		return action.R, action.t

	@staticmethod
	def from_rot_to_quat(action: Action):
		R = action.R

		q00 = R[0, 0] - R[1, 1] - R[2, 2]
		q01 = R[1, 0] + R[0, 1]
		q02 = R[2, 0] + R[0, 2]
		q03 = R[2, 1] - R[1, 2]

		q10 = q01
		q11 = R[1, 1] - R[0, 0] - R[2, 2]
		q12 = R[2, 1] + R[1, 2]
		q13 = R[0, 2] - R[2, 0]

		q20 = q02
		q21 = q12
		q22 = R[2, 2] - R[0, 0] - R[1, 1]
		q23 = R[1, 0] - R[0, 1]

		q30 = q03
		q31 = q13
		q32 = q23
		q33 = R[0, 0] + R[1, 1] + R[2, 2]

		Q = np.mat([[q00, q01, q02, q03],
		            [q10, q11, q12, q13],
		            [q20, q21, q22, q23],
		            [q30, q31, q32, q33]], dtype=float) / 3

		_, eig_vec = np.linalg.eigh(Q)

		q = eig_vec[:, 3]

		return np.array(np.concatenate((q[3].flatten().squeeze(),
		                                q[0].flatten().squeeze(),
		                                q[1].flatten().squeeze(),
		                                q[2].flatten().squeeze()))).squeeze()

	@staticmethod
	def from_quat_to_rot(q):
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
