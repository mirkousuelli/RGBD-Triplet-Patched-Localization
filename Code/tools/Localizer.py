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
		action.second.inliers = action.second.points[action.f_mask.ravel() == 1]

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
	def get_quaternions(action: Action):
		R = action.R
		trace = sum(R[i, i] for i in range(3))
		Q = [0] * 4

		if trace > 0.0:
			s = np.sqrt(trace + 1.0)
			Q[3] = s * 0.5
			s = 0.5 / s
			Q[0] = (R[2, 1] - R[1, 2]) * s
			Q[1] = (R[0, 2] - R[2, 0]) * s
			Q[2] = (R[1, 0] - R[0, 1]) * s
		else:
			if R[0, 0] < R[1, 1]:
				i = 2 if R[1, 1] < R[2, 2] else 1
			else:
				i = 2 if R[0, 0] < R[2, 2] else 0

			j = (i + 1) % 3
			k = (i + 2) % 3

			s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0)
			Q[i] = s * 0.5
			s = 0.5 / s

			Q[3] = (R[k, j] - R[j, k]) * s
			Q[j] = (R[j, i] - R[i, j]) * s
			Q[k] = (R[k, i] - R[i, k]) * s

		return np.array(Q)
