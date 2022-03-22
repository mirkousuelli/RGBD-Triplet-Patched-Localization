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
	def fundamental_matrix(action: Action):
		"""

		:param action:
		:return:
		"""
		assert len(action.first.points) == len(action.second.points)

		F, mask = cv2.findFundamentalMat(
			action.first.points,
			action.second.points,
			cv2.FM_RANSAC,
			ransacReprojThreshold=Localizer.RANSAC_THRESHOLD_PIXEL,
			confidence=Localizer.RANSAC_PROB,
			maxIters=Localizer.RANSAC_ITER
		)

		# We select only inlier points
		action.first.inliers = action.first.points[mask.ravel() == 1]
		action.second.inliers = action.second.points[mask.ravel() == 1]

		if F is None or F.shape == (1, 1):
			# no fundamental matrix found
			raise Exception('No fundamental matrix found')
		elif F.shape[0] > 3:
			# more than one matrix found, just pick the first
			F = F[0:3, 0:3]

		action.f_matrix = F

		return np.matrix(F)

	@staticmethod
	def compute_epipolar_lines(action: Action):
		"""

		:param action:
		:return:
		"""
		if action.f_matrix is None:
			Localizer.fundamental_matrix(action)

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
		if action.first.epi_lines is None or action.second.epi_lines is None:
			Localizer.compute_epipolar_lines(action)

		img_1 = Localizer.__draw_epipolar_lines(action.first, action.second)
		img_2 = Localizer.__draw_epipolar_lines(action.second, action.first)

		final_img = np.concatenate((img_1, img_2), axis=1)
		return final_img
