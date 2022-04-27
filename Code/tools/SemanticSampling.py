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
import numpy as np
import scipy

from Code.camera.Action import Action

PATCH_SIDE = 8

class SemanticSampling:
	"""
	Semantic Sampling procedures for localization.
	"""

	@staticmethod
	def get_patch(
			patch_key_point,
			color_img,
			depth_img
	) -> np.ndarray:
		w = color_img.shape[1]
		h = color_img.shape[0]
		patch = np.zeros((4, 1 + 2 * PATCH_SIDE, 1 + 2 * PATCH_SIDE))
		xc = patch_key_point[0]
		yc = patch_key_point[1]

		# Iterate taking care of border cases
		for x_off in range(2 * PATCH_SIDE + 1):
			for y_off in range(2 * PATCH_SIDE + 1):
				xo = int(max(0, min(xc - PATCH_SIDE,
				                    w - 1 - 2 * PATCH_SIDE)) + x_off)
				yo = int(max(0, min(yc - PATCH_SIDE,
				                    h - 1 - 2 * PATCH_SIDE)) + y_off)

				patch[0, y_off, x_off] = color_img[yo, xo, 0]
				patch[1, y_off, x_off] = color_img[yo, xo, 1]
				patch[2, y_off, x_off] = color_img[yo, xo, 2]
				patch[3, y_off, x_off] = depth_img[yo, xo]

		return patch

	@staticmethod
	def calculate_fundamental_matrix(
		x1,
		x2
	):
		"""
		Method which computes the Fundamental Matrix.

		:param x1:
			Points set for the first frame (homogeneous coordinates).
		:type x1: ndarray

		:param x2:
			Points set for the second frame (homogeneous coordinates).
		:type x2: ndarray

		:return:
			Fundamental Matrix
		"""
		assert x1.shape[0] == x2.shape[0], "Number of points do not match."

		n = x1.shape[0]
		# build matrix for equations
		A = np.zeros((n, 9))
		for i in range(n):
			A[i] = [
				x1[i, 0] * x2[i, 0], x1[i, 0] * x2[i, 1], x1[i, 0] * x2[i, 2],
				x1[i, 1] * x2[i, 0], x1[i, 1] * x2[i, 1], x1[i, 1] * x2[i, 2],
				x1[i, 2] * x2[i, 0], x1[i, 2] * x2[i, 1], x1[i, 2] * x2[i, 2]
			]

		# compute linear least square solution
		U, S, V = scipy.linalg.svd(A)
		F = V[-1].reshape(3, 3)

		# constrain F
		# make rank 2 by zeroing out last singular value
		U, S, V = scipy.linalg.svd(F)
		S[2] = 0
		F = np.dot(U, np.dot(np.diag(S), V))

		# return the normalized matrix
		return F / F[2, 2]

	@staticmethod
	def fundamental_matrix_error(
		x1,
		x2,
		f_matrix
	):
		"""
		Internal loss function of RANSAC which check what is the quadratic error
		for the points sets x1, x2 belonging to the epipolar lines entailed by
		the computed fundamental matrix passed as parameter.

		Intuitively, Sampson error can be roughly thought as the squared
		distance between a point x to the corresponding epipolar line x'F.

		:param x1:
			Points set for the first frame (homogeneous coordinates).
		:type x1: np.matrix

		:param x2:
			Points set for the second frame (homogeneous coordinates).
		:type x2: np.matrix

		:param f_matrix:
			Fundamental matrix.
		:type f_matrix: np.matrix

		:return:
			Quadratic error of consensus acceptance
		"""
		# Sampson distance as error measure
		Fx1 = np.dot(f_matrix, x1.T)
		Fx2 = np.dot(f_matrix, x2.T)
		denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
		err = (np.diag(np.dot(x1, np.dot(f_matrix, x2.T)))) ** 2 / denom

		# return error
		return np.linalg.norm(err)

	def ransac_fundamental_matrix(
			self,
			action: Action,
			error=0.7,  # TODO: to be tuned!
			sampling_rate=0.3,
			iterations=1000,
			semantic=None
	):
		"""
		Custom version of RANSAC able to work with a pseudo-random function
		useful to entail a semantic score distribution if set to True, otherwise
		pure randomness is used.

		:param action:
			Action object entailing the two frames.
		:type action: Action

		:param error:
			Error value of consensus set acceptance.
		:type error: float

		:param sampling_rate:
			Percentage of desired inliers matches to be used as exit condition
			of acceptance.
		:type sampling_rate: float

		:param iterations:
			Number of iterations for RANSAC.
		:type iterations: int

		:param semantic:
			Probability distribution array.
		:param semantic: np.array

		:return:
			Fundamental Matrix, inliers matches mask
		"""
		# number of total links matched
		num_links = len(action.links)

		# best variables to be returned initialized
		best_mask = [0] * num_links
		best_f_matrix = None

		# counters
		best_consensus, consensus = 0, 0

		# ransac iterations
		for _ in range(iterations):
			# find 8 random points through a pseudo random set
			# starting from indexes
			rand_idx = np.random.choice(range(len(action.links)), 8, p=semantic)

			# taking the proper links through the computed indexes above
			rand_samples = [action.links[i] for i in rand_idx]

			# reading the points separately from the pseudo-random sampling
			x1, x2 = [], []
			for m in rand_samples:
				x1.append(action.first.key_points[m.queryIdx].pt)
				x2.append(action.second.key_points[m.trainIdx].pt)
			x1, x2 = np.array(np.int32(x1)), np.array(np.int32(x2))

			# setting the homogeneous coordinates matrices from the points
			# belonging to the pseudo-random sampling
			pt1, pt2 = np.zeros(shape=(8, 3)), np.zeros(shape=(8, 3))
			for i in range(8):
				pt1[i, 0], pt1[i, 1], pt1[i, 2] = x1[i, 0], x1[i, 1], 1
				pt2[i, 0], pt2[i, 1], pt2[i, 2] = x2[i, 0], x2[i, 1], 1

			# compute the fundamental matrix with this pseudo-random setting
			f_matrix = self.calculate_fundamental_matrix(pt1, pt2)

			# computing the consensus set represented by the inlier mask
			mask = []
			for m in action.links:
				# temporarily reading the 2D points
				t1 = np.int32(action.first.key_points[m.queryIdx].pt)
				t2 = np.int32(action.second.key_points[m.trainIdx].pt)

				# homogeneous coordinates
				t1 = np.matrix([t1[0], t1[1], 1])
				t2 = np.matrix([t2[0], t2[1], 1])

				# internal loss function for the consensus set
				mask.append(
					1 if self.fundamental_matrix_error(
						t1, t2, f_matrix
					) < error else 0
				)

			# consensus counters updating
			consensus = sum(i for i in mask)
			if consensus > best_consensus:
				best_consensus = consensus
				best_mask = mask
				best_f_matrix = f_matrix

			# exit condition
			if best_consensus > (num_links * sampling_rate):
				break

		# return the final result of the ransac process
		return best_f_matrix, best_mask
