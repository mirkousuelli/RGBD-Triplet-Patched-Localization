import numpy as np
import scipy

from camera.Action import Action


class SemanticRANSAC:
	"""

	"""

	def semantic_filter(
		self
	):
		pass

	@staticmethod
	def calculate_fundamental_matrix(
		x1,
		x2
	):
		"""

		:param x1:
		:param x2:
		:return:
		"""
		assert x1.shape[0] == x2.shape[0], "Number of points don't match."

		n = x1.shape[0]
		# build matrix for equations
		A = np.zeros((n, 9))
		for i in range(n):
			A[i] = [
				x1[i, 0] * x2[i, 0], x1[i, 0] * x2[i, 1], x1[i, 0] * 1,
				x1[i, 1] * x2[i, 0], x1[i, 1] * x2[i, 1], x1[i, 1] * 1,
				1 * x2[i, 0], 1 * x2[i, 1], 1 * 1
			]

		# compute linear least square solution
		U, S, V = scipy.linalg.svd(A)
		F = V[-1].reshape(3, 3)

		# constrain F
		# make rank 2 by zeroing out last singular value
		U, S, V = scipy.linalg.svd(F)
		S[2] = 0
		F = np.dot(U, np.dot(np.diag(S), V))

		return F / F[2, 2]

	@staticmethod
	def fundamental_matrix_error(
		x1,
		x2,
		f_matrix
	):
		"""

		:param x1:
		:param x2:
		:param f_matrix:
		:return:
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
			error=0.7,
			threshold=0.3,
			iterations=1000,
			semantic=False
	):
		"""

		:param action:
		:param error:
		:param threshold:
		:param iterations:
		:param semantic:
		:return:
		"""
		num_pts = min(len(action.first.points), len(action.second.points))
		best_mask = [0] * num_pts
		best_f_matrix = None

		for _ in range(iterations):
			# find 8 random points
			prob = self.semantic_filter() if semantic else None
			rand_idx = np.random.choice(range(len(action.links)), 8, p=prob)
			rand_samples = [action.links[i] for i in rand_idx]

			x1, x2 = [], []
			for m in rand_samples:
				x1.append(action.first.key_points[m.queryIdx].pt)
				x2.append(action.second.key_points[m.trainIdx].pt)
			x1, x2 = np.array(np.int32(x1)), np.array(np.int32(x2))
			pt1, pt2 = np.zeros(shape=(8, 3)), np.zeros(shape=(8, 3))
			for i in range(8):
				pt1[i, 0], pt1[i, 1], pt1[i, 2] = x1[i, 0], x1[i, 1], 1
				pt2[i, 0], pt2[i, 1], pt2[i, 2] = x2[i, 0], x2[i, 1], 1

			# call the homography function on those points
			f_matrix = self.calculate_fundamental_matrix(pt1, pt2)
			mask = []

			for m in action.links:
				t1 = np.int32(action.first.key_points[m.queryIdx].pt)
				t2 = np.int32(action.second.key_points[m.trainIdx].pt)
				t1 = np.matrix([t1[0], t1[1], 1])
				t2 = np.matrix([t2[0], t2[1], 1])
				if self.fundamental_matrix_error(t1, t2, f_matrix) < error:
					mask.append(1)
				else:
					mask.append(0)

			if sum(i for i in mask) > sum(i for i in best_mask):
				best_mask = mask
				best_f_matrix = f_matrix

			if sum(i for i in best_mask) > (num_pts * threshold):
				break

		return best_f_matrix, best_mask
