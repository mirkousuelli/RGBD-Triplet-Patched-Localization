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
	def calculate_homography(
		correspondences
	):
		"""
		Computers a homography from 4-correspondences.

		:param correspondences:
		:return:
		"""
		# loop through correspondences and create assemble matrix
		aList = []
		for corr in correspondences:
			p1 = np.matrix([corr.item(0), corr.item(1), 1])
			p2 = np.matrix([corr.item(2), corr.item(3), 1])

			a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1),
			      -p2.item(2) * p1.item(2), p2.item(1) * p1.item(0),
			      p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
			a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1),
			      -p2.item(2) * p1.item(2), 0, 0, 0, p2.item(0) * p1.item(0),
			      p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
			aList.append(a1)
			aList.append(a2)

		matrixA = np.matrix(aList)

		# svd composition
		u, s, v = np.linalg.svd(matrixA)

		# reshape the min singular value into a 3 by 3 matrix
		h = np.reshape(v[8], (3, 3))

		# normalize and now we have h
		h = (1 / h.item(8)) * h

		return h

	@staticmethod
	def homography_error(
		correspondences,
		homography
	):
		"""
		Calculate the geometric distance between estimated points and
		original points.

		:param correspondences:
		:param homography:
		:return:
		"""
		p1 = np.transpose(np.matrix([correspondences[0].item(0),
		                             correspondences[0].item(1), 1]))
		estimatep2 = np.dot(homography, p1)
		estimatep2 = (1/estimatep2.item(2))*estimatep2

		p2 = np.transpose(np.matrix([correspondences[0].item(2),
		                             correspondences[0].item(3), 1]))
		error = p2 - estimatep2
		return np.linalg.norm(error)

	def ransac_homography(
		self,
		action: Action,
		distance_error=5,
		threshold=0.4,
		iterations=1000,
		semantic=False
	):
		"""
		Compute a homography through RANSAC.

		:param action:
		:param distance_error:
		:param threshold:
		:param iterations:
		:param semantic:
		:return:
		"""
		num_pts = len(action.first.points)
		best_mask = [0] * num_pts
		h = None

		for i in range(iterations):
			# find 4 random points to calculate a homography
			prob = self.semantic_filter() if semantic else None
			rand_samples = np.random.choice(action.links, 4, p=prob)

			# call the homography function on those points
			h = self.calculate_homography(rand_samples)
			mask = []

			for link in action.links:
				mask.append(
					1 if self.homography_error(link, h) < distance_error else 0
				)

			best_mask = mask if np.cumsum(mask) > np.cumsum(best_mask) else best_mask

			if np.cumsum(best_mask) > (num_pts * threshold):
				break

		return h, best_mask

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
			error=5,
			threshold=0.4,
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
		f_matrix = None

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

			for _ in range(x1.shape[0]):
				mask.append(
					1 if self.fundamental_matrix_error(pt1, pt2, f_matrix) < error else 0
				)

			if sum(i for i in mask) > sum(i for i in best_mask):
				best_mask = mask

			if sum(i for i in best_mask) > (num_pts * threshold):
				break

		return f_matrix, best_mask
