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


class Localizer:
	def __init__(self):
		pass

	@staticmethod
	def key_points_to_array(key_points_1, key_points_2, matches):
		pts_1 = []
		pts_2 = []

		# ratio test as per Lowe's paper
		for m, n in matches:
			if m.distance < 0.7 * n.distance:
				pts_1.append(key_points_1[m.queryIdx].pt)
				pts_2.append(key_points_2[m.trainIdx].pt)
		return pts_1, pts_2

	def fundamental_matrix(self, key_points_1, key_points_2, matches):
		assert points_img_1.shape[0] == points_img_2.shape[0]

		arr_a = np.column_stack((points_img_1, [1] * points_img_1.shape[0]))
		arr_b = np.column_stack((points_img_2, [1] * points_img_2.shape[0]))

		arr_a = np.tile(arr_a, 3)
		arr_b = arr_b.repeat(3, axis=1)
		A = np.multiply(arr_a, arr_b)

		_, _, V = np.linalg.svd(A)
		F_matrix = V[-1]
		F_matrix = np.reshape(F_matrix, (3, 3))

		# resolve det(F) = 0 constraint using SVD
		U, S, V = np.linalg.svd(F_matrix)
		S[-1] = 0

		return U @ np.diagflat(S) @ V

	def estimate_fundamental_matrix_with_normalize(Points_a, Points_b):
		# Try to implement this function as efficiently as possible. It will be
		# called repeatly for part III of the project
		#

		#                                              [f11
		# [u1u1' v1u1' u1' u1v1' v1v1' v1' u1 v1 1      f12     [0
		#  u2u2' v2v2' u2' u2v2' v2v2' v2' u2 v2 1      f13      0
		#  ...                                      *   ...  =  ...
		#  ...                                          ...     ...
		#  unun' vnun' un' unvn' vnvn' vn' un vn 1]     f32      0]
		#                                               f33]
		assert Points_a.shape[0] == Points_b.shape[0]

		mean_a = Points_a.mean(axis=0)
		mean_b = Points_b.mean(axis=0)
		std_a = np.sqrt(np.mean(np.sum((Points_a-mean_a)**2, axis=1), axis=0))
		std_b = np.sqrt(np.mean(np.sum((Points_b-mean_b)**2, axis=1), axis=0))

		Ta1 = np.diagflat(np.array([np.sqrt(2)/std_a, np.sqrt(2)/std_a, 1]))
		Ta2 = np.column_stack((np.row_stack((np.eye(2), [[0, 0]])), [-mean_a[0], -mean_a[1], 1]))

		Tb1 = np.diagflat(np.array([np.sqrt(2)/std_b, np.sqrt(2)/std_b, 1]))
		Tb2 = np.column_stack((np.row_stack((np.eye(2), [[0, 0]])), [-mean_b[0], -mean_b[1], 1]))

		Ta = np.matmul(Ta1, Ta2)
		Tb = np.matmul(Tb1, Tb2)

		arr_a = np.column_stack((Points_a, [1]*Points_a.shape[0]))
		arr_b = np.column_stack((Points_b, [1]*Points_b.shape[0]))

		arr_a = np.matmul(Ta, arr_a.T)
		arr_b = np.matmul(Tb, arr_b.T)

		arr_a = arr_a.T
		arr_b = arr_b.T

		arr_a = np.tile(arr_a, 3)
		arr_b = arr_b.repeat(3, axis=1)
		A = np.multiply(arr_a, arr_b)

		'''Solve f from Af=0'''
		'''solution 1'''
		U, s, V = np.linalg.svd(A)
		F_matrix = V[-1]
		F_matrix = np.reshape(F_matrix, (3, 3))
		F_matrix /= np.linalg.norm(F_matrix)

		'''solution 2'''
		# b = A[:, 0].copy()
		# F_matrix = np.linalg.lstsq(A[:, 1:], -b)[0]
		# F_matrix = np.r_[1, F_matrix]
		# F_matrix = F_matrix.reshape((3, 3))

		'''Resolve det(F) = 0 constraint using SVD'''
		U, S, Vh = np.linalg.svd(F_matrix)
		S[-1] = 0
		F_matrix = U @ np.diagflat(S) @ Vh

		F_matrix = Tb.T @ F_matrix @ Ta
		return F_matrix

	# Find the best fundamental matrix using RANSAC on potentially matching
	# points
	# 'matches_a' and 'matches_b' are the Nx2 coordinates of the possibly
	# matching points from pic_a and pic_b. Each row is a correspondence (e.g.
	# row 42 of matches_a is a point that corresponds to row 42 of matches_b.
	# 'Best_Fmatrix' is the 3x3 fundamental matrix
	# 'inliers_a' and 'inliers_b' are the Mx2 corresponding points (some subset
	# of 'matches_a' and 'matches_b') that are inliers with respect to
	# Best_Fmatrix.
	def ransac_fundamental_matrix(matches_a, matches_b):
		# For this section, use RANSAC to find the best fundamental matrix by
		# randomly sampling interest points. You would reuse
		# estimate_fundamental_matrix() from part 2 of this assignment.
		# If you are trying to produce an uncluttered visualization of epipolar
		# lines, you may want to return no more than 30 points for either left or
		# right images.

		# Your ransac loop should contain a call to 'estimate_fundamental_matrix()'
		# that you wrote for part II.

		num_iterator = 10000
		threshold = 0.002
		best_F_matrix = np.zeros((3, 3))
		max_inlier = 0
		num_sample_rand = 8

		xa = np.column_stack((matches_a, [1]*matches_a.shape[0]))
		xb = np.column_stack((matches_b, [1]*matches_b.shape[0]))
		xa = np.tile(xa, 3)
		xb = xb.repeat(3, axis=1)
		A = np.multiply(xa, xb)

		for i in range(num_iterator):
			index_rand = np.random.randint(matches_a.shape[0], size=num_sample_rand)
			F_matrix = estimate_fundamental_matrix_with_normalize(matches_a[index_rand, :], matches_b[index_rand, :])
			err = np.abs(np.matmul(A, F_matrix.reshape((-1))))
			current_inlier = np.sum(err <= threshold)
			if current_inlier > max_inlier:
				best_F_matrix = F_matrix.copy()
				max_inlier = current_inlier

		err = np.abs(np.matmul(A, best_F_matrix.reshape((-1))))
		index = np.argsort(err)
		# print(best_F_matrix)
		# print(np.sum(err <= threshold), "/", err.shape[0])
		return best_F_matrix, matches_a[index[:29]], matches_b[index[:29]]
