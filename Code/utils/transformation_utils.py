import random
from math import factorial, cos, sin

import numpy as np
from scipy.optimize import minimize, Bounds


def vec4(vec3: np.ndarray, fourth: int) -> np.ndarray:
	"""Extend a 3D vector to be 4D.
	
	:param vec3:
		3D vector to extend to be a 4D vector.
		
	:param fourth:
		Fourth element to add.
	
	:return:
		A 4D vector.
	"""
	if vec3.ndim != 1:
		raise ValueError("Only vectors can be extended")
	elif vec3.shape[0] != 3:
		raise ValueError("Only 3D vectors can be extended")
	
	return np.array([vec3[0], vec3[1], vec3[2], fourth])

def find_3d_affine(point_set1: np.ndarray,
				   point_set2: np.ndarray,
				   centroid1: np.ndarray,
				   centroid2: np.ndarray,
				   max_iter: int = 50) -> np.ndarray:
	"""Computes an affine 3d transformation from points set 1 to point set 2.
	
	:param point_set1:
		A set of 3D points.
	
	:param point_set2:
		A set of 3D points.
		
	:param centroid1:
		A set of 3D points.
		
	:param centroid2:
		A set of 3D points.
	
	:param max_iter:
		The maximum number of iterations to perform.
	
	:return:
		The 4x4 matrix representing the 3D affine transformation translating the
		point set 1 into point set 2.
	"""
	if point_set1.ndim != 2 or point_set2.ndim != 2:
		raise ValueError("Point sets must have 2 dimensions: number of points "
						 "and 3 (because points are 3D points)")
	elif point_set1.shape[1] != 3 or point_set2.shape[1] != 3:
		raise ValueError("Points must be 3D points")
	elif point_set1.shape != point_set2.shape:
		raise ValueError("The points' set must have the same shape")
	elif point_set1.shape[0] < 4:
		raise ValueError("To estimate a 3D affine transformation the minimal "
						 "number of points is 4.")
	elif centroid1.ndim != 1 or centroid2.ndim != 1:
		raise ValueError("Centroids must have 1 dimension")
	elif centroid1.shape[0] != 3 or centroid2.shape[0] != 3:
		raise ValueError("Centroids must be 3D points")
	elif max_iter < 1:
		raise ValueError("Max iterations must be strictly positive")
	
	num_pts = point_set1.shape[0]
	max_comb = factorial(num_pts) / (factorial(4) * factorial(num_pts - 4))
	max_iter = int(max_comb) if max_comb < max_iter else int(max_iter)
	
	homo_set1 = []
	homo_set2 = []
	for i in range(point_set1.shape[0]):
		homo_set1.append(vec4(point_set1[i], 1))
		homo_set2.append(vec4(point_set2[i], 1))
	homo_set1 = np.array(homo_set1)
	homo_set2 = np.array(homo_set2)
	
	# We know that an affine 3d transformation is:
	# [ a b c d ] P1 = P2  ==>	[ a b c d ] = P2 P1^-1
	# [ e f g h ] 				[ e f g h ]
	# [ i l m n ]				[ i l m n ]
	# In which P1 and P2 are the matrices of the homogeneous points of the first
	# and second set of points.
	
	best_config = None
	best_error = np.inf
	for i in range(max_iter):
		points = random.sample(range(num_pts), 4)
		P1 = np.array([homo_set1[points[0]],
					   homo_set1[points[1]],
					   homo_set1[points[2]],
					   homo_set1[points[3]]]).T
		P2 = np.array([homo_set2[points[0]],
					   homo_set2[points[1]],
					   homo_set2[points[2]],
					   homo_set2[points[3]]]).T
		
		if np.linalg.matrix_rank(P1) == 4:
			# To compute a matrix, I need the rotation matrix and the translation
			# vector. Therefore, I need a matrix with unitary determinant that
			# also represent a rotation around the axis. Therefore, I use the
			# fact that a rototraslation can be computed a translation into the
			# center, the needed rotations and then the translation on the right
			# point. To do so, I need to minimize a function of 6 values: the
			# translation on x, y, z and the rotation angles around the axis
			def get_transform(x):
				# Get rotation angles and translation components
				x_rot, y_rot, z_rot = x[0], x[1], x[2]
				x_tra, y_tra, z_tra = x[3], x[4], x[5]
				
				center = get_4x4_transform_from_translation(-centroid1)
				x_mat = get_4x4_rotation_from_angle_axis(x_rot, 0)
				y_mat = get_4x4_rotation_from_angle_axis(y_rot, 1)
				z_mat = get_4x4_rotation_from_angle_axis(z_rot, 2)
				translation = get_4x4_transform_from_translation(np.array([x_tra, y_tra, z_tra]))
				return translation @ center @ z_mat @ y_mat @ x_mat @ np.linalg.inv(center)
			
			def transform_error(x, *args):
				T_ = get_transform(x)
				new_p1s = T_ @ P1
				return np.mean(np.linalg.norm(P2 - new_p1s, axis=1))
			
			ret = minimize(transform_error,
						   np.array([0, 0, 0, 0, 0, 0]),
						   bounds=Bounds([0, 0, 0, -np.inf, -np.inf, -np.inf],
										 [2*np.pi, 2*np.pi, 2*np.pi, np.inf, np.inf, np.inf]))
			T = get_transform(ret.x)
			new_points = T @ homo_set1.T
			error = np.mean(np.linalg.norm(homo_set2 - new_points.T, axis=1))
			
			if error < best_error:
				best_config = T
				best_error = error
				
	t = best_config
	
	return t

def get_4x4_transform_from_translation(translation: np.ndarray) -> np.ndarray:
	"""Compute a 4x4 transformation from a translation vector.
	
	:param translation:
		A vector representing a transformation in the 3D space, thus a vector
		with shape (3,). The first element is the translation on x.
	
	:return:
		A numpy 4x4 matrix representing a homogeneous transformation.
	"""
	return np.array([[1, 0, 0, translation[0]],
					 [0, 1, 0, translation[1]],
					 [0, 0, 1, translation[2]],
					 [0, 0, 0, 1]], dtype=np.double)

def get_4x4_rotation_from_angle_axis(angle: float, axis: int) -> np.ndarray:
	"""Compute the transformation from angle, axis pair.
	
	:param angle:
		The angle of the rotation to apply in radians.
	 
	:param axis:
		The axis on which a rotation must be performed.
		
	:return:
		A numpy 4x4 matrix representing a homogeneous transformation.
	"""
	if axis not in [0, 1, 2]:
		raise ValueError("Axis can be 0, 1, 2 for x, y, z")
	
	if axis == 0:
		return np.array([[1,			0,			0,			0],
						 [0,	 		cos(angle),	-sin(angle),0],
						 [0,	 		sin(angle),	cos(angle),	0],
						 [0,	 		0,			0,			1]])
	elif axis == 1:
		return np.array([[cos(angle),	0,			sin(angle),	0],
						 [0,	 		1,			0,			0],
						 [-sin(angle),	0,			cos(angle),	0],
						 [0,	 		0,			0,			1]])
	else:
		return np.array([[cos(angle),	-sin(angle),0,			0],
						 [sin(angle),	cos(angle),	0,			0],
						 [0,	 		0,			1,			0],
						 [0,	 		0,			0,			1]])

def get_4x4_transform_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
	"""Compute a 4x4 transformation from a quaternion.
	
	:param quaternion:
		A vector representing the coefficients of a quaternion, thus a vector
		with shape (4,). The quaternion convention is a + bi + cj + dk.
	
	:return:
		A numpy 4x4 matrix representing a homogeneous transformation.
	"""
	# Proxy to reduce expressions' length
	q = quaternion
	s = np.linalg.norm(quaternion)
	
	# A quaternion is translated into (a=0, b=1, c=2, d=3)
	#	[ 1 - 2s(c^2 + d^2)			2s(bc + ad)		   2s(bd - ac)		0]
	#	[	 2s(bc - ad)		 1 - 2s(b^2 + d^2)	   2s(cd + ab)		0]
	#	[	 2s(bd + ac)			2s(cd - ab)		1 - 2s(b^2 + c^2)	0]
	#	[		  0						 0				    0			1]
	# Where s is its norm.
	r11 = 1 - 2 * s * (q[2] ** 2 + q[3] ** 2)
	r12 = 2 * s * (q[1] * q[2] + q[0] * q[3])
	r13 = 2 * s * (q[1] * q[3] - q[0] * q[2])
	r21 = 2 * s * (q[1] * q[2] - q[0] * q[3])
	r22 = 1 - 2 * s * (q[1] ** 2 + q[3] ** 2)
	r23 = 2 * s * (q[2] * q[3] + q[0] * q[1])
	r31 = 2 * s * (q[1] * q[3] + q[0] * q[2])
	r32 = 2 * s * (q[2] * q[3] - q[0] * q[1])
	r33 = 1 - 2 * s * (q[1] ** 2 + q[2] ** 2)
	return np.array([[r11, r12, r13, 0],
					 [r21, r22, r23, 0],
					 [r31, r32, r33, 0],
					 [0,   0,   0,   1]], dtype=np.double)

def get_3x3_rotation_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
	"""From Quaternions to Rotation Matrix.

	:param quaternion:
		Quaternions.

	:return:
		Rotation Matrix.
	"""
	q = quaternion
	# scale term
	s = np.linalg.norm(quaternion)
	
	# A quaternion is translated into (a=0, b=1, c=2, d=3)
	#	[ 1 - 2s(c^2 + d^2)			2s(bc - ad)		   2s(bd + ac)		]
	#	[	 2s(bc + ad)		 1 - 2s(b^2 + d^2)	   2s(cd - ab)		]
	#	[	 2s(bd - ac)			2s(cd + ab)		1 - 2s(b^2 + c^2)	]
	# Where s is its norm.
	
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
				   [r20, r21, r22]], dtype=np.float)
