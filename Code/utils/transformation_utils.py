import numpy as np


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
					 [0, 0, 0, 1]])

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
					 [0,   0,   0,   1]])
