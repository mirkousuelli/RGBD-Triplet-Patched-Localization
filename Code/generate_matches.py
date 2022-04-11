import numpy as np

from camera.Action import Action
from camera.Frame import Frame
from tools.Merger import Merger
from utils.utils import get_str

def multiply_quaternions(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
	"""Gets to quaternions and multiply them.
	
	:param q1: First quaternion.
	:param q2: Second quaternion.
	:return: Resulting quaternion.
	
	Notes
	-----
	See https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation for more
	details on quaternion multiplication.
	"""
	new_a = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
	new_b = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
	new_c = q1[0]*q2[2] + q1[2]*q2[0] + q1[3]*q2[1] - q1[1]*q2[3]
	new_d = q1[0]*q2[3] + q1[3]*q2[0] + q1[1]*q2[2] - q1[2]*q2[1]
	return np.array([new_a, new_b, new_c, new_d])

verbose = True
where_to_write = "localization/01_direct.matches"
matches_str = "1 0 0 0 0 0 0"
for i in range(2, 834, 1):
	first_img = Frame("../Dataset/Colors/00" + get_str(i-1) + "-color.png",
					  "../Dataset/Depths/00" + get_str(i-1) + "-depth.png",
					  i-1)
	second_img = Frame("../Dataset/Colors/00" + get_str(i) + "-color.png",
					   "../Dataset/Depths/00" + get_str(i) + "-depth.png",
					   i)
	action = Action(first_img, second_img)
	
	if verbose:
		print("Executing step: ", i)
	
	merger = Merger(num_features=5000,
					detector_method="ORB",
					matcher_method="FLANN")
	merge_image = merger.merge_action(action)
	
	action.compute_fundamental_matrix()
	action.compute_essential_matrix()
	action.roto_translation(normalize_em=False)
	q = action.from_rot_to_quat(normalize_em=False)
	matches_str += "\n" + str(q[0]) + " " + str(q[1]) + " " + str(q[2]) + " " + str(q[3])
	matches_str += " " + str(action.t[0]) + " " + str(action.t[1]) + " " + str(action.t[2])
	print("Figure ", action.first._Frame__color_path, " and figure ", action.second._Frame__color_path)
	print(str(q[0]) + " " + str(q[1]) + " " + str(q[2]) + " " + str(q[3]) + " " + str(action.t[0]) + " " + str(action.t[1]) + " " + str(action.t[2]))
	del action
	del first_img
	del second_img
	del q
	del merge_image
	del merger

clean_matches = "1 0 0 0 0 0 0"

overall_quaternion = np.array([1, 0, 0, 0])
overall_translation = np.array([0, 0, 0])
lines = matches_str.split("\n")
for i in range(1, 834, 1):
	break
	relative_pos = lines[i].split(" ")
	relative_pos = np.array(relative_pos, dtype=float)
	
	quaternion = relative_pos[0:4]
	position = relative_pos[4:7]
	
	if verbose:
		print("Getting results step: ", i)
	
	overall_quaternion = multiply_quaternions(overall_quaternion, quaternion)
	overall_translation = overall_translation + position
	
	q = overall_quaternion
	t = overall_translation
	
	clean_matches += "\n" + str(q[0]) + " " + str(q[1]) + " " + str(q[2]) + " " + str(q[3])
	clean_matches += " " + str(t[0]) + " " + str(t[1]) + " " + str(t[2])

#f = open(where_to_write, "w")
#f.write(matches_str)
#f.close()

