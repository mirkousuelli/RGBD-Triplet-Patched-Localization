

def get_str(num) -> str:
	"""Returns a 4 digits string from a number.
	
	:param num:
		An integer to be translated into a string.
		
	:return:
		The string representing the number.
	:rtype str:
	"""
	return ("0" + str(num)) if num < 10 else str(num)


def get_rgb_triplet_dataset_path(
	root,
	folder,
	scene,
	num
) -> str:
	return root + '/' + folder + '/' + str(scene) + "/Colors/000" + \
	       get_str(num) + "-color.png"


def get_depth_triplet_dataset_path(
	root,
	folder,
	scene,
	num
) -> str:
	return root + '/' + folder + '/' + str(scene) + "/Depths/000" + \
	       get_str(num) + "-depth.png"


def get_pose_triplet_dataset_path(
	root,
	folder,
	scene
) -> str:
	return root + '/' + folder + '/' + str(scene) + "/Poses/" + \
	       get_str(scene) + ".pose"
