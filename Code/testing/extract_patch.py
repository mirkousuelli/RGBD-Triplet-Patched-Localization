import numpy as np

from camera.Action import Action
from camera.Frame import Frame
from tools.Merger import Merger
from utils.utils import get_str

num_samples = 5
first = 0
second = 60
frame1 = Frame(
	"../../Dataset/Testing/2/Colors/00" + get_str(first) + "-color.png",
	"../../Dataset/Testing/2/Depths/00" + get_str(first) + "-depth.png",
	first)
frame2 = Frame(
	"../../Dataset/Testing/2/Colors/00" + get_str(second) + "-color.png",
	"../../Dataset/Testing/2/Depths/00" + get_str(second) + "-depth.png",
	second)
action = Action(frame1, frame2)

merger = Merger(num_features=5000,
				detector_method="ORB",
				matcher_method="FLANN")

merge_image = merger.merge_action(action)
action.compute_fundamental_matrix()
action.compute_essential_matrix()
action.roto_translation(normalize_em=False)
action.compute_inliers()
action.compute_epipolar_lines()

# Generate always the same numbers
rng = np.random.default_rng(22)
matches = action.links_inliers
num_samples = num_samples if num_samples < len(matches) else len(matches)
selected_features = rng.choice(len(matches), num_samples, replace=False)
features = [action.first.key_points[matches[x].queryIdx] for x in
			selected_features]

# I get all the keypoints of the first image and of the second image
first_keys = [key.pt
			  for key in features]
first_keys = [np.array([pt[0], pt[1], 1])
			  for pt in first_keys.copy()]
first_keys = np.array(first_keys)

second_keys = [key.pt
			   for key in action.second.key_points]
second_keys = [np.array([pt[0], pt[1], 1])
			   for pt in second_keys.copy()]
second_keys = np.array(second_keys)

# I find all the transformed points using the fundamental matrix
triplets = []
for key_point in first_keys:
	x1Fx2 = [key_point @ action.f_matrix @ np.transpose(key2)
			 for key2 in second_keys]
	x1Fx2 = np.absolute(x1Fx2)
	pos_idx = np.argmin(x1Fx2)
	neg_idx = np.argmax(x1Fx2)
	
	# I add the triplet anchor-positive-negative to the triplets list
	triplets.append(np.array([key_point,
							  second_keys[pos_idx],
							  second_keys[neg_idx]]))

# I convert the triplet list to a numpy array
triplets = np.array(triplets)
patch_side = 8

###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
# I extract all the patches
patches = []
for triplet in triplets:
	# Get rgb and depth for both images
	first_rgbd = action.first.get_rgbd_image()
	first_color = np.asarray(first_rgbd.color)
	first_depth = np.asarray(first_rgbd.depth)
	second_rgbd = action.second.get_rgbd_image()
	second_color = np.asarray(second_rgbd.color)
	second_depth = np.asarray(second_rgbd.depth)
	w = first_color.shape[1]
	h = first_color.shape[0]
	
	triplet_patch = []
	for idx, coord in enumerate(triplet):
		patch = np.zeros((1 + 2*patch_side, 1 + 2*patch_side, 4))
		xc = coord[0]
		yc = coord[1]
		
		# Iterate taking care of border cases
		for x_off in range(2*patch_side + 1):
			for y_off in range(2*patch_side + 1):
				xo = int(max(0, min(xc - patch_side, w - 1 - 2*patch_side)) + x_off)
				yo = int(max(0, min(yc - patch_side, h - 1 - 2*patch_side)) + y_off)
				if idx == 0:
					color_img = first_color
					depth_img = first_depth
				else:
					color_img = second_color
					depth_img = second_depth
				
				patch[y_off, x_off, 0] = color_img[yo, xo, 0]
				patch[y_off, x_off, 1] = color_img[yo, xo, 1]
				patch[y_off, x_off, 2] = color_img[yo, xo, 2]
				patch[y_off, x_off, 3] = depth_img[yo, xo]
				
		triplet_patch.append(patch)
		
	triplet_patch = np.array(triplet_patch)
	patches.append(triplet_patch)
	
patches = np.array(patches)

print(patches)
