import numpy as np

from camera.Action import Action
from camera.Frame import Frame
from tools.Merger import Merger
from utils.utils import *

num_samples = 5
first = 0
second = 60
frame1 = Frame(
	get_rgb_triplet_dataset_path("../../Dataset", "Testing", 2, first),
	get_depth_triplet_dataset_path("../../Dataset", "Testing", 2, first),
	get_pose_triplet_dataset_path("../../Dataset", "Testing", 2),
	first
)
frame2 = Frame(
	get_rgb_triplet_dataset_path("../../Dataset", "Testing", 2, second),
	get_depth_triplet_dataset_path("../../Dataset", "Testing", 2, second),
	get_pose_triplet_dataset_path("../../Dataset", "Testing", 2),
	second
)
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

###############################################################
###############################################################
###############################################################
###############################################################
###############################################################

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
	has_found_good_neg = False
	
	while not has_found_good_neg:
		# Generate random negative
		neg_idx = rng.integers(low=0, high=x1Fx2.shape[0] - 1)
		
		# Compute distance between pos and neg
		pos_point = np.asarray(second_keys[pos_idx])
		neg_point = np.asarray(second_keys[neg_idx])
		dist = np.linalg.norm(pos_point - neg_point)
		
		# Check that neg is not in the circle of pos descriptor
		if dist > action.second.key_points[pos_idx].size / 2:
			has_found_good_neg = True

	# I add the triplet anchor-positive-negative to the triplets list
	triplets.append(
		np.array(
			[key_point,
			 second_keys[pos_idx],
			 second_keys[neg_idx]],
			dtype=np.int32
		)
	)

# I convert the triplet list to a numpy array
triplets = np.array(triplets)
print(triplets)
