import numpy as np

from Code.camera.Action import Action
from Code.camera.Frame import Frame
from Code.tools.Merger import Merger
from Code.utils.utils import *

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
	neg_idx = np.argmax(x1Fx2)

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
