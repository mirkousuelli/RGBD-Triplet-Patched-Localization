import numpy as np
import cv2

from camera.Action import Action
from camera.Frame import Frame
from tools.Merger import Merger
from tools.SemanticSampling import SemanticSampling
from tools.Visualizer import Visualizer
from utils.utils import get_rgb_triplet_dataset_path, get_depth_triplet_dataset_path, get_pose_triplet_dataset_path

DETECTION_METHOD = "ORB"
first = 0
second = 60

# PIPELINE TO PERFORM SEMANTIC SAMPLING USING CLASSICAL COMPUTER VISION
# PHASE 1: DETECTION
# Description: Using SIFT or ORB, features are detected using classical computer
# vision techniques.
print("# Phase 1: performing detection")
frame1 = Frame(
	get_rgb_triplet_dataset_path("../Dataset", "Testing", 2, first),
	get_depth_triplet_dataset_path("../Dataset", "Testing", 2, first),
	get_pose_triplet_dataset_path("../Dataset", "Testing", 2),
	first
)
frame2 = Frame(
	get_rgb_triplet_dataset_path("../Dataset", "Testing", 2, second),
	get_depth_triplet_dataset_path("../Dataset", "Testing", 2, second),
	get_pose_triplet_dataset_path("../Dataset", "Testing", 2),
	second
)
action = Action(frame1, frame2)
# Detection is embedded into the Merger class
pass

# PHASE 2: MATCHING
# Description: Using an extractor, from the detected features we compose matches
# between points in images.
print("# Phase 2: performing matching")
merger = Merger(num_features=5000,
				detector_method=DETECTION_METHOD,
				matcher_method="FLANN")
merge_image = merger.merge_action(action)

# PHASE 3: RANSAC
# Description: RANSAC over the matches to find fundamental matrix.
print("# Phase 3: executing ransac")
semantic_ransac = SemanticSampling()
best_f, best_mask = semantic_ransac.ransac_fundamental_matrix(action,
															  iterations=5000)

# PHASE 4: LOCALIZATION
# Description: Given the results of RANSAC, we perform localization.
print("# Phase 4: doing localization")
action.set_fundamental_matrix(best_f, best_mask)
action.compute_essential_matrix()
action.roto_translation()
print("F = %s" % best_f)
print("E = %s" % action.e_matrix)
print("R, t = %s, %s" % (action.R, action.t))
print("Q = %s" % action.from_rot_to_quat(normalize_em=False))

# PHASE 5: VISUALIZATION
# Description: Given the results of the pipeline, point clouds are visualized.
print("# Phase 5: printing visualization")
action.compute_inliers()
inliers_image = merger.merge_inliers(action)

visualizer = Visualizer(action=action)
visualizer.plot_action_point_cloud(registration_method="standard")

cv2.imshow("Inliers", inliers_image)
cv2.waitKey()