import numpy as np

from camera.Action import Action
from camera.Frame import Frame
from tools.Merger import Merger
from utils.utils import get_str

num_samples = 5
first = 0
second = 60
frame1 = Frame("../../Dataset/Testing/2/Colors/00" + get_str(first) + "-color.png",
			  "../../Dataset/Testing/2/Depths/00" + get_str(first) + "-depth.png",
			  first)
frame2 = Frame("../../Dataset/Testing/2/Colors/00" + get_str(second) + "-color.png",
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
print(matches)
num_samples = num_samples if num_samples < len(matches) else len(matches)
selected_features = rng.choice(len(matches), num_samples, replace=False)
print(selected_features)
features = [action.first.key_points[matches[x].queryIdx] for x in selected_features]
print(features)
