from Code.camera.Frame import Frame
from Code.utils.utils import *

index = 60
frame1 = Frame(
	get_rgb_triplet_dataset_path("../../Dataset", "Testing", 2, index),
	get_depth_triplet_dataset_path("../../Dataset", "Testing", 2, index),
	get_pose_triplet_dataset_path("../../Dataset", "Testing", 2),
	index
)

print(frame1)
