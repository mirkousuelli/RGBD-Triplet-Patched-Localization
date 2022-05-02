import copy

import numpy as np

from camera.Action import Action
from camera.Frame import Frame
from camera.Recording import Recording
from tools.Merger import Merger
from tools.Visualizer import Visualizer
from utils.utils import get_str, get_rgb_triplet_dataset_path, \
	get_depth_triplet_dataset_path, get_pose_triplet_dataset_path


def get_list_of_actions(start, stop, step):
	actions = []
	previous_frame = None
	for i in range(start, stop, step):
		if previous_frame is None:
			previous_frame = Frame(get_rgb_triplet_dataset_path("../../Dataset", "Testing", 2, i),
								   get_depth_triplet_dataset_path("../../Dataset", "Testing", 2, i),
								   get_pose_triplet_dataset_path("../../Dataset", "Testing", 2),
								   i)
		else:
			temp = Frame(get_rgb_triplet_dataset_path("../../Dataset", "Testing", 2, i),
						 get_depth_triplet_dataset_path("../../Dataset", "Testing", 2, i),
						 get_pose_triplet_dataset_path("../../Dataset", "Testing", 2),
						 i)
			actions.append(Action(previous_frame, temp))
			previous_frame = temp
	return actions

def view_actions_one_at_a_time(start, stop, step) -> None:
	for i in range(start, stop, step):
		frame1_ = Frame(get_rgb_triplet_dataset_path("../../Dataset", "Testing", 2, i),
						get_depth_triplet_dataset_path("../../Dataset", "Testing", 2, i),
						get_pose_triplet_dataset_path("../../Dataset", "Testing", 2),
						i)
		frame2_ = Frame(get_rgb_triplet_dataset_path("../../Dataset", "Testing", 2, i+step),
						get_depth_triplet_dataset_path("../../Dataset", "Testing", 2, i+step),
						get_pose_triplet_dataset_path("../../Dataset", "Testing", 2),
						i+step)

		action_ = Action(frame1_, frame2_)
		viewer = Visualizer(action=action_)
		viewer.plot_action_point_cloud(registration_method="pose",
									   verbose=False)
		
		print("Action %s and action %s" % (i, i+step))

first = 0
second = 20
third = 50
frame1 = Frame(get_rgb_triplet_dataset_path("../../Dataset", "Testing", 2, first),
			   get_depth_triplet_dataset_path("../../Dataset", "Testing", 2, first),
			   get_pose_triplet_dataset_path("../../Dataset", "Testing", 2),
			   first)
frame2 = Frame(get_rgb_triplet_dataset_path("../../Dataset", "Testing", 2, second),
			   get_depth_triplet_dataset_path("../../Dataset", "Testing", 2, second),
			   get_pose_triplet_dataset_path("../../Dataset", "Testing", 2),
			   second)
frame3 = Frame(get_rgb_triplet_dataset_path("../../Dataset", "Testing", 2, third),
			   get_depth_triplet_dataset_path("../../Dataset", "Testing", 2, third),
			   get_pose_triplet_dataset_path("../../Dataset", "Testing", 2),
			   third)

action = Action(frame1, frame2)
action2 = Action(copy.deepcopy(frame1), frame3)

recording = Recording(get_list_of_actions(0, 200, 1))

vw = Visualizer(frame1, action, recording)
vw2 = Visualizer(frame1, action2, recording)

merger = Merger(num_features=5000,
                detector_method="ORB",
                matcher_method="FLANN")
merge_image = merger.merge_action(action)
action.compute_fundamental_matrix()
action.compute_essential_matrix()

merger2 = Merger(num_features=5000,
                detector_method="ORB",
                matcher_method="FLANN")
merge_image2 = merger2.merge_action(action2)
action2.compute_fundamental_matrix()
action2.compute_essential_matrix()

#view_actions_one_at_a_time(0, 500, 10)

#vw.plot_image_and_depth()
#vw.plot_frame_point_cloud()
vw.plot_action_point_cloud(original_color=True, registration_method="pose", color1=np.array([255, 0, 204]), color2=np.array([255, 166, 0]))
vw2.plot_action_point_cloud(original_color=True, registration_method="pose", color1=np.array([255, 0, 204]), color2=np.array([255, 166, 0]))
#vw.plot_recording_point_cloud()
