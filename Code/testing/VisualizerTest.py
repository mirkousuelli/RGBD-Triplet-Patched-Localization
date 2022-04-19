import numpy as np

from camera.Action import Action
from camera.Frame import Frame
from camera.Recording import Recording
from tools.Visualizer import Visualizer
from utils.utils import get_str


def get_list_of_actions(start, stop, step):
	actions = []
	previous_frame = None
	for i in range(start, stop, step):
		if previous_frame is None:
			previous_frame = Frame("../../Dataset/Testing/2/Colors/00" + get_str(i) + "-color.png",
								   "../../Dataset/Testing/2/Depths/00" + get_str(i) + "-depth.png",
								   i)
		else:
			temp = Frame("../../Dataset/Testing/2/Colors/00" + get_str(i) + "-color.png",
						 "../../Dataset/Testing/2/Depths/00" + get_str(i) + "-depth.png",
						 i)
			actions.append(Action(previous_frame, temp))
			previous_frame = temp
	return actions

def view_actions_one_at_a_time(start, stop, step) -> None:
	for i in range(start, stop, step):
		frame1_ = Frame("../../Dataset/Testing/2/Colors/00" + get_str(i) + "-color.png",
					   "../../Dataset/Testing/2/Depths/00" + get_str(i) + "-depth.png",
					   i)
		frame2_ = Frame("../../Dataset/Testing/2/Colors/00" + get_str(i+step) + "-color.png",
					   "../../Dataset/Testing/2/Depths/00" + get_str(i+step) + "-depth.png",
					   i+step)

		action_ = Action(frame1_, frame2_)
		viewer = Visualizer(action=action_)
		viewer.plot_action_point_cloud(registration_method="pose",
									   verbose=False)
		
		print("Action %s and action %s" % (i, i+step))

first = 0
second = 50
frame1 = Frame("../../Dataset/Testing/2/Colors/00" + get_str(first) + "-color.png",
			  "../../Dataset/Testing/2/Depths/00" + get_str(first) + "-depth.png",
			  first)
frame2 = Frame("../../Dataset/Testing/2/Colors/00" + get_str(second) + "-color.png",
			  "../../Dataset/Testing/2/Depths/00" + get_str(second) + "-depth.png",
			  second)

action = Action(frame1, frame2)

recording = Recording(get_list_of_actions(0, 200, 1))

vw = Visualizer(frame1, action, recording)

#view_actions_one_at_a_time(0, 500, 10)

#vw.plot_image_and_depth()
#vw.plot_frame_point_cloud()
vw.plot_action_point_cloud(original_color=True, registration_method="pose", color1=np.array([255, 0, 204]), color2=np.array([255, 166, 0]))
#vw.plot_recording_point_cloud()
