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
			previous_frame = Frame("../../Dataset/Colors/00" + get_str(i) + "-color.png",
								   "../../Dataset/Depths/00" + get_str(i) + "-depth.png",
								   i)
		else:
			temp = Frame("../../Dataset/Colors/00" + get_str(i) + "-color.png",
						 "../../Dataset/Depths/00" + get_str(i) + "-depth.png",
						 i)
			actions.append(Action(previous_frame, temp))
			previous_frame = temp
	return actions

def view_actions_one_at_a_time(start, stop, step) -> None:
	for i in range(start, stop, step):
		frame1_ = Frame("../../Dataset/Colors/00" + get_str(i) + "-color.png",
					   "../../Dataset/Depths/00" + get_str(i) + "-depth.png",
					   i)
		frame2_ = Frame("../../Dataset/Colors/00" + get_str(i+step) + "-color.png",
					   "../../Dataset/Depths/00" + get_str(i+step) + "-depth.png",
					   i+step)

		action_ = Action(frame1_, frame2_)
		viewer = Visualizer(action=action_)
		viewer.plot_action_point_cloud(verbose=False)
		
		print("Action %s and action %s" % (i, i+step))

first = 40
second = 60
frame1 = Frame("../../Dataset/Colors/00" + get_str(first) + "-color.png",
			  "../../Dataset/Depths/00" + get_str(first) + "-depth.png",
			  first)
frame2 = Frame("../../Dataset/Colors/00" + get_str(second) + "-color.png",
			  "../../Dataset/Depths/00" + get_str(second) + "-depth.png",
			  second)

action = Action(frame1, frame2)

recording = Recording(get_list_of_actions(0, 200, 1))

vw = Visualizer(frame1, action, recording)

#view_actions_one_at_a_time(100, 500, 5)

#vw.plot_image_and_depth()
#vw.plot_frame_point_cloud()
#vw.plot_action_point_cloud(color1=np.array([255, 0, 204]), color2=np.array([255, 166, 0]))
#vw.plot_recording_point_cloud()
