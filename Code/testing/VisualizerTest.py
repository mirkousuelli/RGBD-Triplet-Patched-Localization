import numpy as np

from camera.Action import Action
from camera.Frame import Frame
from camera.Recording import Recording
from tools.Visualizer import Visualizer

def get_str(num) -> str:
	if num < 10:
		return "00" + str(num)
	elif 10 <= num < 100:
		return "0" + str(num)
	else:
		return str(num)

def get_list_of_actions(start, stop, step) -> list[Action]:
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

frame1 = Frame("../../Dataset/Colors/00000-color.png",
			  "../../Dataset/Depths/00000-depth.png",
			  0)
frame2 = Frame("../../Dataset/Colors/00400-color.png",
			  "../../Dataset/Depths/00400-depth.png",
			  399)

pose1 = frame1.extract_pose()
pose2 = frame2.extract_pose()

action = Action(frame1, frame2)

recording = Recording(get_list_of_actions(0, 400, 50))

vw = Visualizer(frame1, action, recording)

#vw.plot_image_and_depth()
#vw.plot_frame_point_cloud()
vw.plot_action_point_cloud(color1="g", color2="b")
#vw.plot_recording_point_cloud()
