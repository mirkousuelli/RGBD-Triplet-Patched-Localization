from camera.Frame import Frame
from tools.Visualizer import Visualizer

frame = Frame("../../Dataset/Colors/00000-color.png",
			  "../../Dataset/Depths/00000-depth.png",
			  1)
pose = frame.extract_pose()

vw = Visualizer(frame)
vw.plot_image_and_depth()

vw.plot_frame_point_cloud()
