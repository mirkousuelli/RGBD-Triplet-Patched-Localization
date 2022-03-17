from camera.Frame import Frame
from tools.Visualizer import Visualizer

frame = Frame("../../Dataset/Colors/00000-color.png",
			  "../../Dataset/Depths/00000-depth.png")
vw = Visualizer(frame)
vw.plot_image_and_depth()
