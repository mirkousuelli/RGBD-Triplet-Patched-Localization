"""
Project : RGB-D Semantic Sampling
Authors : Marco Petri and Mirko Usuelli
-----------------------------------------------
Degree : M.Sc. Computer Science and Engineering
Course : Image Analysis and Computer Vision
Professor : Vincenzo Caglioti
Advisors : Giacomo Boracchi, Luca Magri
University : Politecnico di Milano - A.Y. 2021/2022
"""
# Python imports
from typing import Tuple

# External imports
import matplotlib.axes
import matplotlib.pyplot as plt

# Project imports
import numpy as np
from PIL import Image

from camera.Frame import Frame
from camera.Action import Action
from camera.Recording import Recording
from ProjectObject import ProjectObject


class Visualizer(ProjectObject):
	"""A class implementing the point clouds visualizations."""
	ERROR_KEY = ProjectObject.ERROR_KEY + ["visualizer"]
	
	def __init__(self, frame: Frame = None,
				 action: Action = None,
				 recording: Recording = None):
		super().__init__()
		self.frame = frame
		self.action = action
		self.recording = recording

	def plot_image_and_depth(self, color_scale: str = 'b',
							 fig_size: Tuple = (8,4)) -> None:
		"""This method plots RGB image aside the depth image.
		
		Parameters
		----------
		color_scale : str ('b', 'g' or 'r')
			It is a character representing the color scale used to print the
			depth image.
		
		fig_size : Tuple
			It is the dimension of the figure on which the image is plotted.
		"""
		if color_scale not in ['b', 'g', 'r']:
			raise ValueError(self._raise_error("rgb_colors"))
		elif self.frame is None:
			raise ValueError(self._raise_error("missing_frame"))
		
		color, depth = self.frame.get_pil_images()
		dn = np.array(depth)
		dn = dn * 255 / np.max(dn)
		depth = Image.fromarray(np.uint8(dn))
		
		fig, ax = plt.subplots(1, 2, figsize=fig_size)
		ax[0].imshow(color)
		ax[0].set_title("RGB image")

		ax[1].imshow(depth)
		ax[1].set_title("Depth image")
		fig.show()

	def plot_frame_point_cloud(self) -> None:
		pass
	
	def plot_action_point_cloud(self) -> None:
		pass
	
	def plot_recording_point_cloud(self) -> None:
		pass
	
