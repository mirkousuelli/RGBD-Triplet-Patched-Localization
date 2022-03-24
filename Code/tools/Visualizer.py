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
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from PIL import Image

# Project imports
from camera.Frame import Frame
from camera.Action import Action
from camera.Recording import Recording
from ProjectObject import ProjectObject

visualization = o3d.visualization
PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic
RGBDImage = o3d.geometry.RGBDImage
PrimeSenseDefault = o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
PointCloud = o3d.geometry.PointCloud
	

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

	def _get_pc_color_scale(self, cloud: o3d.geometry.PointCloud,
							scale: str = "r") -> None:
		"""Perform color scaling on an open3d point cloud.

		:param cloud:
			The point cloud on which the colors must be scaled.
		:param scale:
			The color on which the colors must be scaled.
		:return:
			None.
		"""
		if scale not in ["r", "g", "b"]:
			raise ValueError("Only RGB supported. Funny huh?!")

		colors = np.asarray(cloud.colors)
		for i in range(colors.shape[0]):
			new_scale = 0.2126 * colors[i, 0] + 0.7152 * colors[i, 1] + 0.0722 * colors[i, 2]
			cloud.colors[i][0] = 0
			cloud.colors[i][1] = 0
			cloud.colors[i][2] = 0
			if scale == "r":
				cloud.colors[i][0] = new_scale
			elif scale == "g":
				cloud.colors[i][1] = new_scale
			else:
				cloud.colors[i][2] = new_scale

	def plot_image_and_depth(self, color_scale: str = 'b',
							 fig_size: Tuple = (8,4)) -> None:
		"""This method plots RGB image aside the depth image of the frame.
		
		:param str color_scale:
			{'r', 'g', 'b'} It is a character representing the color scale used
			to print the depth image.
		
		:param Tuple fig_size:
			It is the dimension of the figure on which the image is plotted.
			
		:return:
			None
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

	@staticmethod
	def _get_frame_point_cloud(frame: Frame) -> o3d.geometry.PointCloud:
		"""Return transformed point cloud of the frame.
		
		:param Frame frame:
			The frame from which we want the point cloud.
		:return:
			The point cloud rotated with the pose of the frame.
		:rtype:
			o3d.geometry.PointCloud
		"""
		# Get the rgbd image
		rgbd_image = frame.get_rgbd_image()
		
		# Build the point cloud
		pcd = PointCloud.create_from_rgbd_image(rgbd_image,
												PinholeCameraIntrinsic(PrimeSenseDefault))
		
		# Transform the image using the pose of the camera
		frame_pose = frame.extract_pose()
		quaternions = frame_pose[0:4]
		position = frame_pose[4:7] * np.array([0.3, 0.5, 0.5])
		rotation = PointCloud.get_rotation_matrix_from_quaternion(quaternions)
		
		result = pcd
		result = result.translate(position)
		#rotation = np.linalg.inv(rotation)
		result = result.rotate(rotation)
		return result

	def plot_frame_point_cloud(self) -> None:
		"""Plot the point cloud from the frame and the camera if it is passed.
		
		:param camera_pos:
			The position of the camera with respect to the frame.
		
		:return:
			None
		"""
		frame_point_cloud = self._get_frame_point_cloud(self.frame)
		
		# View the images
		vis = o3d.visualization.Visualizer()
		vis.create_window()
		vis.add_geometry(frame_point_cloud)
		o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.25)
		vis.run()
	
	def plot_action_point_cloud(self, color1: str = None,
								color2: str = None,
								camera_pos = None) -> None:
		"""Plot the point cloud from the action and the camera if it is passed
		
		:param camera_pos:
			The position of the camera with respect to the frame.
		
		:return:
			None
		"""
		frame_1_point_cloud = self._get_frame_point_cloud(self.action.first)
		if color1 is not None:
			self._get_pc_color_scale(frame_1_point_cloud, color1)

		frame_2_point_cloud = self._get_frame_point_cloud(self.action.second)
		if color2 is not None:
			self._get_pc_color_scale(frame_2_point_cloud, color2)
		
		# View the images
		vis = o3d.visualization.Visualizer()
		vis.create_window()
		new_cloud = frame_1_point_cloud + frame_2_point_cloud
		vis.add_geometry(new_cloud)
		#vis.add_geometry(frame_2_point_cloud)
		o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.25)
		vis.run()
	
	def plot_recording_point_cloud(self) -> None:
		"""Plots the point cloud from the recording.
		
		:return:
			None
		"""
		# View the images
		vis = o3d.visualization.Visualizer()
		vis.create_window()
		
		frames = self.recording.get_all_frames()
		for frame in frames:
			geometry = self._get_frame_point_cloud(frame)
			vis.add_geometry(geometry)
		
		o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.25)
		vis.run()
