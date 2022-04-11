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
from utils.transformation_utils import get_4x4_transform_from_translation

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
							scale: np.ndarray = None) -> None:
		"""Perform color scaling on an open3d point cloud.

		:param cloud:
			The point cloud on which the colors must be scaled.
		
		:param scale:
			The color on which the colors must be scaled.
		
		:return:
			None.
		"""
		if scale is None:
			raise ValueError("You must pass a color")
		elif scale.shape[0] != 3 or scale.ndim > 1:
			raise ValueError("You must provide a (3,) ndarray.")

		red_percentage = 0.0
		green_percentage = 0.0
		blue_percentage = 0.0
		if scale[0] >= scale[1] and scale[0] >= scale[2]:
			red_percentage = 1.0
			green_percentage = scale[1] / scale[0]
			blue_percentage = scale[2] / scale[0]
		elif scale[1] >= scale[0] and scale[1] >= scale[2]:
			red_percentage = scale[0] / scale[1]
			green_percentage = 1.0
			blue_percentage = scale[2] / scale[1]
		elif scale[2] >= scale[0] and scale[2] >= scale[1]:
			red_percentage = scale[0] / scale[2]
			green_percentage = scale[1] / scale[2]
			blue_percentage = 1.0

		colors = np.asarray(cloud.colors)
		for i in range(colors.shape[0]):
			new_scale = 0.2126 * colors[i, 0] + 0.7152 * colors[i, 1] + 0.0722 * colors[i, 2]
			cloud.colors[i][0] = new_scale * red_percentage
			cloud.colors[i][1] = new_scale * green_percentage
			cloud.colors[i][2] = new_scale * blue_percentage

	def plot_image_and_depth(self, fig_size: Tuple = (8,4)) -> None:
		"""This method plots RGB image aside the depth image of the frame.
		
		:param Tuple fig_size:
			It is the dimension of the figure on which the image is plotted.
			
		:return:
			None
		"""
		if self.frame is None:
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
		# Transform the image using the pose of the camera
		frame_pose = frame.extract_pose()
		quaternions = frame_pose[0:4]
		position = frame_pose[4:7] * -1
		rotation = PointCloud.get_rotation_matrix_from_quaternion(quaternions)
		
		result = frame.get_point_cloud()
		result = result.translate(position)
		rotation = np.matrix.transpose(rotation)
		rotation = np.linalg.inv(rotation)
		result = result.rotate(rotation)
		return result

	def plot_frame_point_cloud(self) -> None:
		"""Plot the point cloud from the frame and the camera if it is passed.
			
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
	
	def plot_action_point_cloud(self, original_color: bool = True,
								color1: np.ndarray = None,
								color2: np.ndarray = None,
								registration_method: str = "icp",
								verbose: bool = True) -> None:
		"""Plot the point cloud from the action and the camera if it is passed
		
		:param original_color:
			States whether the point cloud must be viewed with the original
			colors or not.
			
		:param color1:
			The colorscale to use to print the first frame.
			
		:param color2:
			The colorscale to use to print the second frame.
			
		:param registration_method:
			The registration method to be used to align the point clouds.
			
		:param verbose:
			States if detailed printing must be shown.
		
		:return:
			None
		"""
		if registration_method not in ["icp"]:
			raise ValueError("Registration method not present")
		
		transformation_from_2_to_1 = self.action.roto_translation_with_icp(0.02,
																		   verbose=verbose)
		frame_1_point_cloud = self.action.first.get_point_cloud()
		frame_2_point_cloud = self.action.second.get_point_cloud()
		frame_2_point_cloud.transform(transformation_from_2_to_1)
		
		if not original_color:
			if color1 is not None:
				self._get_pc_color_scale(frame_1_point_cloud, color1)
			if color2 is not None:
				self._get_pc_color_scale(frame_2_point_cloud, color2)
		
		# View the images
		o3d.visualization.draw_geometries([frame_1_point_cloud,
										   frame_2_point_cloud])
	
	def plot_recording_point_cloud(self, original_color: bool = True,
								   registration_method: str = "icp") -> None:
		"""Plots the point cloud from the recording.
		
		:param original_color:
			States whether the point cloud must be viewed with the original
			colors or not.
			
		:param registration_method:
			The registration method to be used to align the point clouds.
		
		:return:
			None
		"""
		if registration_method not in ["icp"]:
			raise ValueError("Registration method not present")
		
		point_clouds = [self.recording.actions[0].first.get_point_cloud()]
		total_transformation = get_4x4_transform_from_translation(np.array([0,
																			0,
																			0]))
		for action in self.recording.actions:
			transformation_from_2_to_1 = action.roto_translation_with_icp(0.02)
			frame_2_point_cloud = action.second.get_point_cloud()
			total_transformation = np.matmul(total_transformation,
											 transformation_from_2_to_1)
			frame_2_point_cloud.transform(total_transformation)
			point_clouds.append(frame_2_point_cloud)

		# View the images
		o3d.visualization.draw_geometries(point_clouds)
