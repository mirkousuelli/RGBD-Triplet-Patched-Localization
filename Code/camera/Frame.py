# Python imports
import linecache
import os
from typing import Tuple, Union

# External imports
import cv2
import numpy as np
import open3d as o3d
from PIL import Image

# In-project imports
from ProjectObject import ProjectObject


class Frame(ProjectObject):
	ERROR_KEY = ProjectObject.ERROR_KEY + ["frame"]

	def __init__(self, color_path: str,
				 depth_path: str,
				 img_index: int):
		super().__init__()
		self.__color_path = color_path
		self.__depth_path = depth_path
		self.__index = img_index
		self.key_points = None
		self.descriptors = None
		self.epi_lines = None
		self.points = []
		self.inliers = []

		# Kinect v1 intrinsic parameters
		self.fx = 522.259
		self.fy = 523.419
		self.Cx = 330.18
		self.Cy = 254.437

	def extract_pose(self) -> np.ndarray:
		"""Get the pose of the image from the paths.
		
		:return:
			The array specifying the pose of the current image with respect to
			the first image.
		:rtype: np.ndarray
		"""
		camera_dir = os.path.dirname(__file__)
		file_path = os.path.join(camera_dir, '../../Dataset/Poses/02.pose')
		pose = linecache.getline(file_path, self.__index + 1)
		pose = pose.split(" ")[:-1]
		pose = np.array(pose, dtype=float)
		return pose

	def get_pil_images(self, ret : str = None) -> Union[Image.Image,
	                                                    Tuple[Image.Image,
	                                                          Image.Image]]:
		"""Return the PIL images of color and depth.
		:param ret:
			Specified whether to return one or both the images.

		:return:
			The color image specified for this frame.
		:rtype: PIL.image
			The image with the depth information.
		:rtype: PIL.image
		"""
		if ret is None:
			return Image.open(self.__color_path), Image.open(self.__depth_path)
		elif ret == "rgb":
			return Image.open(self.__color_path)
		elif ret == "depth":
			return Image.open(self.__depth_path)

	def get_o3d_images(self, ret : str = None) -> Union[o3d.geometry.Image,
	                                                    Tuple[o3d.geometry.Image,
	                                                          o3d.geometry.Image]]:
		"""Return the open3d images of color and depth.
		:param ret:
			Specified whether to return one or both the images.

		:return:
			The color image specified for this frame.
		:rtype: open3d.image
			The image with the depth information.
		:rtype: open3d.image
		"""
		if ret is None:
			return o3d.io.read_image(self.__color_path), o3d.io.read_image(self.__depth_path)
		elif ret == "rgb":
			return o3d.io.read_image(self.__color_path)
		elif ret == "depth":
			return o3d.io.read_image(self.__depth_path)

	def get_cv2_images(self, ret: str = None):
		"""Return the cv2 images of color and depth.
		:param ret:
			Specified whether to return one or both the images.

		:return:
			The color image specified for this frame.
		:rtype: cv2.image
			The image with the depth information.
		:rtype: cv2.image
		"""
		if ret is None:
			return cv2.imread(self.__color_path), cv2.imread(self.__depth_path)
		elif ret == "rgb":
			return cv2.imread(self.__color_path)
		elif ret == "depth":
			return cv2.imread(self.__depth_path)
	
	def get_rgbd_image(self) -> o3d.geometry.RGBDImage:
		# TODO: implement automatic way to extract rgbd from a couple of any type of images
		color, depth = self.get_o3d_images()
		return o3d.geometry.RGBDImage.create_from_tum_format(color, depth)

	def get_size(self):
		with Image.open(self.__color_path) as img:
			return img.size

	def calibration_matrix(self):
		return np.mat([[self.fx, 0, self.Cx],
		               [0, self.fy, self.Cy],
		               [0, 0, 1]])
