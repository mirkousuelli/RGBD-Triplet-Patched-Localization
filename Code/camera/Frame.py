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
from cv2 import KeyPoint
from open3d.cpu.pybind.camera import PinholeCameraIntrinsic, PrimeSenseDefault
from open3d.cpu.pybind.geometry import PointCloud

from ProjectObject import ProjectObject


class Frame(ProjectObject):
	ERROR_KEY = ProjectObject.ERROR_KEY + ["frame"]

	def __init__(
		self,
	    color_path: str,
		depth_path: str,
		pose_path: str,
		img_index: int
	):
		super().__init__()
		self.color_path = color_path
		self.depth_path = depth_path
		self.pose_path = pose_path
		self.index = img_index
		self.key_points = None  #: list[KeyPoint] = None
		self.descriptors = None  # : list = None
		self.epi_lines = None
		self.points = []
		self.inliers = []
		self.key_points_inliers = []
		self.descriptors_inliers = np.array([], dtype=np.uint8)
		self.pose = self.extract_pose()
		self.R, self.t = self.from_pose_to_rototrasl()
		self.f_matrix = None

		# Kinect v1 intrinsic parameters
		self.fx = 522.259  # 514.120
		self.fy = 523.419  # 513.841
		self.Cx = 330.18  # 310.744
		self.Cy = 254.437  # 262.611

	def extract_pose(self) -> np.ndarray:
		"""Get the pose of the image from the paths.
		
		:return:
			The array specifying the pose of the current image with respect to
			the first image.
		:rtype: np.ndarray
		"""
		camera_dir = os.path.dirname(__file__)

		file_path = os.path.join(camera_dir, self.pose_path)
		self.pose = linecache.getline(file_path, self.index + 1)
		self.pose = self.pose.split(" ")[:-1]
		self.pose = np.array(self.pose, dtype=float)
		return self.pose

	def from_pose_to_rototrasl(
		self
	):
		q = np.ndarray([])
		q = np.append(q, -self.pose[1])
		q = np.append(q, -self.pose[2])
		q = np.append(q, -self.pose[3])
		q = np.append(q, -self.pose[0])
		self.t = self.pose[4:]

		s = sum(i ** 2 for i in q) ** (-2)

		# First row of the rotation matrix
		r00 = 1 - 2 * s * (q[2] ** 2 + q[3] ** 2)
		r01 = 2 * s * (q[1] * q[2] - q[0] * q[3])
		r02 = 2 * s * (q[1] * q[3] + q[0] * q[2])

		# Second row of the rotation matrix
		r10 = 2 * s * (q[1] * q[2] + q[0] * q[3])
		r11 = 1 - 2 * s * (q[1] ** 2 + q[3] ** 2)
		r12 = 2 * s * (q[2] * q[3] - q[0] * q[1])

		# Third row of the rotation matrix
		r20 = 2 * s * (q[1] * q[3] - q[0] * q[2])
		r21 = 2 * s * (q[2] * q[3] + q[0] * q[1])
		r22 = 1 - 2 * s * (q[1] ** 2 + q[2] ** 2)

		self.R = np.mat(
			[[r00, r01, r02],
			 [r10, r11, r12],
			 [r20, r21, r22]], dtype=float
		)
		self.R /= self.R[2, 2]

		# 3x3 rotation matrix
		return self.R, self.t

	def from_rototrasl_to_f_matrix(
		self
	):
		K = self.calibration_matrix()
		A = K @ self.R.T @ self.t
		C = np.mat(
			[[0.0, -A[0, 2], +A[0, 1]],
			 [+A[0, 2], 0.0, -A[0, 0]],
			 [-A[0, 1], +A[0, 0], 0.0]], dtype=float
		)
		self.f_matrix = np.linalg.inv(K).T @ self.R @ K.T @ C
		self.f_matrix /= self.f_matrix[2, 2]
		return self.f_matrix

	def get_pil_images(self, ret: str = None) -> Union[Image.Image,
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
			return Image.open(self.color_path), Image.open(self.depth_path)
		elif ret == "rgb":
			return Image.open(self.color_path)
		elif ret == "depth":
			return Image.open(self.depth_path)

	def get_o3d_images(self, ret: str = None) -> Union[o3d.geometry.Image,
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
			return o3d.io.read_image(self.color_path), o3d.io.read_image(self.depth_path)
		elif ret == "rgb":
			return o3d.io.read_image(self.color_path)
		elif ret == "depth":
			return o3d.io.read_image(self.depth_path)

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
			return cv2.imread(self.color_path), cv2.imread(self.depth_path)
		elif ret == "rgb":
			return cv2.imread(self.color_path)
		elif ret == "depth":
			return cv2.imread(self.depth_path)
	
	def get_rgbd_image(self) -> o3d.geometry.RGBDImage:
		# TODO: implement automatic way to extract rgbd from a couple of any type of images
		color, depth = self.get_o3d_images()
		rgbd = o3d.geometry.RGBDImage.create_from_tum_format(color,
															 depth)
		color = o3d.geometry.RGBDImage.create_from_color_and_depth(color,
																   depth,
																   convert_rgb_to_intensity=False)
		rgbd.color = color.color
		return rgbd

	def get_point_cloud(self) -> PointCloud:
		"""Gets the point cloud of the frame.
		
		:return: The point cloud of the frame.
		"""
		pcd = PointCloud.create_from_rgbd_image(self.get_rgbd_image(),
												PinholeCameraIntrinsic(PrimeSenseDefault))
		return pcd

	def get_size(self):
		with Image.open(self.color_path) as img:
			return img.size

	def calibration_matrix(self):
		return np.mat([[self.fx, 0, self.Cx],
		               [0, self.fy, self.Cy],
		               [0, 0, 1]])
