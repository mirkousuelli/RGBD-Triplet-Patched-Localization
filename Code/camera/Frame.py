# Python imports
from typing import Tuple

# External imports
import open3d as o3d
from PIL import Image

# In-project imports
from ProjectObject import ProjectObject


class Frame(ProjectObject):
	ERROR_KEY = ProjectObject.ERROR_KEY + ["frame"]

	def __init__(self, color_path: str,
				 depth_path: str):
		super().__init__()
		self.__color_path = color_path
		self.__depth_path = depth_path
		self.key_points = None
		self.descriptors = None
		
	def get_pil_images(self) -> Tuple[Image.Image, Image.Image]:
		"""Return the PIL images of color and depth.
		
		Returns
		-------
		color_img : PIL.Image
			The color image specified for this frame.
		depth_img : PIL.Image
			The image with the depth information.
		"""
		return Image.open(self.__color_path), Image.open(self.__depth_path)
	
	def get_o3d_images(self) -> Tuple[o3d.geometry.Image, o3d.geometry.Image]:
		"""Return the PIL images of color and depth.

		Returns
		-------
		color_img : PIL.Image
			The color image specified for this frame.
		depth_img : PIL.Image
			The image with the depth information.
		"""
		return o3d.io.read_image(self.__color_path), o3d.io.read_image(self.__depth_path)
	
	def get_rgbd_image(self) -> o3d.geometry.RGBDImage:
		# TODO: implement automatic way to extract rgbd from a couple of any type of images
		pass

	def set_features(self, _key_points,
					 _descriptors):
		# TODO: add documentation
		self.key_points = _key_points
		self.descriptors = _descriptors
