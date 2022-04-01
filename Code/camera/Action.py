# Python imports

# External imports

# In-project imports
from camera.Frame import Frame
from ProjectObject import ProjectObject


# TODO: wow, implement me please
class Action(ProjectObject):
	ERROR_KEY = ProjectObject.ERROR_KEY + ["action"]
	
	def __init__(self,
	             first: Frame,
	             second: Frame):
		super().__init__()

		# Frames
		self.first = first
		self.second = second

		# Matches
		self.matches = None
		self.links = None
		self.links_inliers = []

		# Fundamental Matrix
		self.f_matrix = None

		# Inliers mask
		self.f_mask = None

		# Essential Matrix
		self.e_matrix = None

		# Roto-Translation
		self.R = None
		self.t = None

	def normalize_essential_matrix(self):
		"""Normalizes the essential matrix"""
		# pre-conditions
		assert self.e_matrix is not None
		
		self.e_matrix = self.e_matrix / self.e_matrix[2, 2]
