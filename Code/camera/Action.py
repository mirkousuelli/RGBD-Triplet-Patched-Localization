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
