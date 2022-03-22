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
		self.first = first
		self.second = second
		self.matches = None
		self.links = None
		self.f_matrix = None
