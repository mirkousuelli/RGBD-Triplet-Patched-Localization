# Python imports

# External imports

# In-project imports
from camera.Frame import Frame
from camera.CameraObject import CameraObject

# TODO: wow, implement me please
class Action(CameraObject):
	
	def __init__(self, first: Frame,
				 second: Frame):
		super().__init__()
		self.first = first
		self.second = second
