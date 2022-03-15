# Python imports

# External imports

# In-project imports
from ProjectObject import ProjectObject


# TODO: wow, implement me please
class Frame(ProjectObject):
	ERROR_KEY = ProjectObject.ERROR_KEY + ["frame"]

	def __init__(self):
		super().__init__()
		self.key_points = None
		self.descriptors = None

	def set_features(self, _key_points, _descriptors):
		self.key_points = _key_points
		self.descriptors = _descriptors
