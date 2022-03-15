# Python imports

# External imports

# In-project imports
from ProjectObject import ProjectObject

# TODO: wow, implement me please
class Recording(ProjectObject):
	ERROR_KEY = ProjectObject.ERROR_KEY + ["recording"]
	
	def __init__(self):
		super().__init__()
