# Python imports

# External imports

# In-project imports
from ProjectObject import ProjectObject
from camera.Action import Action
from camera.Frame import Frame


class Recording(ProjectObject):
	ERROR_KEY = ProjectObject.ERROR_KEY + ["recording"]
	
	def __init__(self,
				 actions: list[Action]):
		super().__init__()
		
		for i in range(len(actions)-1):
			actions[i].second = actions[i+1].first
		
		self.actions = actions.copy()
		
	def get_all_frames(self) -> list[Frame]:
		"""Get all the images in the recording.
		
		:return:
			The list of all images stored in the recording.
		:rtype: Frame
		"""
		frames = []
		
		for i in range(len(self.actions)):
			frames.append(self.actions[i].first)
			
			if i == len(self.actions) -1:
				frames.append(self.actions[i].second)
				
		return frames
