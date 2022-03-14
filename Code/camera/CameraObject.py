# Python imports
import os

# External imports
import json

# In-project imports

# TODO: wow, implement me please
class CameraObject(object):
	ERROR_KEY = "NONE"
	
	def __init__(self):
		super().__init__()
		camera_dir = os.path.dirname(__file__)
		file_path = os.path.join(camera_dir, 'errors.json')
		file = open(file_path)
		self.errors = json.load(file)
		file.close()
		
	def _raise_error(self, error_name: str) -> str:
		"""Prints the error message given the error name key.
		
		Parameters
		----------
		error_name : str
			The key for the error type of that object.
		"""
		try:
			return self.errors[self.ERROR_KEY][error_name]
		except KeyError:
			return self.errors["parent"][error_name]