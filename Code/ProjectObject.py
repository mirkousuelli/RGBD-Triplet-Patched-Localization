# Python imports
import os

# External imports
import json


class ProjectObject(object):
	ERROR_KEY = ["object"]
	
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
		for i in range(len(self.ERROR_KEY)):
			try:
				curr = - (i + 1)
				return self.errors[self.ERROR_KEY[curr]][error_name]
			except KeyError:
				pass
		
		return self.errors["object"]["generic"]