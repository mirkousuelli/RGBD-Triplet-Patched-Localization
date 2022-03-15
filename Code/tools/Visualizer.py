"""
Project : RGB-D Semantic Sampling
Authors : Marco Petri and Mirko Usuelli
-----------------------------------------------
Degree : M.Sc. Computer Science and Engineering
Course : Image Analysis and Computer Vision
Professor : Vincenzo Caglioti
Advisors : Giacomo Boracchi, Luca Magri
University : Politecnico di Milano - A.Y. 2021/2022
"""
# Python imports

# External imports

# Project imports
from ProjectObject import ProjectObject


class Visualizer(ProjectObject):
	"""A class implementing the point clouds visualizations."""
	ERROR_KEY = ProjectObject.ERROR_KEY + ["visualizer"]
	
	def __init__(self):
		super().__init__()
