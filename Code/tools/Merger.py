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
import cv2
from camera.Frame import Frame


class Merger:
	"""
	Class implementing the tool 'Merger' which encapsulates both the Detector and the Matcher
	in purpose of simplifying the overall sampling procedure
	"""

	def __init__(self, _detector, _matcher):
		"""
		Constructor.
		I : detector object, matcher object
		O : -
		"""
		self.detector = _detector
		self.matcher = _matcher

	def merge(self, _img_1, _img_2, _limit=-1):
		"""
		Merge two images by first detecting their features and then by matching them in one shot
		I : images to be merged
		O : final image collage with linked matches
		"""
		key_points_1, descriptors_1 = self.detector.detectAndCompute(_img_1)
		key_points_2, descriptors_2 = self.detector.detectAndCompute(_img_2)

		matches = self.matcher.match(descriptors_1, descriptors_2)
		return self.matcher.drawMatches(_img_1, key_points_1, _img_2, key_points_2, matches, limit=_limit)
