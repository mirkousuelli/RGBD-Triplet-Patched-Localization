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
from tools.Detector import Detector
from tools.Matcher import Matcher


class Merger:
	""" Class.
	Class implementing the tool 'Merger' which encapsulates both the
	Detector and the Matcher in purpose of simplifying the overall sampling
	procedure.
	"""

	def __init__(self,
	             num_features=50,
	             detector_method="ORB",
	             matcher_method="FLANN"):
		""" Constructor.

		Parameters
		----------
		num_features : int
			The number of features to be detected and matched afterwards.

		detector_method : str
			The string name of the chosen detecting-method.

		matcher_method : str
			The string name of the chosen matching-method.
		"""
		self.detector = Detector(num_features, detector_method)
		self.matcher = Matcher(num_features, matcher_method)

	def merge(self,
	          img_1,  # : Frame
	          img_2,  # : Frame
	          limit=-1):
		""" Method.
		Merge two images by first detecting their features and then matching
		their descriptors all at once.

		Parameters
		----------
		img_1 : Frame
			First image.

		img_2 : Frame
			Second image.

		limit : int
			Integer number which limits how many matching links to be drawn.
		"""
		key_points_1, descriptors_1 = self.detector.detect_and_compute(img_1)
		key_points_2, descriptors_2 = self.detector.detect_and_compute(img_2)

		matches = self.matcher.match(descriptors_1, descriptors_2)
		return self.matcher.draw_matches(img_1, key_points_1,
		                                 img_2, key_points_2,
		                                 matches, limit=limit)
