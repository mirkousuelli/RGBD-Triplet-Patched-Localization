"""
Project : RGB-D Semantic Sampling
Authors : Marco Petri and Mirko Usuelli
--------------------------------------------------------------------------------
Degree : M.Sc. Computer Science and Engineering
Course : Image Analysis and Computer Vision
Professor : Vincenzo Caglioti
Advisors : Giacomo Boracchi, Luca Magri
University : Politecnico di Milano - A.Y. 2021/2022
"""
from camera.Frame import Frame
from tools.Detector import Detector
from tools.Matcher import Matcher


class Merger:
	""" Class implementing the tool 'Merger' which encapsulates both the
	Detector and the Matcher in purpose of simplifying the overall sampling
	procedure.
	"""

	def __init__(self,
	             num_features=50,
	             detector_method="ORB",
	             matcher_method="FLANN"):
		""" Constructor.

		:param num_features:
			The number of features to be detected and matched afterwards.
		:type num_features: int

		:param detector_method:
			The string name of the chosen detecting-method.
		:type detector_method: str

		:param matcher_method:
			The string name of the chosen matching-method.
		:type matcher_method: str
		"""
		# Detector initialization
		self.detector = Detector(num_features, detector_method)

		# the algorithm changes based on the technique adopted
		algorithm = 0 if detector_method == "SIFT" else 6

		# Matcher initialization
		self.matcher = Matcher(num_features,
		                       matcher_method,
		                       search_algorithm=algorithm,
		                       filter_test=0.7)

	def merge(self,
	          img_1: Frame,
	          img_2: Frame,
	          limit=-1):
		""" Merge two images by first detecting their features and then matching
		their descriptors all at once.

		:param img_1:
			First image.
		:type img_1: Frame

		:param img_2:
			Second image.
		:type img_2: Frame

		:param limit:
			Integer number which limits how many matching links to be drawn.
		:type limit: int

		:return:
			The two images merged into one image with matching links drawn.
		:rtype: image
		"""
		self.detector.detect_and_compute(img_1)
		self.detector.detect_and_compute(img_2)

		matches = self.matcher.match(img_1, img_2)
		return self.matcher.draw_matches(img_1, img_2, matches, limit=limit)
