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
import cv2


class Detector:
    """ Class implementing the tool 'Detector' able to detect relevant
    corners/features in a single image.
    """

    # Techniques available:
    ORB = "ORB"  # Matcher.search_algorithm = 6 : LSH
    SIFT = "SIFT"  # Matcher.search_algorithm = 0 or 1 : KD-tree
    DNN = "DNN"

    def __init__(self,
                 num_features,
                 method):
        """ Constructor.

        :param num_features:
            The number of features to be detected and matched afterwards.
        :type num_features: int

        :param method:
            The string name of the chosen detecting-method.
        :type method: str
        """
        self.num_features = num_features

        # method choice
        if method == self.ORB:
            self.core = cv2.ORB_create(self.num_features)
        elif method == self.SIFT:
            self.core = cv2.SIFT_create(self.num_features)
        elif method == self.DNN:
            # TODO : link and develop Detecting Deep Neural Network
            print('\033[91m' + 'DNN to be done yet...' + '\033[0m')
        else:
            print('\033[91m' + 'Method not found' + '\033[0m')

    @staticmethod
    def _preprocess(img):  # : Frame
        """ Private static method useful to preprocess the image in grayscale
        and in a built-in way within the class.

        :param img:
            Image in RGB to be processed into grayscale.
        :type img: Frame

        :return:
            Grayscale image.
        :rtype: image
        """
        # checking that the image has more than one channel
        assert img.shape[2] > 1

        # returning it back to grayscale
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def detect_and_compute(self,
                           img):  # : Frame
        """ Merge the behaviour of all possible core techniques in one function
        in purpose of detecting and computing features.

        :param img:
            Image to be feature-detected.
        :rtype img: Frame

        :returns:
            Key-points and Descriptors
        :rtype: Tuple[object, object]
        """
        # before the proper detection it is needed a grayscale preprocess
        return self.core.detectAndCompute(self._preprocess(img), None)
