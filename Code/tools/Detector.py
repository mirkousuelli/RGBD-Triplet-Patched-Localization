"""
Project : RGB-D Semantic Sampling
Authors : Marco Petri and Mirko Usuelli
-----------------------------------------------
Degree : M.Sc. Computer Science and Engineering
Course : Image Analysis and Computer Vision
Professor : Vincenzo Caglioti
Advisors : Giacomo Boracchi, Luca Magri
University : Politecnico di Milano - (A.Y. 2021/2022)
"""
import cv2


class Detector:
    """
    Class implementing the tool 'Detector' able to detect relevant
    corners/features in a single image.
    """

    # Techniques available:
    ORB = 'ORB'
    SIFT = 'SIFT'
    DNN = 'DNN'

    def __init__(self, _num_features, _method):
        """
        Constructor.
        I : number of features to be detected and the desired detecting method
        O : -
        """
        self.num_features = _num_features

        # method choice
        if _method == self.ORB:
            self.core = cv2.ORB_create(self.num_features)
        elif _method == self.SIFT:
            self.core = cv2.SIFT_create(self.num_features)
        elif _method == self.DNN:
            # TODO : link and develop Detecting Deep Neural Network
            print('\033[91m' + '[Detector][WIP] DNN to be done yet...' + '\033[0m')
        else:
            print('\033[91m' + '[Detector][ERROR] Method not found' + '\033[0m')

    @staticmethod
    def __preprocess(_img):
        """
        Private static method useful to preprocess the image in grayscale and in a built-in way within the class.
        I : image in RGB
        O : image in grayscale
        """
        # checking that the image has more than one channel
        assert _img.shape[2] > 1

        # returning it back to grayscale
        return cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)

    def detectAndCompute(self, _img):
        """
        Merge the behaviour of all possible core techniques in one function in purpose
        of detecting and computing features.
        I : image
        O : key points within the given image + their associated descriptors
        """
        # before the proper detection it is needed a grayscale preprocess
        return self.core.detectAndCompute(self.__preprocess(_img), None)
