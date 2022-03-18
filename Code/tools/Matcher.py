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


class Matcher:
    """ Class implementing the tool 'Matcher' able to match relevant
    descriptors in an action, namely two images.
    """

    # Techniques available:
    FLANN = 'FLANN'
    DNN = 'DNN'

    def __init__(self,
                 num_features,
                 method,
                 search_algorithm=6,
                 filter_test=0.7):
        """ Constructor.

        :param num_features
            The number of features to be detected and matched afterwards.
        :type num_features: int

        :param method:
            The string name of the chosen matching-method.
        :type method: str

        :param search_algorithm:
            Search algorithm used by the matcher that must be recognizable also
            by the detecting method.
        :type search_algorithm: int

        :param filter_test:
            Value used to filter matching features through Lowe's Test.
        :type filter_test: float
        """
        self.num_features = num_features
        self.filter_test = filter_test

        # method choice
        if method == self.FLANN:
            # FLANN hyper-parameters by default
            if search_algorithm == 6:
                index_params = dict(algorithm=search_algorithm,
                                    table_number=6,
                                    key_size=12,
                                    multi_probe_level=1)
            else:
                index_params = dict(algorithm=search_algorithm,
                                    trees=5)
            search_params = dict(checks=self.num_features)
            self.core = cv2.FlannBasedMatcher(indexParams=index_params,
                                              searchParams=search_params)
        elif method == self.DNN:
            # TODO : link and develop Matching Deep Neural Network
            print('\033[91m' + 'DNN matcher to be done yet...' + '\033[0m')
        else:
            print('\033[91m' + 'Method not found' + '\033[0m')

    @staticmethod
    def _filter(matches,
                filter_test):
        """ Private static method useful to filter the images matching in a
        built-in way within the class.

        :param matches:
            Matched features.
        :type matches: list

        :param filter_test:
            Value used to filter matching features through Lowe's Test.
        :type: int

        :return:
            Good matches which passed the Lowe's test.
        :rtype: list
        """
        # empty list which will be enriched of inliers
        good = []
        try:
            for m, n in matches:
                # Lowe's test
                if m.distance < filter_test * n.distance:
                    good.append(m)
            return good
        except ValueError:
            print('\033[91m' + 'Filter matching error' + '\033[0m')
            pass

    def match(self,
              descriptors_1,
              descriptors_2):
        """ Merge the behaviour of all possible core techniques in one function
        in purpose of matching descriptors of the two images.

        :param descriptors_1:
            Feature descriptors of the first image.
        :type descriptors_1: object

        :param descriptors_2:
            Feature descriptors of the second image.
        :type descriptors_2: object

        :return:
            Good matches which passed the Lowe's test.
        :rtype: list
        """
        matches = self.core.knnMatch(descriptors_1, descriptors_2, k=2)
        matches = [x for x in matches if len(x) == 2]
        return self._filter(matches, self.filter_test)

    @staticmethod
    def draw_matches(img_1,  # : Frame
                     key_points_1,
                     img_2,  # : Frame
                     key_points_2,
                     matches,
                     limit=-1):
        """ Private static method to be used to draw the final result of the
        matching procedure.

        :param img_1:
            First image.
        :type img_1: Frame

        :param key_points_1:
            Key points of the first image, i.e. the features.
        :type key_points_1: list

        :param img_2:
            Second image.
        :type img_2: Frame

        :param key_points_2:
            Key points of the second image, i.e. the features.
        :type key_points_2: list

        :param matches:
            Matched features.
        :type matches: list

        :param limit:
            Integer number which limits how many matching links to be drawn.
        :type limit: int

        :return:
        The two images merged into one image with matching links drawn.
        :rtype: image
        """
        # pre-conditions
        assert img_1.shape == img_2.shape

        # hyper-parameters before drawing
        draw_params = dict(matchColor=-1,  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=None,  # draw only inliers
                           flags=2)

        # proper drawing method
        final_img = cv2.drawMatches(img_1, key_points_1,
                                    img_2, key_points_2,
                                    matches[:limit],
                                    None, **draw_params)

        return cv2.resize(final_img, (img_1.shape[1] * 2, img_1.shape[0]))
