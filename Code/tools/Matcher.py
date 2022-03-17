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


class Matcher:
    """ Class.
    Class implementing the tool 'Matcher' able to match relevant
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

        Parameters
        ----------
        num_features : int
            The number of features to be detected and matched afterwards.

        method : str
            The string name of the chosen matching-method.

        search_algorithm : int
            Search algorithm used by the matcher that must be recognizable also
            by the detecting method.

        filter_test : float
            Value used to filter matching features through Lowe's Test.
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
        """ Static Method.
        Private static method useful to filter the images matching in a built-in
        way within the class.

        Parameters
        ----------
        matches : list
            Matched features.

        filter_test : int
            Value used to filter matching features through Lowe's Test.
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
        """ Method.
        Merge the behaviour of all possible core techniques in one function in
        purpose of matching descriptors of the two images.

        Parameters
        ----------
        descriptors_1 : list
            Feature descriptors of the first image.

        descriptors_2 : list
            Feature descriptors of the second image.
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
        """ Static Method.
        Private static method to be used to draw the final result of the
        matching procedure.

        Parameters
        ----------
        img_1 : Frame
            First image.

        key_points_1 : list
            Key points of the first image, i.e. the features.

        img_2 : Frame
            Second image.

        key_points_2 : list
            Key points of the second image, i.e. the features.

        matches : list
            Matched features.

        limit : int
            Integer number which limits how many matching links to be drawn.
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
