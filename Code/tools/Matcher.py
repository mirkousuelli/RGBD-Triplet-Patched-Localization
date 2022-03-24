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
import numpy as np

from camera.Frame import Frame
from camera.Action import Action

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
    def _filter(img_1: Frame,
                img_2: Frame,
                matches,
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
                    img_1.points.append(img_1.key_points[m.queryIdx].pt)
                    img_2.points.append(img_2.key_points[m.trainIdx].pt)
                    good.append(m)
            img_1.points = np.int32(img_1.points)
            img_2.points = np.int32(img_2.points)
            return good
        except ValueError:
            print('\033[91m' + 'Filter matching error' + '\033[0m')
            pass

    def match_frames(self,
                     img_1: Frame,
                     img_2: Frame):
        """ Merge the behaviour of all possible core techniques in one function
        in purpose of matching descriptors of the two images.

        :param img_1:
            First image.
        :type img_1: Frame

        :param img_2:
            Second image.
        :type img_2: Frame

        :return:
            Good matches which passed the Lowe's test.
        :rtype: list
        """
        matches = self.core.knnMatch(img_1.descriptors, img_2.descriptors, k=2)
        matches = [x for x in matches if len(x) == 2]
        return self._filter(img_1, img_2, matches, self.filter_test)

    def match_action(self,
                     action: Action):
        """ Merge the behaviour of all possible core techniques in one function
        in purpose of matching descriptors of the two images.

        :param action:
            Action of two frames.
        :type action: Action

        :return:
            Good matches which passed the Lowe's test.
        :rtype: list
        """
        action.matches = self.core.knnMatch(action.first.descriptors,
                                            action.second.descriptors,
                                            k=2)
        action.links = [x for x in action.matches if len(x) == 2]
        action.links = self._filter(action.first, action.second,
                                    action.links, self.filter_test)

    @staticmethod
    def draw_frames_matches(img_1: Frame,
                            img_2: Frame,
                            matches,
                            limit=-1):
        """ Private static method to be used to draw the final result of the
        matching procedure.

        :param img_1:
            First image.
        :type img_1: Frame

        :param img_2:
            Second image.
        :type img_2: Frame

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
        assert img_1.get_size() == img_2.get_size(), "Images do not have the" \
                                                     " same size!"

        # hyper-parameters before drawing
        draw_params = dict(matchColor=-1,  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=None,  # draw only inliers
                           flags=2)

        # proper drawing method
        final_img = cv2.drawMatches(img_1.get_cv2_images(ret="rgb"),
                                    img_1.key_points,
                                    img_2.get_cv2_images(ret="rgb"),
                                    img_2.key_points,
                                    matches[:limit],
                                    None, **draw_params)

        width, height = img_1.get_size()
        return cv2.resize(final_img, (width * 2, height))

    @staticmethod
    def draw_action_matches(action: Action,
                            limit=-1):
        """ Private static method to be used to draw the final result of the
        matching procedure.

        :param action:
            Action of two frames.
        :type action: Action

        :param limit:
            Integer number which limits how many matching links to be drawn.
        :type limit: int

        :return:
            The two images merged into one image with matching links drawn.
        :rtype: image
        """
        # pre-conditions
        assert action.first.get_size() == action.second.get_size()

        # hyper-parameters before drawing
        draw_params = dict(matchColor=-1,  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=None,  # draw only inliers
                           flags=2)

        # proper drawing method
        final_img = cv2.drawMatches(action.first.get_cv2_images(ret="rgb"),
                                    action.first.key_points,
                                    action.second.get_cv2_images(ret="rgb"),
                                    action.second.key_points,
                                    action.links[:limit],
                                    None, **draw_params)

        width, height = action.first.get_size()
        return cv2.resize(final_img, (width * 2, height))

    @staticmethod
    def draw_inliers_matches(action: Action):

        # pre-conditions
        assert action.first.get_size() == action.second.get_size()

        # hyper-parameters before drawing
        draw_params = dict(matchColor=-1,  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=None,  # draw only inliers
                           flags=2)

        # proper drawing method
        final_img = cv2.drawMatches(action.first.get_cv2_images(ret="rgb"),
                                    action.first.inliers,
                                    action.second.get_cv2_images(ret="rgb"),
                                    action.second.inliers,
                                    action.links[:limit],
                                    None, **draw_params)

        width, height = action.first.get_size()
        return cv2.resize(final_img, (width * 2, height))
