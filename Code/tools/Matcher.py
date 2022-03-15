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
    """
    Class implementing the tool 'Matcher' able to match relevant
    descriptors in an action, namely two images.
    """

    # Techniques available:
    FLANN = 'FLANN'
    DNN = 'DNN'

    def __init__(self, _num_features, _method):
        """
        Constructor.
        I : number of features to be detected and the desired matching method
        O : -
        """
        self.num_features = _num_features

        # method choice
        if _method == self.FLANN:
            # FLANN hyper-parameters by default
            index_params = dict(algorithm=6, table_number=6, key_size=12,
                                multi_probe_level=1)
            search_params = dict(checks=self.num_features)
            self.core = cv2.FlannBasedMatcher(indexParams=index_params,
                                              searchParams=search_params)
        elif _method == self.DNN:
            # TODO : link and develop Matching Deep Neural Network
            print('\033[91m' + ' DNN matcher to be done yet...' + '\033[0m')
        else:
            print('\033[91m' + ' Method not found' + '\033[0m')

    @staticmethod
    def _filter(_matches):
        """
        Private static method useful to filter the images matching in a
        built-in way within the class.
        I : matching alternatives list
        O : selected matching list
        """
        # empty list which will be enriched of inliers matching
        good = []
        try:
            for m, n in _matches:
                # appending the best distance between the two alternatives
                good.append(m if m.distance < n.distance else n)
            # sorting from the closest to the worst matching
            # found in terms of distances
            good.sort(key=lambda x: x.distance, reverse=False)
            return good
        except ValueError:
            print('\033[91m' + 'kNN matching error' + '\033[0m')
            pass

    def match(self, _descriptors_1, _descriptors_2):
        """
        Merge the behaviour of all possible core techniques in one function
        in purpose of matching descriptors of the two images.
        I : images' descriptors
        O : selected matching list
        """
        matches = self.core.knnMatch(_descriptors_1, _descriptors_2, k=2)
        matches = [x for x in matches if len(x) == 2]
        return self._filter(matches)

    @staticmethod
    def draw_matches(_img_1, _key_points_1, _img_2, _key_points_2, _matches,
                     limit=-1):
        """
        Private static method to be used to draw the final result of the
        matching procedure
        I : images + images' key points (i.e. features) + matching list
        (+ matching visual limit in drawing)
        O : action image (i.e. merge of the two frames + matching links
        highlighted between key points)
        """
        # pre-conditions
        assert _img_1.shape == _img_2.shape

        # hyper-parameters before drawing
        draw_params = dict(matchColor=-1,  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=None,  # draw only inliers
                           flags=2)

        # proper drawing method
        final_img = cv2.drawMatches(_img_1, _key_points_1,
                                    _img_2, _key_points_2,
                                    _matches[:limit], None, **draw_params)

        return cv2.resize(final_img, (_img_1.shape[1] * 2, _img_1.shape[0]))
