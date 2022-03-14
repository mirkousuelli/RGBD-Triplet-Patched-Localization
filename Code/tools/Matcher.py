"""
Project : RGB-D Semantic Sampling
Authors : Marco Petri and Mirko Usuelli
-----------------------------------------------
Degree : M.Sc. Computer Science and Engineering
Course : Image Analysis and Computer Vision
Professor : Vincenzo Caglioti
Advisors : Giacomo Boracchi, Luca Magri
University : Politecnico di Milano
A.Y. : 2021 - 2022
"""
import cv2


class Matcher:
    FLANN = 'FLANN'
    DNN = 'DNN'

    def __init__(self, _num_features, _method):
        self.num_features = _num_features

        if _method == self.FLANN:
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=self.num_features)
            self.core = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        elif _method == self.DNN:
            print('\033[91m' + '[Matcher][WIP] DNN matcher to be done yet...' + '\033[0m')
        else:
            print('\033[91m' + '[Matcher][ERROR] Method not found' + '\033[0m')

    @staticmethod
    def __filter(_matches):
        good = []
        try:
            for m, n in _matches:
                good.append(m if m.distance < n.distance else n)
            good.sort(key=lambda x: x.distance, reverse=False)
            return good
        except ValueError:
            print('\033[91m' + '[Matcher][ERROR] kNN matching error' + '\033[0m')
            pass

    def match(self, _descriptors_1, _descriptors_2):
        matches = self.core.knnMatch(_descriptors_1, _descriptors_2, k=2)
        matches = [x for x in matches if len(x) == 2]
        return self.__filter(matches)

    @staticmethod
    def drawMatches(_img_1, _key_points_1, _img_2, _key_points_2, _matches, limit=-1):
        assert _img_1.shape == _img_2.shape
        draw_params = dict(matchColor=-1,  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=None,  # draw only inliers
                           flags=2)
        final_img = cv2.drawMatches(_img_1, _key_points_1,
                                    _img_2, _key_points_2,
                                    _matches[:limit], None, **draw_params)
        return cv2.resize(final_img, (_img_1.shape[1] * 2, _img_1.shape[0]))
