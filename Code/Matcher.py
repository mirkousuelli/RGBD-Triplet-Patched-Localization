import cv2


class Matcher:
    FLANN = 'FLANN'
    DNN = 'DNN'

    def __init__(self, _num_features, _method, _accuracy):
        self.num_features = _num_features
        self.accuracy = _accuracy

        if _method == self.FLANN:
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=self.num_features)
            self.core = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
            print('[Matcher] Flann matcher initialized')
        elif _method == self.DNN:
            print('[Matcher][WIP] DNN matcher to be done yet...')
        else:
            print('\033[91m' + '[Matcher][ERROR] Method not found' + '\033[0m')

    @staticmethod
    def __filter(_accuracy, _matches):
        good = []
        for m, n in _matches:
            if m.distance < _accuracy * n.distance:
                good.append(m)
        print('[Matcher] Matches successfully filtered')
        return good

    def match(self, _descriptors_1, _descriptors_2):
        matches = self.core.knnMatch(_descriptors_1, _descriptors_2, k=2)
        print('[Matcher] Matches computed')
        return self.__filter(self.accuracy, matches)

    @staticmethod
    def drawMatches(_img_1, _key_points_1, _img_2, _key_points_2, _matches):
        assert _img_1.shape == _img_2.shape
        draw_params = dict(matchColor=-1,  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=None,  # draw only inliers
                           flags=2)
        final_img = cv2.drawMatches(_img_1, _key_points_1,
                                    _img_2, _key_points_2,
                                    _matches, None, **draw_params)
        print('[Matcher] Matches ready to be drawn')
        return cv2.resize(final_img, (_img_1.shape[1] * 2, _img_1.shape[0]))
