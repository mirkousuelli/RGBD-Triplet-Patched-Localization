import cv2


class Detector:
    ORB = 'ORB'
    SIFT = 'SIFT'
    DNN = 'DNN'

    def __init__(self, _num_features, _method):
        self.num_features = _num_features

        if _method == self.ORB:
            self.core = cv2.ORB_create(self.num_features)
            print('[Detector] ORB detector initialized')
        elif _method == self.SIFT:
            self.core = cv2.SIFT_create(self.num_features)
            print('[Detector] SIFT detector initialized')
        elif _method == self.DNN:
            print('[Detector][WIP] DNN to be done yet...')
        else:
            print('\033[91m' + '[Detector][ERROR] Method not found' + '\033[0m')

    @staticmethod
    def __preprocess(_img):
        assert _img.shape[2] > 1
        print('[Detector] Image preprocessed to gray scale')
        return cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)

    def detectAndCompute(self, _img):
        print('[Detector] Key points and descriptors are going to be detected')
        return self.core.detectAndCompute(self.__preprocess(_img), None)
