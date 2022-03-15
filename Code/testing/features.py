import cv2
from tools.Detector import Detector
from tools.Matcher import Matcher
from tools.Merger import Merger

# image loading
img_1 = cv2.imread('../../Dataset/Colors/00000-color.png')
img_2 = cv2.imread('../../Dataset/Colors/00060-color.png')

# objects initialization
merger = Merger(Detector(50, 'ORB'), Matcher(50, 'FLANN'))

# Show the final image
cv2.imshow("Matches", merger.merge(img_1, img_2, _limit=20))
cv2.waitKey()
