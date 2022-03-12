import cv2
from Detector import Detector
from Matcher import Matcher

# image loading
img_1 = cv2.imread('../Dataset/00000-color.png')
img_2 = cv2.imread('../Dataset/00060-color.png')

# objects initialization
detector = Detector(50, 'ORB')
matcher = Matcher(50, 'FLANN', 0.8)

# detecting
key_points_1, descriptors_1 = detector.detectAndCompute(img_1)
key_points_2, descriptors_2 = detector.detectAndCompute(img_2)

# matching
matches = matcher.match(descriptors_1, descriptors_2)
final_img = matcher.drawMatches(img_1, key_points_1, img_2, key_points_2, matches)

# Show the final image
cv2.imshow("Matches", final_img)
cv2.waitKey()
