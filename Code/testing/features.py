import cv2
from tools.Merger import Merger
from camera.Frame import Frame

# image loading
# img_1 = cv2.imread('../../Dataset/Colors/00000-color.png')
# img_2 = cv2.imread('../../Dataset/Colors/00060-color.png')
frame_1 = Frame("../../Dataset/Colors/00000-color.png",
                "../../Dataset/Depths/00000-depth.png")
frame_2 = Frame("../../Dataset/Colors/00060-color.png",
                "../../Dataset/Depths/00060-depth.png")

# objects initialization
merger = Merger(num_features=2000,
                detector_method="ORB",
                matcher_method="FLANN")

# Show the final image
cv2.imshow("Matches", merger.merge(frame_1, frame_2))
cv2.waitKey()
