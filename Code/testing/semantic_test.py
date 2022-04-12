import cv2

from camera.Frame import Frame
from camera.Action import Action

from tools.Merger import Merger
from tools.SemanticSampling import SemanticSampling

# image loading
img_1 = Frame("../../Dataset/Colors/00000-color.png",
              "../../Dataset/Depths/00000-depth.png", 0)
img_2 = Frame("../../Dataset/Colors/00060-color.png",
              "../../Dataset/Depths/00060-depth.png", 60)
action = Action(img_1, img_2)

# objects initialization
merger = Merger(num_features=5000,
                detector_method="ORB",
                matcher_method="FLANN")
merger.merge_action(action)
model = SemanticSampling()

f_matrix, mask = model.ransac_fundamental_matrix(action,
                                                 sampling_rate=0.4,
                                                 error=0.8)
action.set_inliers(mask)

inliers_image = merger.merge_inliers(action)

cv2.imshow("Inliers", inliers_image)
cv2.waitKey()
