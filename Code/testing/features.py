import cv2

from camera.Frame import Frame
from camera.Action import Action

from tools.Merger import Merger
from tools.Localizer import Localizer

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
localizer = Localizer()

merge_image = merger.merge_action(action)

print("F = ")
print(localizer.compute_fundamental_matrix(action))
print("\n")
print("E = ")
print(localizer.compute_essential_matrix(action))
print("\n")
print("R, t = ")
print(localizer.roto_translation(action))
print("\n")
print("Q = ")
print(localizer.from_rot_to_quat(action))
print("\n")
print("R* = ")
print(localizer.from_quat_to_rot(localizer.from_rot_to_quat(action)))

localizer.compute_inliers(action)
localizer.compute_epipolar_lines(action)
epi_image = localizer.show_epipolar_lines(action)

# Show the final image
cv2.imshow("Matches", merge_image)
cv2.imshow("EpiLines", epi_image)
cv2.waitKey()
