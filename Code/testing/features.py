import cv2

from camera.Frame import Frame
from camera.Action import Action

from tools.Merger import Merger
from tools.Localizer import Localizer

# image loading
from utils.utils import get_str

first = 1
second = 2
img_1 = Frame("../../Dataset/Colors/00" + get_str(first) + "-color.png",
              "../../Dataset/Depths/00" + get_str(first) + "-depth.png", first)
img_2 = Frame("../../Dataset/Colors/00" + get_str(second) + "-color.png",
              "../../Dataset/Depths/00" + get_str(second) + "-depth.png", second)
action = Action(img_1, img_2)

print("Figure ", action.first._Frame__color_path, " and figure ", action.second._Frame__color_path)

# objects initialization
merger = Merger(num_features=5000,
                detector_method="ORB",
                matcher_method="FLANN")
localizer = Localizer()

merge_image = merger.merge_action(action)

print("F = ")
localizer.compute_fundamental_matrix(action)
print(action.f_matrix)
print("\n")
print("E = ")
localizer.compute_essential_matrix(action)
print(action.e_matrix)
print("\n")
print("R, t = ")
localizer.roto_translation(action, normalize_em=False)
print(action.R, action.t)
print("\n")
print("Q = ")
print(localizer.from_rot_to_quat(action, normalize_em=False))
print("\n")
print("R* = ")
print(localizer.from_quat_to_rot(localizer.from_rot_to_quat(action)))

localizer.compute_inliers(action)
localizer.compute_epipolar_lines(action)
epi_image = localizer.show_epipolar_lines(action)
inliers_image = merger.merge_inliers(action)

# Show the final image
cv2.imshow("Matches", merge_image)
cv2.imshow("EpiLines", epi_image)
cv2.imshow("Inliers", inliers_image)
cv2.waitKey()
