import cv2

from camera.Frame import Frame
from camera.Action import Action

from tools.Merger import Merger

# image loading
from utils.transformation_utils import get_3x3_rotation_from_quaternion
from utils.utils import get_str

first = 0
second = 60
img_1 = Frame("../../Dataset/Testing/2/Colors/00" + get_str(first) + "-color.png",
              "../../Dataset/Testing/2/Depths/00" + get_str(first) + "-depth.png", first)
img_2 = Frame("../../Dataset/Testing/2/Colors/00" + get_str(second) + "-color.png",
              "../../Dataset/Testing/2/Depths/00" + get_str(second) + "-depth.png", second)
action = Action(img_1, img_2)

print("Figure ", action.first._Frame__color_path, " and figure ", action.second._Frame__color_path)

# objects initialization
merger = Merger(num_features=5000,
                detector_method="ORB",
                matcher_method="FLANN")

merge_image = merger.merge_action(action)

print("F = ")
action.compute_fundamental_matrix()
print(action.f_matrix)
print("\n")
print("E = ")
action.compute_essential_matrix()
print(action.e_matrix)
print("\n")
print("R, t = ")
action.roto_translation(normalize_em=False)
print(action.R, action.t)
print("\n")
print("Q = ")
print(action.from_rot_to_quat(normalize_em=False))
print("\n")
print("R* = ")
print(get_3x3_rotation_from_quaternion(action.from_rot_to_quat()))

action.compute_inliers()
action.compute_epipolar_lines()
epi_image = action.show_epipolar_lines()
inliers_image = merger.merge_inliers(action)

# Show the final image
cv2.imshow("Matches", merge_image)
cv2.imshow("EpiLines", epi_image)
cv2.imshow("Inliers", inliers_image)
cv2.waitKey()
