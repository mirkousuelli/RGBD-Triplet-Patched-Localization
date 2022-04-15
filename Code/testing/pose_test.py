from camera.Frame import Frame
from camera.Action import Action
import numpy as np
import cv2

from tools.Merger import Merger


def compute_epipolar_lines_opencv(
	action: Action
):
	# first frame
	action.first.epi_lines = cv2.computeCorrespondEpilines(
		action.second.points.reshape(-1, 1, 2), 2, action.f_matrix
	)
	action.first.epi_lines = action.first.epi_lines.reshape(-1, 3)

	# second frame
	action.second.epi_lines = cv2.computeCorrespondEpilines(
		action.first.points.reshape(-1, 1, 2), 1, action.f_matrix
	)
	action.second.epi_lines = action.second.epi_lines.reshape(-1, 3)


def compute_epipolar_lines(
	action: Action
):
	# first frame
	action.first.epi_lines = cv2.computeCorrespondEpilines(
		action.second.points.reshape(-1, 1, 2), 2, action.f_matrix
	)
	action.first.epi_lines = action.first.epi_lines.reshape(-1, 3)

	# second frame
	action.second.epi_lines = cv2.computeCorrespondEpilines(
		action.first.points.reshape(-1, 1, 2), 1, action.f_matrix
	)
	action.second.epi_lines = action.second.epi_lines.reshape(-1, 3)

	action.first.epi_lines, action.second.epi_lines = action.second.epi_lines, \
	                                                  action.first.epi_lines


def draw_epipolar_lines(
		frame: Frame,
):
	img = frame.get_cv2_images(ret="rgb")
	_, c = frame.get_size()

	# fixed the seed to match the same color in the other frame
	np.random.seed(42)

	i = 0
	# for each epipolar lines and inlier point relying on both frames
	for lines, pt in zip(frame.epi_lines, frame.points):
		# choose random color
		color = np.random.randint(0, 255, 3).tolist()

		# select two point for the epipolar lines
		x0, y0 = map(int, [0, -lines[2] / lines[1]])
		x1, y1 = map(int, [c, -(lines[2] + lines[0] * c) / lines[1]])

		# print lines for the epipolar lines
		img = cv2.line(img, (x0, y0), (x1, y1), color, 1)

		# print circles fo the inliner key points
		img = cv2.circle(img, tuple(pt), 5, color, -1)

		if i == 20:
			break
		else:
			i += 1

	return img


def show_epipolar_lines(
		action: Action
):
	# draw epipolar lines for each of the two frames in the action object
	img1 = draw_epipolar_lines(action.first)
	img2 = draw_epipolar_lines(action.second)

	# return the final image
	return np.concatenate((img1, img2), axis=1)


img_1 = Frame(
	"../../Dataset/Colors/00000-color.png",
	"../../Dataset/Depths/00000-depth.png", 0
)
img_2 = Frame(
	"../../Dataset/Colors/00060-color.png",
	"../../Dataset/Depths/00060-depth.png", 60
)

act = Action(img_1, img_2)
merger = Merger(num_features=5000,
                detector_method="ORB",
                matcher_method="FLANN")

merge_image = merger.merge_action(act)

print(img_2.extract_pose())
print(img_2.from_pose_to_rototrasl())

act.f_matrix, _ = act.compute_fundamental_matrix(inplace=False)
print("opencv F=")
print(act.f_matrix)
# compute_points(act)
compute_epipolar_lines_opencv(act)
epi_image = show_epipolar_lines(act)
# Show the final image
cv2.imshow("OpenCV", epi_image)

act.f_matrix = img_2.from_rototrasl_to_f_matrix()
print("ground truth F=")
print(act.f_matrix)
# compute_points(act)
compute_epipolar_lines(act)
epi_image = show_epipolar_lines(act)
# Show the final image
cv2.imshow("Ground Truth", epi_image)
cv2.waitKey()
