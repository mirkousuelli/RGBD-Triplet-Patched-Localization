import numpy as np
import cv2

# image loading
img_1 = cv2.imread('../../Dataset/00000-color.png')
img_2 = cv2.imread('../../Dataset/00060-color.png')

# convert it to grayscale
img_1_bw = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img_2_bw = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

# detector initialization
detector = cv2.ORB_create()

# matcher initialization
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

# extract features
key_points_1, descriptors_1 = detector.detectAndCompute(img_1_bw, None)
key_points_2, descriptors_2 = detector.detectAndCompute(img_2_bw, None)

# match descriptors through kNN
matches = matcher.knnMatch(descriptors_1, descriptors_2, k=2)

# filter the good ones
good = []
try:
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
except ValueError:
    pass

# plotting settings
draw_params = dict(matchColor=-1,  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=None,  # draw only inliers
                   flags=2)
final_img = cv2.drawMatches(img_1, key_points_1,
                            img_2, key_points_2,
                            good[:10], None, **draw_params)
final_img = cv2.resize(final_img, (1280, 480))

# Show the final image
cv2.imshow("Matches", final_img)
cv2.waitKey()
