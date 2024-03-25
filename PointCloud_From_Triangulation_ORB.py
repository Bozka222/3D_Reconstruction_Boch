import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt


def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())

        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])

        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('Data/Input/stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

P1 = cv_file.getNode('PL').mat()
P2 = cv_file.getNode('PR').mat()
F = cv_file.getNode('F').mat()

imgL = cv2.imread('Data/Output/Dataset/Stereo_Data/Stereo_Left_Image/Stereo_Left_Image3.jpg', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('Data/Output/Dataset/Stereo_Data/Stereo_Right_Image/Stereo_Right_Image4.jpg', cv2.IMREAD_GRAYSCALE)

# imgL = cv2.imread('Data/Input/Images_Without_BG/Stereo_Left_Image3.png', cv2.IMREAD_GRAYSCALE)
# imgR = cv2.imread('Data/Input/Images_Without_BG/Stereo_Right_Image4.png', cv2.IMREAD_GRAYSCALE)

# Show the frames
cv2.imshow("frame right", imgR)
cv2.imshow("frame left", imgL)
cv2.waitKey(0)

# Undistort and rectify images
imgR = cv2.remap(imgR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
imgL = cv2.remap(imgL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

# Show the frames
cv2.imshow("frame right", imgR)
cv2.imshow("frame left", imgL)
cv2.waitKey(0)

# Detect the SIFT key points and compute the descriptors for the two images
orb = cv2.BRISK.create()

# # FREAK FEATURE DESCRIPTOR
# freak = cv2.xfeatures2d.FREAK.create()
# star = cv2.xfeatures2d.StarDetector.create()
# keyPointsLeft = star.detect(imgL, None)
# keyPointsRight = star.detect(imgR, None)
# keyPointsLeft, descriptorsLeft = freak.compute(imgL, keyPointsLeft)
# keyPointsRight, descriptorsRight = freak.compute(imgR, keyPointsRight)

# # BRIEF DETECTOR
# star = cv2.xfeatures2d.StarDetector.create()
# # Initiate BRIEF extractor
# brief = cv2.xfeatures2d.BriefDescriptorExtractor.create()
# # find the keypoints with STAR
# keyPointsLeft = star.detect(imgL, None)
# keyPointsRight = star.detect(imgR, None)
# # compute the descriptors with BRIEF
# keyPointsLeft, descriptorsLeft = brief.compute(imgL, keyPointsLeft)
# keyPointsRight, descriptorsRight = brief.compute(imgR, keyPointsRight)

keyPointsLeft, descriptorsLeft = orb.detectAndCompute(imgL, None)
keyPointsRight, descriptorsRight = orb.detectAndCompute(imgR, None)

# draw only keypoints location,not size and orientation
keypointsL = cv2.drawKeypoints(imgL, keyPointsLeft, None, color=(0, 255, 0), flags=0)
keypointsR = cv2.drawKeypoints(imgR, keyPointsRight, None, color=(0, 255, 0), flags=0)
plt.imshow(keypointsL)
plt.show()

# # FLANN Parameters
# FLANN_INDEX_LSH = 6
# flann_params = dict(algorithm=FLANN_INDEX_LSH,
#                     table_number=6,
#                     key_size=12,
#                     multi_probe_level=1)
# search_params = dict(checks=500)
#
# # FLANN Matcher
# flann = cv2.FlannBasedMatcher(flann_params, search_params)
#
# # Matching
# matches = flann.knnMatch(descriptorsLeft, descriptorsRight, k=2)
# print(matches)
# good = []
# pts1 = []
# pts2 = []
#
# # Get Matched points under distance's threshold
# for (m, n) in enumerate(matches):
#     if m.distance < 1.0 * n.distance:
#         good.append([m])
#         pts2.append(keyPointsRight[m.trainIdx].pt)
#         pts1.append(keyPointsLeft[m.queryIdx].pt)
#
# # Draws the small circles on the locations of keypoints
# img_matched = cv2.drawMatchesKnn(imgL, keyPointsLeft, imgR, keyPointsRight, good, None, flags=2)
# cv2.imshow("Matches", img_matched)
# cv2.waitKey(0)
#
# # Print number of matched feature points
# print('Matched Num:', len(pts1))

# Create a Brute Force Matcher object.
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Perform the matching between the ORB descriptors of the training image and the test image
matches = bf.match(descriptorsLeft, descriptorsRight)

# The matches with shorter distance are the ones we want.
matches = sorted(matches, key=lambda x: x.distance)
print(matches)
good = []
pts1 = []
pts2 = []

# Get Matched points under distance's threshold
for m in matches:
    good.append([m])
    pts2.append(keyPointsRight[m.trainIdx].pt)
    pts1.append(keyPointsLeft[m.queryIdx].pt)

img_matched = cv2.drawMatches(imgL, keyPointsLeft, imgR, keyPointsRight, matches, None, flags=2)
cv2.imshow("Matches", img_matched)
cv2.waitKey(0)

# Print total number of matching points between the training and query images
print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))

# Set array for keypoints
pts1 = np.array(pts1)
pts2 = np.array(pts2)

# Find epilines corresponding to points in right image (second image) and drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(imgL, imgR, lines1, np.int32(pts1), np.int32(pts2))

# Find epilines corresponding to points in left image (first image) and drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(imgL, imgR, lines2, np.int32(pts1), np.int32(pts2))

plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.show()

# Triangulation
pts1 = np.transpose(pts1)
pts2 = np.transpose(pts2)

points_3d = cv2.triangulatePoints(P1, P2, pts1, pts2)
points_3d /= points_3d[3]

opt_variables = np.hstack((P2.ravel(), points_3d.ravel(order="F")))
num_points = len(pts2[0])

X = []
Y = []
Z = []

X = np.concatenate((X, points_3d[0]))
Y = np.concatenate((Y, points_3d[1]))
Z = np.concatenate((Z, points_3d[2]))

points = np.zeros((num_points, 3))
points[:, 0] = X
points[:, 1] = Y
points[:, 2] = Z

# Visualization
pcd = o3d.geometry.PointCloud()
pc_points = np.array(points, np.float32)
pc_color = np.array([], np.float32)

# RGB to BGR for pc_points
imgLU = cv2.cvtColor(imgL, cv2.COLOR_RGB2BGR)

point_color = np.transpose(pts1)

for i in range(len(point_color)):
    u = np.int32(point_color[i][1])
    v = np.int32(point_color[i][0])

    # pc_colors
    pc_color = np.append(pc_color, np.array(np.float32(imgLU[u][v] / 255)))
    pc_color = np.reshape(pc_color, (-1, 3))

# add position and color to point cloud
pcd.points = o3d.utility.Vector3dVector(pc_points)
pcd.colors = o3d.utility.Vector3dVector(pc_color)
o3d.visualization.draw_geometries([pcd])
cv2.destroyAllWindows()
