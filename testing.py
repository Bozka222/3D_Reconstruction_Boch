import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt


def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255,
                                        3).tolist())

        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int,
                     [c, -(r[2] + r[0] * c) / r[1]])

        img1 = cv2.line(img1,
                        (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1,
                          tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2,
                          tuple(pt2), 5, color, -1)
    return img1, img2

# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('Data/Input/stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

Q = cv_file.getNode('q').mat()
P1 = cv_file.getNode('PL').mat()
P2 = cv_file.getNode('PR').mat()
F = cv_file.getNode('F').mat()
print(Q)

imgL = cv2.imread('Data/Output/Dataset/Stereo_Data/Stereo_Left_Image/Color_image65.jpg')
imgR = cv2.imread('Data/Output/Dataset/Stereo_Data/Stereo_Right_Image/RGB_image65.jpg')

imgLgray = cv2.imread('Data/Output/Dataset/Stereo_Data/Stereo_Left_Image/Color_image65.jpg', cv2.IMREAD_GRAYSCALE)
imgRgray = cv2.imread('Data/Output/Dataset/Stereo_Data/Stereo_Right_Image/RGB_image65.jpg', cv2.IMREAD_GRAYSCALE)

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
