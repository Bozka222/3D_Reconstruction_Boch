# import cv2 as cv
# import sys
# import numpy as np
#
# print("Your OpenCV version is: " + cv.__version__)
#
# img = cv.imread("Data\starry_night.jpg")
#
# if img is None:
#     sys.exit("Could not read the image.")
#
# cv.imshow("Display window", img)
# k = cv.waitKey(0)
# if k == ord("s"):
#     cv.imwrite("Data\starry_night.png", img)
#
# img = cv.imread("Data\starry_night.png")
# cv.imshow("2. obrazek", img)
# k = cv.waitKey(0)

import cv2
img = cv2.imread('Data/starry_night.jpg') # load a dummy image
while(1):
    cv2.imshow('img',img)
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break
    elif k==-1:  # normally -1 returned,so don't print it
        continue
    else:
        print(k) # else print its value
