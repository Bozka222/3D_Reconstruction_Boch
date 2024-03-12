import cv2
import numpy as np

cv_file = cv2.FileStorage()
cv_file.open('Data/Input/stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
Q = cv_file.getNode('q').mat()

imgL = cv2.imread('Data/Output/Dataset/Stereo_Data/Stereo_Left_Image/Stereo_Left_Image29.jpg')
imgR = cv2.imread('Data/Output/Dataset/Stereo_Data/Stereo_Right_Image/Stereo_Right_Image30.jpg')

# Show the frames
# cv2.imshow("frame right", imgR)
# cv2.imshow("frame left", imgL)
#
# cv2.waitKey(0)

# Undistort and rectify images
imgR = cv2.remap(imgR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
imgL = cv2.remap(imgL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

# # Show the frames
cv2.imshow("frame right", imgR)
cv2.imshow("frame left", imgL)

cv2.waitKey(0)


def nothing(x):
    pass


cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp', 720, 1280)

cv2.createTrackbar('minDisparity', 'disp', 0, 100, nothing)
cv2.createTrackbar('numDisparities', 'disp', 1, 17, nothing)
cv2.createTrackbar('blockSize', 'disp', 0, 21, nothing)
cv2.createTrackbar('disp12MaxDiff', 'disp', 12, 100, nothing)
cv2.createTrackbar('preFilterCap', 'disp', 63, 63, nothing)
cv2.createTrackbar('uniquenessRatio', 'disp', 1, 100, nothing)
cv2.createTrackbar('speckleWindowSize', 'disp', 1, 200, nothing)
cv2.createTrackbar('speckleRange', 'disp', 2, 100, nothing)
cv2.createTrackbar('P1', 'disp', 5, 100, nothing)
cv2.createTrackbar('P2', 'disp', 32, 100, nothing)

# Creating an object of StereoBM algorithm
stereo = cv2.StereoSGBM.create()

while True:
    # Updating the parameters based on the trackbar positions
    numDisparities = cv2.getTrackbarPos('numDisparities', 'disp') * 16
    blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 1
    preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
    speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp')
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
    minDisparity = cv2.getTrackbarPos('minDisparity', 'disp') * (-1)
    P1 = cv2.getTrackbarPos('P1', 'disp') * 3 * blockSize ** 2
    P2 = cv2.getTrackbarPos('P2', 'disp') * 3 * blockSize ** 2

    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)
    stereo.setP1(P1)
    stereo.setP2(P2)

    # Calculating disparity using the StereoBM algorithm
    disparity = stereo.compute(imgL, imgR)
    # NOTE: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it
    # is essential to convert it to CV_32F and scale it down 16 times.

    # Converting to float32
    disparity = disparity.astype(np.float32)

    # Scaling down the disparity values and normalizing them
    disparity = (disparity / 16.0 - minDisparity) / numDisparities

    # # Normalize the disparity map to 0-255 range for better visualization
    # disparity_map_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
    #                                          dtype=cv2.CV_8U)
    #
    # # Convert the disparity map to a false color representation
    # disparity_map_color = cv2.applyColorMap(disparity_map_normalized, cv2.COLORMAP_JET)

    # Displaying the disparity map
    cv2.imshow("disp", disparity)
    cv2.imwrite("Data/Output/Disparity_Map/disp_01.png", disparity)

    # Close window using esc key
    if cv2.waitKey(1) == 27:
        break

print(f"P1: {P1}\nP2: {P2}\nblockSize: {blockSize}\nnumDisparities: {numDisparities}\nminDisparity: {minDisparity}")
