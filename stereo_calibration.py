import numpy as np
import cv2 as cv
import glob

# ---------------------- # FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS # ---------------------- #

chessboardSize = (9, 6)  # Corners does not count (11,7)(9,6) Must be width x height
frameSize = (720, 1280)  # (1024, 576)(720, 1280)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

size_of_chessboard_squares_mm = 80  # Set size for chessboard (30)(80)
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpointsL = []  # 2d points in image plane.
imgpointsR = []  # 2d points in image plane.

imagesLeft = sorted(glob.glob('Data/Input/Camera_Calibration_Images/stereoLeft/*.png'))
imagesRight = sorted(glob.glob('Data/Input/Camera_Calibration_Images/stereoRight/*.png'))
global imgL, imgR, grayL, grayR
i = 0

for imgLeft, imgRight in zip(imagesLeft, imagesRight):
    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if retL and retR:
        objpoints.append(objp)
        # Window size is size of window around found corner in which can this corner be refined to another coordination.
        cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR)

        # Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.imshow('Img_Left', imgL)
        cv.imwrite(f"Data/Output/Chessboard_with_corners/Left/Img_Left_{i}.png", imgL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.imshow('Img_Right', imgR)
        cv.imwrite(f"Data/Output/Chessboard_with_corners/Right/Img_Right_{i}.png", imgR)
        cv.waitKey(1000)

        i += 1

cv.destroyAllWindows()

# ---------------------- # CALIBRATION - GET ALL PARAMETERS # ---------------------- #

# Returns RMSE(Per pixel projection error) , Intrinsic matrix, Distortion coefficients,
# Rotation vector and Translation vector)
retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))
print(f"Projection Error - Single Camera Calibration:\n"
      f"Left Camera: {retL}\n"
      f"Right Camera: {retR}\n"
      f"LCameraIntrinsic:\n{newCameraMatrixL}\n"
      f"RCameraIntrinsic:\n{newCameraMatrixR}\n"
      f"DistortionCoefL:\n{distL}\n"
      f"DistortionCoefR:\n{distR}\n")

# ---------------------- # STEREO VISION CALIBRATION # ---------------------- #

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camera matrices so that only Rot, Trans, Emat and Fmat are calculated.
# Hence, intrinsic parameters are the same

criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# This step is performed to transformation between the two cameras and calculate Essential and Fundamental matrix
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = (cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags))
print(f"Stereo Calibration Parameters:\n"
      f"Projection Error: {retStereo}\n"
      f"LCameraIntrinsic:\n{newCameraMatrixL}\n"
      f"RCameraIntrinsic:\n{newCameraMatrixR}\n"
      f"Rotation:\n{rot}\n"
      f"Translation:\n{trans}\n"
      f"EssentialMatrix:\n{essentialMatrix}\n"
      f"FundamentalMatrix:\n{fundamentalMatrix}\n"
      f"DistortionCoefL:{distL}\n"
      f"DistortionCoefR:{distR}\n")

# ---------------------- # STEREO RECTIFICATION # ---------------------- #

rectifyScale = 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = (cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale, (0, 0)))

print(f"Rectification Parameters:\n"
      f"Rotation Matrix L:\n{rectL}\n"
      f"Rotation Matrix R:\n{rectR}\n"
      f"Projection Matrix L:\n{projMatrixL}\n"
      f"Projection Matrix R:\n{projMatrixR}\n"
      f"Q:\n{Q}\n")

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

print("Saving parameters!")
cv_file = cv.FileStorage('Data/Input/stereoMap.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x', stereoMapL[0])
cv_file.write('stereoMapL_y', stereoMapL[1])
cv_file.write('stereoMapR_x', stereoMapR[0])
cv_file.write('stereoMapR_y', stereoMapR[1])
cv_file.write('q', Q)
cv_file.write('PL', projMatrixL)
cv_file.write('PR', projMatrixR)
cv_file.write('F', fundamentalMatrix)
cv_file.write('KR', newCameraMatrixR)

cv_file.release()
