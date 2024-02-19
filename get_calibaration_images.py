import cv2
import numpy as np
import pyrealsense2 as rs

# Set RGB Intel Camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, format=rs.format.bgr8, framerate=30)
profile = pipeline.start(config)

# Set RGB Image camera
cap = cv2.VideoCapture(3, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 15)

num = 0

while cap.isOpened():

    # Get all frames
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    success, img = cap.read()
    # new_img = cv2.resize(img, (1280, 720))
    cropped_img = img[0:720, 0:1280]

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite('Data/Input/Camera_Calibration_Images/stereoLeft/imageL' + str(num) + '.png', color_image)
        cv2.imwrite('Data/Input/Camera_Calibration_Images/stereoRight/imageR' + str(num) + '.png', cropped_img)
        # cv2.imwrite('Data/Input/Deformed/imageL' + str(num) + '.png', color_image)
        # cv2.imwrite('Data/Input/Deformed/imageR' + str(num) + '.png', cropped_img)
        print("images saved!")
        num += 1

    cv2.imshow('stereoLeft', color_image)
    cv2.imshow('stereoRight', cropped_img)

# Release and destroy all windows before termination
cap.release()
pipeline.stop()

cv2.destroyAllWindows()
