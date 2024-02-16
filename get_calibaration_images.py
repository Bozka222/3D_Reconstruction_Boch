import cv2
import pyrealsense2 as rs

# Set RGB Intel Camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, format=rs.format.bgr8, framerate=30)
profile = pipeline.start(config)

# Set RGB Image camera
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

num = 0

while cap.isOpened():

    # Get all frames
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    success, img = cap.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite('Data/Input/Camera_Calibration_Images/stereoLeft/imageL' + str(num) + '.png', color_frame)
        cv2.imwrite('Data/Input/Camera_Calibration_Images/stereoRight/imageR' + str(num) + '.png', img)
        print("images saved!")
        num += 1

    cv2.imshow('stereoLeft', color_frame)
    cv2.imshow('stereoRight', img)

# Release and destroy all windows before termination
cap.release()
pipeline.stop()

cv2.destroyAllWindows()
