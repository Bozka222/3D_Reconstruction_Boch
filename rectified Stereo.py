import cv2
import numpy as np
import pyrealsense2 as rs

# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('Data/Input/stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

# Set RGB Intel Camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, format=rs.format.bgr8, framerate=30)
profile = pipeline.start(config)

# Set RGB Image camera
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 15)

while cap.isOpened():

    # Get all frames
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    frame_left = np.asanyarray(color_frame.get_data())
    success, frame_right = cap.read()

    cropped_img = frame_right[0:720, 0:1280]
    rotated_image_R = cv2.rotate(cropped_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rotated_image_L = cv2.rotate(frame_left, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # # Undistorted and rectify images
    frame_right = cv2.remap(rotated_image_R, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frame_left = cv2.remap(rotated_image_L, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    # Show the frames
    cv2.imshow("frame right", frame_right)
    cv2.imshow("frame left", frame_left)

    key = cv2.waitKey(1)
    if key == ord("\x1b"):  # End stream when pressing ESC
        break

# Release and destroy all windows before termination
cap.release()
pipeline.stop()
cv2.destroyAllWindows()
