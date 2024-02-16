import json
import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Create pipeline and config
pipeline = rs.pipeline()
config = rs.config()

# Getting Camera Info
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.firmware_version))
print(device_product_line)

# Set configuration parameters for streams,  Open CV supports BGRA formats!!!!
config.enable_stream(rs.stream.color, 1280, 720, format=rs.format.bgr8, framerate=30)
config.enable_stream(rs.stream.depth, 1280, 720, format=rs.format.z16, framerate=30)

# Start pipeline with configuration
profile = pipeline.start(config)
start = time.time()

# Turn-on global time, torn-off auto-exposure
depth_sensor = profile.get_device().query_sensors()[0]
depth_sensor.set_option(rs.option.global_time_enabled, 1)
depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
color_sensor = profile.get_device().query_sensors()[1]
color_sensor.set_option(rs.option.global_time_enabled, 1)
color_sensor.set_option(rs.option.enable_auto_exposure, 0)

vid = cv2.VideoCapture(1)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
# Alignment means transforming depth frame to RGB frames because they come from slightly different view
# (Not time alignment)

# align_to = rs.stream.color
# align = rs.align(align_to)

i = -1
j = 0
metadata_3D = {}
metadata_RGB = {}

# Stream continues
while True:
    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()

    # Align the depth frame to color frame
    # aligned_frames = align.process(frames)

    # Get frames
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    ret, frame = vid.read()
    fps = vid.get(cv2.CAP_PROP_FPS)
    print('Video frame rate={0}'.format(fps))

    # Validate that both frames are valid
    if not depth_frame or not color_frame:
        continue

    # print(
    #     f" Frame Number {frames.frame_number}"
    #     f" Start: {start}"
    #     # f" Sensor: {aligned_frames.get_frame_metadata(rs.frame_metadata_value.sensor_timestamp) / 1000. / 1000.}"
    #     # f" Frame: {aligned_frames.get_frame_metadata(rs.frame_metadata_value.frame_timestamp) / 1000. / 1000.}"
    #     # f" Backend: {aligned_frames.get_frame_metadata(rs.frame_metadata_value.backend_timestamp)}"
    #     # f" Arrival: {aligned_frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival) / 1000}"
    #     f" Global (depth): {depth_frame.get_timestamp() / 1000}"
    #     f" Global (color): {color_frame.get_timestamp() / 1000}"
    #     f" Diff (depth): {(depth_frame.get_timestamp() / 1000) - start}"
    #     f" Diff (color): {(color_frame.get_timestamp() / 1000) - start}"
    # )

    metadata_3D[j] = {
        "frame_number": frames.frame_number,
        "capture_start_time_stamp": start,
        "global_depth_time_stamp": depth_frame.get_timestamp()/1000,
        "global_color_time_stamp": color_frame.get_timestamp()/1000,
    }

    metadata_RGB[j] = {
        "frame_number": i,
        "capture_start_time_stamp": start,
        "global_color_time_stamp": time.time(),
    }

    j += 1

    # Convert image frame to numpy array
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET)
    # gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # images = np.hstack((color_image, depth_colormap))
    # Show stream with openCV
    # camera = np.concatenate((color_image, depth_cm), axis=0)

    # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('RealSense', images)
    # cv2.waitKey(1)
    cv2.imshow("RGB_Stream", color_image)
    cv2.imshow("Depth_Stream", depth_colormap)
    cv2.imshow('RGB_CAM', frame)
    key = cv2.waitKey(1)

    # if key == ord("s"):

    # if i == -1:
    #     time.sleep(1)  # Wait for the color sensor
    #     i += 1

    cv2.imwrite(f"Pictures/Output/Color_image/Color_image{i}.jpg", color_image)
    cv2.imwrite(f"Pictures/Output/Depth_image/Depth_image{i}.jpg", depth_colormap)
    cv2.imwrite(f"Pictures/Output/RGB_CAM/RGB_image{i}.jpg", frame)
    i += 1
    if key == ord("\x1b"):  # End stream when pressing ESC
        cv2.destroyAllWindows()
        break

# End stream
pipeline.stop()
vid.release()
cv2.destroyAllWindows()

with open(f"../../../../Dropbox/Diplomka/3D_Reconstruction_Boch/Pictures/Output/Metadata/metadata.txt", "w") as metadata_file_1:
    metadata_file_1.write(json.dumps(metadata_3D))

with open(f"Pictures/Output/Metadata/metadata_RGB.txt", "w") as metadata_file_2:
    metadata_file_2.write(json.dumps(metadata_RGB))

print(metadata_3D)
print(metadata_RGB)
