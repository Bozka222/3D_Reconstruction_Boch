import pyrealsense2 as rs
import cv2
import numpy as np
import json
import time

# Create point cloud object
pc = rs.pointcloud()
points = rs.points()
# saver = rs.save_single_frameset("Data/Output/PointClouds/3D_Cam/frameset")

# Declare RealSense pipeline, encapsulating the actual device and sensors
pipeline = rs.pipeline()
config = rs.config()

# Getting Camera Info
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
depth_sensor = device.first_depth_sensor()

# Set Depth and Color stream parameters
config.enable_stream(rs.stream.color, 1280, 720, format=rs.format.bgr8, framerate=30)
config.enable_stream(rs.stream.depth, 1280, 720, format=rs.format.z16, framerate=30)

# Start streaming with chosen configuration
profile = pipeline.start(config)
# Start timer
start = time.time()
depth_sensor.set_option(rs.option.global_time_enabled, 1)
color_sensor = profile.get_device().query_sensors()[1]
color_sensor.set_option(rs.option.global_time_enabled, 1)

# Set RGB Camera
vid = cv2.VideoCapture(3, cv2.CAP_DSHOW)
# vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# vid.set(cv2.CAP_PROP_FPS, 15)

# Create filters for depth
spat_filter = rs.spatial_filter(0.40, 20.0, 5.0, 0.0)  # Spatial - edge-preserving spatial smoothing
temp_filter = rs.temporal_filter(0.8, 30.0, 8)  # Temporal - reduces temporal noise
hole_filter = rs.hole_filling_filter(2)

# Set variables to default
i = 0
j = 0
metadata_3D = {}
metadata_RGB = {}

while True:
    # Wait for the next set of frames from the cameras
    frames = pipeline.wait_for_frames()
    depth_raw_frame = frames.get_depth_frame()
    color_raw_frame = frames.get_color_frame()
    _, RGB_frame = vid.read()

    # Filter depth frames
    filtered = spat_filter.process(depth_raw_frame)
    filtered1 = temp_filter.process(filtered)
    depth_filtered_frame = hole_filter.process(filtered1)
    # saver.process(depth_filtered_frame)

    # Create 3D vertices
    points = pc.calculate(depth_filtered_frame)
    vertices = np.asanyarray(points.get_vertices(dims=2))

    # Get images for visualization
    depth_image = np.asanyarray(depth_filtered_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.025), cv2.COLORMAP_JET)
    color_image = np.asanyarray(color_raw_frame.get_data())
    blue_image = cv2.cvtColor(color_image, cv2.COLOR_RGBA2BGRA)
    cropped_img = RGB_frame[0:720, 0:1280]

    # Rotate images to show them
    rotated_depth_colormap = cv2.rotate(depth_colormap, cv2.ROTATE_90_CLOCKWISE)
    rotated_color_image = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)
    rotated_RGB_image = cv2.rotate(cropped_img, cv2.ROTATE_90_CLOCKWISE)

    cv2.imshow("Color_Image", rotated_color_image)
    cv2.imshow("Color_Image_ForDepth", rotated_depth_colormap)
    cv2.imshow('RGB_CAM', rotated_RGB_image)

    metadata_3D[j] = {
        "frame_number": frames.frame_number,
        "capture_start_time_stamp": start,
        "global_depth_time_stamp": depth_raw_frame.get_timestamp() / 1000,
        "global_color_time_stamp": color_raw_frame.get_timestamp() / 1000,
    }

    metadata_RGB[j] = {
        "frame_number": i,
        "capture_start_time_stamp": start,
        "global_color_time_stamp": time.time(),
    }

    j += 1

    vertices.astype("float32").tofile(f"Data/Output/Dataset/Depth_Data/Raw_Depth/Raw_Depth{i}.raw")
    cv2.imwrite(f"Data/Output/Dataset/Depth_Data/Raw_Color/Raw_Color{i}.jpg", blue_image)
    cv2.imwrite(f"Data/Output/Dataset/Depth_Data/Depth_Color_Image/Depth_Color_Image{i}.jpg", rotated_depth_colormap)
    cv2.imwrite(f"Data/Output/Dataset/Stereo_Data/Stereo_Left_Image/Stereo_Left_Image{i}.jpg", rotated_color_image)
    cv2.imwrite(f"Data/Output/Dataset/Stereo_Data/Stereo_Right_Image/Stereo_Right_Image{i}.jpg", rotated_RGB_image)
    key = cv2.waitKey(1)

    i += 1
    if key == ord("\x1b"):  # End stream when pressing ESC
        break

pipeline.stop()
vid.release()
cv2.destroyAllWindows()

with open(f"Data/Output/Dataset/Metadata/Metadata_3D.txt", "w") as metadata_file_1:
    metadata_file_1.write(json.dumps(metadata_3D))

with open(f"Data/Output/Dataset/Metadata/Metadata_RGB.txt", "w") as metadata_file_2:
    metadata_file_2.write(json.dumps(metadata_RGB))

print(metadata_3D)
print(metadata_RGB)
