import pyrealsense2 as rs
import cv2
import numpy as np

pc = rs.pointcloud()
points = rs.points()
saver = rs.save_single_frameset("Data/Output/PointClouds/3D_Cam/frameset")

# Declare RealSense pipeline, encapsulating the actual device and sensors
pipeline = rs.pipeline()
config = rs.config()

# Getting Camera Info
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
depth_sensor = device.first_depth_sensor()

# Enable depth stream
config.enable_stream(rs.stream.color, 1280, 720, format=rs.format.bgr8, framerate=30)
config.enable_stream(rs.stream.depth, 1280, 720, format=rs.format.z16, framerate=30)

# Start streaming with chosen configuration
profile = pipeline.start(config)

spat_filter = rs.spatial_filter(0.40, 20.0, 5.0, 0.0)  # Spatial - edge-preserving spatial smoothing
temp_filter = rs.temporal_filter(0.8, 30.0, 8)  # Temporal - reduces temporal noise
hole_filter = rs.hole_filling_filter(2)

try:
    # Wait for the next set of frames from the camera
    frames = pipeline.wait_for_frames()
    depth_raw_frame = frames.get_depth_frame()
    color_raw_frame = frames.get_color_frame()

    filtered = spat_filter.process(depth_raw_frame)
    filtered1 = temp_filter.process(filtered)
    depth_filtered_frame = hole_filter.process(filtered1)
    saver.process(depth_filtered_frame)

    points = pc.calculate(depth_filtered_frame)
    vertices = np.asanyarray(points.get_vertices(dims=2))

    depth_image = np.asanyarray(depth_filtered_frame.get_data())
    color_image = np.asanyarray(color_raw_frame.get_data())
    image = cv2.cvtColor(color_image, cv2.COLOR_RGBA2BGRA)

    vertices.astype("float32").tofile("Data/Output/PointClouds/3D_Cam/depth.raw")
    cv2.imwrite("Data/Output/PointClouds/3D_Cam/color_raw.jpg", image)
    cv2.imwrite("Data/Output/PointClouds/3D_Cam/color_image.jpg", color_image)

finally:
    pipeline.stop()
