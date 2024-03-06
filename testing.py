import pyrealsense2 as rs
import cv2
import open3d as o3d
import numpy as np

pc = rs.pointcloud()
points = rs.points()

# Declare RealSense pipeline, encapsulating the actual device and sensors
pipeline = rs.pipeline()
config = rs.config()

# Getting Camera Info
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
depth_sensor = device.first_depth_sensor()
# Get depth scale of the device
depth_scale = depth_sensor.get_depth_scale()

# Enable depth stream
config.enable_stream(rs.stream.color, 1280, 720, format=rs.format.bgr8, framerate=30)
config.enable_stream(rs.stream.depth, 1280, 720, format=rs.format.z16, framerate=30)

# Start streaming with chosen configuration
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

# We'll use the colorizer to generate texture for our PLY
# (alternatively, texture can be obtained from color or infrared stream)
colorizer = rs.colorizer(1)

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

    points = pc.calculate(depth_filtered_frame)
    vertices = np.asanyarray(points.get_vertices(dims=2))
    w = depth_raw_frame.get_width()
    image_Points = np.reshape(vertices, (-1, w, 3))

    depth_image = np.asanyarray(depth_filtered_frame.get_data())
    color_image = np.asanyarray(color_raw_frame.get_data()).reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(vertices.astype(np.float32) / 255)
    pcd.colors = o3d.utility.Vector3dVector(color_image.astype(np.float64) / 255)
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud("Data/Output/PointClouds/3D_Cam/3D_Camera_PointCloud.ply", pcd, format="ply")

    # # Create save_to_ply object
    # ply = rs.save_to_ply("Data/Output/PointClouds/3D_Cam/3D.ply")
    #
    # # Set options to the desired values
    # # In this example we'll generate a textual PLY with normals (mesh is already created by default)
    # ply.set_option(rs.save_to_ply.option_ply_binary, False)
    # ply.set_option(rs.save_to_ply.option_ply_normals, True)
    # ply.set_option(rs.save_to_ply.option_ply_mesh, False)
    #
    # print("Saving to 1.ply...")
    # # Apply the processing block to the frameset which contains the depth frame and the texture
    # ply.process(colorized)
    # print("Done")

    # pc.map_to(color_raw_frame)
    # points = pc.calculate(depth_raw_frame)
    # print("Saving to 1.ply...")
    # points.export_to_ply("Data/Output/PointClouds/3D_Cam/3D.ply", color_raw_frame)
    # print("Done")
finally:
    pipeline.stop()
