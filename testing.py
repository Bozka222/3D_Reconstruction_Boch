import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

cv_file = cv2.FileStorage()
cv_file.open('Data/Input/stereoMap.xml', cv2.FileStorage_READ)

KR = cv_file.getNode('KR').mat()
print(KR)

depthmap = cv2.imread('Data/Output/PointClouds/DepthMiDaS.jpg').astype(np.float32)
depth_image_o3d = o3d.geometry.Image(depthmap)

o3d_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(720, 1280,
                                                          KR[0, 0], KR[0, 2],
                                                          KR[1, 1], KR[1, 2])
print(o3d_camera_intrinsic)

# Create the point cloud from the open3d depth image
pcd = o3d.geometry.PointCloud.create_from_depth_image(depth=depth_image_o3d, intrinsic=o3d_camera_intrinsic)
o3d.visualization.draw_geometries([pcd], window_name="MiDaS point cloud", width=1280, height=720)

# # Create pipeline and config
# pipeline = rs.pipeline()
# config = rs.config()
#
# # Set configuration parameters for streams,  Open CV supports BGRA formats!!!!
# config.enable_stream(rs.stream.color, 1280, 720, format=rs.format.bgr8, framerate=30)
# config.enable_stream(rs.stream.depth, 1280, 720, format=rs.format.z16, framerate=30)
#
# # Start pipeline with configuration
# profile = pipeline.start(config)
#
# # Get intrinsic
# depth_stream = profile.get_stream(rs.stream.depth)
# color_stream = profile.get_stream(rs.stream.color)
# depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
# color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
#
# # raw_vertices = np.fromfile('Data/Output/Dataset/Depth_Data/Raw_Depth/Raw_Depth4.raw', dtype=np.float32).reshape(-1, 3)
# # print(raw_vertices.shape)
# # color_raw = cv2.imread('Data/Output/Dataset/Depth_Data/Raw_Color/Raw_Color4.jpg', cv2.IMREAD_GRAYSCALE)
# # print(color_raw)
# # cv2.imshow("raw_color", color_raw)
# # cv2.waitKey(0)
# # print(color_raw.shape)
#
# pcd_load = o3d.io.read_point_cloud("Data/Output/PointClouds/3D_Cam/3D_Camera_PointCloud4.ply")
# vertices = np.asarray(pcd_load.points)
# print(vertices.shape)
#
# pixel_coordinates = []
# for vertex in vertices:
#     color_pixel_coordinate = rs.rs2_project_point_to_pixel(color_intrinsics, vertex)
#     pixel_coordinates.append(color_pixel_coordinate)
# print(pixel_coordinates)
# right_image = cv2.imread("Data/Output/Dataset/Stereo_Data/Stereo_Right_Image/Stereo_Right_Image5.jpg", cv2.IMREAD_GRAYSCALE)
# cv2.imshow("Right Image", right_image)
# cv2.waitKey(0)
# # depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, depth_pixel)
