import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load in color and depth image to create the point cloud
color_raw = o3d.io.read_image("Data/Output/Color_image/Color_image34.jpg")
depth_raw = o3d.io.read_image("Data/Output/Depth_image/Depth_image34.jpg")
color = o3d.geometry.Image(np.array(np.asarray(color_raw)[:, :, :3]).astype('uint8'))
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=color, depth=depth_raw,
                                                                convert_rgb_to_intensity=False,
                                                                depth_scale=1000,
                                                                depth_trunc=3.0)
print(rgbd_image)

# Plot the images
plt.subplot(1, 2, 1)
plt.title('Color image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Depth image')
plt.imshow(rgbd_image.depth)
plt.show()

# Camera intrinsic parameters from camera used to get color and depth images - Camera Calibration
cv_file = cv2.FileStorage()
cv_file.open('Data/Input/3D_intrinsic.xml', cv2.FileStorage_READ)

camera_intrinsic = cv_file.getNode('Intrinsic').mat()

# Set the intrinsic camera parameters
camera_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720, fx=camera_intrinsic[0][0],
                                                         fy=camera_intrinsic[1][1], cx=camera_intrinsic[0][2],
                                                         cy=camera_intrinsic[1][2])
print(camera_intrinsic_o3d.intrinsic_matrix)

# Create the point cloud from images and camera intrinsic parameters
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_image, intrinsic=camera_intrinsic_o3d)
# pc = o3d.geometry.PointCloud.create_from_depth_image(image=depth_raw, intrinsic=camera_intrinsic_o3d,
# depth_scale=1000)

# Flip it, otherwise the point cloud will be upside down
# o3d.geometry.PointCloud.remove_radius_outlier(pcd, 2, 5, True)

pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])
# pc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
# o3d.visualization.draw_geometries([pc])
o3d.io.write_point_cloud("Data/Output/PointClouds/3D_Cam/3D_Camera_PointCloud.ply", pcd, format="ply")
