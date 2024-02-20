import open3d as o3d
import matplotlib.pyplot as plt
import cv2

# Load in color and depth image to create the point cloud
color_raw = o3d.io.read_image("Data/Output/Color_image/Color_image0.jpg")
depth_raw = o3d.io.read_image("Data/Output/Depth_image/Depth_image0.jpg")
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=color_raw, depth=depth_raw)
print(rgbd_image)

# Plot the images
plt.subplot(1, 2, 1)
plt.title('Grayscale image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Depth image')
plt.imshow(rgbd_image.depth)
plt.show()

# Camera intrinsic parameters from camera used to get color and depth images - Camera Calibration
cv_file = cv2.FileStorage()
cv_file.open('Data/Input/3D_intrinsic.xml', cv2.FileStorage_READ)

camera_intrinsic = cv_file.getNode('Intrinsic').mat()
print(camera_intrinsic)

# Set the intrinsic camera parameters
camera_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720, fx=camera_intrinsic[0][0],
                                                         fy=camera_intrinsic[1][1], cx=camera_intrinsic[0][2],
                                                         cy=camera_intrinsic[1][2])
print(camera_intrinsic_o3d.intrinsic_matrix)

# Create the point cloud from images and camera intrinsic parameters
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_image, intrinsic=camera_intrinsic_o3d)

# Flip it, otherwise the point cloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd], zoom=0.5)
