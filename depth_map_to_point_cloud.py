import open3d as o3d
import cv2
import numpy as np
import pyrealsense2 as rs
import glob
from bagpy import bagreader

pc = rs.pointcloud()
points = rs.points()

# depth_filtered_frame = bagreader('Data/Output/PointClouds/3D_Cam/frameset1.bag')
# print(type(depth_filtered_frame))
# points = pc.calculate(depth_filtered_frame)
# vertices = np.asanyarray(points.get_vertices(dims=2))

raw_depths = sorted(glob.glob('Data/Output/Dataset/Depth_Data/Raw_Depth/*.raw'))
raw_colors = sorted(glob.glob('Data/Output/Dataset/Depth_Data/Raw_Color/*.jpg'))

i = 0
for raw_depth, raw_color in zip(raw_depths, raw_colors):
    raw_vertices = np.fromfile(raw_depth, dtype=np.float32).reshape(-1, 3)
    color_raw = cv2.imread(raw_color).reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(raw_vertices.astype(np.float64) / 255)
    pcd.colors = o3d.utility.Vector3dVector(color_raw.astype(np.float64) / 255)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.io.write_point_cloud(f"Data/Output/PointClouds/3D_Cam/3D_Camera_PointCloud_Uncropped{i}.ply", pcd, format="ply")

    # corners = np.array([[0.005, 0.009, -0.005],  # Skoda Corners
    #                     [0.005, -0.010, -0.005],
    #                     [0.005, -0.010,  -0.020],
    #                     [0.005, 0.009,  -0.020],
    #                     [-0.003, 0.009,  -0.005],
    #                     [-0.003, -0.010,  -0.005],
    #                     [-0.003, -0.010,  -0.020],
    #                     [-0.003, 0.009,  -0.020]])

    corners = np.array([[0.005, 0.009, -0.005],  # Ford Corners
                        [0.005, -0.008, -0.005],
                        [0.005, -0.008,  -0.021],
                        [0.005, 0.009,  -0.021],
                        [-0.003, 0.009,  -0.005],
                        [-0.003, -0.008,  -0.005],
                        [-0.003, -0.008,  -0.021],
                        [-0.003, 0.009,  -0.021]])

    # Convert the corners array to have type float64
    bounding_polygon = corners.astype("float64")

    # Create a SelectionPolygonVolume
    vol = o3d.visualization.SelectionPolygonVolume()

    # You need to specify what axis to orient the polygon to.
    # I choose the "Y" axis. I made the max value the maximum Y of
    # the polygon vertices and the min value the minimum Y of the
    # polygon vertices.
    vol.orthogonal_axis = "Y"
    vol.axis_max = np.max(bounding_polygon[:, 1])
    vol.axis_min = np.min(bounding_polygon[:, 1])

    # Set all the Y values to 0 (they aren't needed since we specified what they
    # should be using just vol.axis_max and vol.axis_min).
    bounding_polygon[:, 1] = 0

    # Convert the np.array to a Vector3dVector
    vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)

    # Crop the point cloud using the Vector3dVector
    cropped_pcd = vol.crop_point_cloud(pcd)

    # Get a nice looking bounding box to display around the newly cropped point cloud
    # (This part is optional and just for display purposes)
    bounding_box = cropped_pcd.get_axis_aligned_bounding_box()
    bounding_box.color = (1, 0, 0)

    # Draw the newly cropped PCD and bounding box
    # o3d.visualization.draw_geometries([cropped_pcd, bounding_box])

    # o3d.visualization.draw_geometries([car])
    o3d.io.write_point_cloud(f"Data/Output/PointClouds/3D_Cam/3D_Camera_PointCloud{i}.ply", cropped_pcd, format="ply")
    i += 1
