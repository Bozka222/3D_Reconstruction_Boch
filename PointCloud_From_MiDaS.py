import cv2
import numpy as np
import open3d as o3d
import glob


def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')


cv_file = cv2.FileStorage()
cv_file.open('Data/Input/stereoMap.xml', cv2.FileStorage_READ)

Q = cv_file.getNode('Q').mat()
print(Q)

depth_maps = sorted(glob.glob('Data/Output/PointClouds/MiDaS_Depth/*.png'))
color_images = sorted(glob.glob('Data/Output/Dataset/Stereo_Data/Stereo_Right_Image/*.jpg'))
i = 0

for depth_map, color_image in zip(depth_maps, color_images):

    img = cv2.imread(color_image)
    # cv2.imshow("color img", img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.waitKey(0)
    depthmap = cv2.imread(depth_map, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    output = cv2.normalize(depthmap, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    # plt.imshow(output)
    # plt.show()

    # Reproject points into 3D
    points_3D = cv2.reprojectImageTo3D(output, Q, handleMissingValues=False)
    points_3D = points_3D/100000000

    # Get rid of points with value 0 (i.e. no depth)
    mask_map = output > output.min()

    # Mask colors and points.
    output_points = points_3D[mask_map]
    output_colors = img[mask_map]

    output_file = f"Data/Output/PointClouds/Stereo/Uncropped/Right_Cam_PointCloud_Uncropped_{i}.ply"

    # Generate point cloud
    create_output(output_points, output_colors, output_file)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = output_points
    # pcd.colors = output_colors

    pcd = o3d.io.read_point_cloud(f"Data/Output/PointClouds/Stereo/Uncropped/Right_Cam_PointCloud_Uncropped_{i}.ply")
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])

    corners = np.array([[0.009, 0.010, -0.000],  # Ford Corners
                        [0.009, -0.008, -0.000],
                        [0.009, -0.008,  -0.021],
                        [0.009, 0.010,  -0.021],
                        [-0.007, 0.010,  -0.000],
                        [-0.007, -0.008,  -0.000],
                        [-0.007, -0.008,  -0.021],
                        [-0.007, 0.010,  -0.021]])

    # Convert the corners array to have type float64
    bounding_polygon = corners.astype("float64")

    # Create a SelectionPolygonVolume
    vol = o3d.visualization.SelectionPolygonVolume()

    vol.orthogonal_axis = "Y"
    vol.axis_max = np.max(bounding_polygon[:, 1])
    vol.axis_min = np.min(bounding_polygon[:, 1])

    bounding_polygon[:, 1] = 0
    vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)
    cropped_pcd = vol.crop_point_cloud(pcd)

    bounding_box = cropped_pcd.get_axis_aligned_bounding_box()
    bounding_box.color = (1, 0, 0)

    # o3d.visualization.draw_geometries([cropped_pcd, bounding_box])
    o3d.io.write_point_cloud(f"Data/Output/PointClouds/Stereo/Cropped/Right_Cam_PointCloud_Cropped_{i}.ply",
                             cropped_pcd, format="ply")
    print(f"Point Cloud {i} has been created.")
    i += 1
