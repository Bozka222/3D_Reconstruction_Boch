import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt


# =====================================
# Function declarations
# =====================================

# Function to create point cloud file
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


# Function that Down samples image x number (reduce_factor) of times.
def down_sample_image(image, reduce_factor):
    for i in range(0, reduce_factor):
        # Check if image is color or grayscale
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape

        image = cv2.pyrDown(image, dstsize=(col // 2, row // 2))
    return image


# =========================================================
# Stereo 3D reconstruction
# =========================================================
# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('Data/Input/stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

Q = cv_file.getNode('q').mat()
print(Q)

imgL = cv2.imread('Data/Output/Color_image/Color_image0.jpg', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('Data/Output/RGB_CAM/RGB_image0.jpg', cv2.IMREAD_GRAYSCALE)

# Show the frames
cv2.imshow("frame right", imgR)
cv2.imshow("frame left", imgL)

cv2.waitKey(0)

# Undistort and rectify images
imgR = cv2.remap(imgR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
imgL = cv2.remap(imgL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

# Show the frames
cv2.imshow("frame right", imgR)
cv2.imshow("frame left", imgL)

cv2.waitKey(0)

# Down_sample each image 3 times (because they're too big)
# imgL = down_sample_image(imgL, 2)
# imgR = down_sample_image(imgR, 2)
#
# cv2.imshow("frame right", imgR)
# cv2.imshow("frame left", imgL)
#
# cv2.waitKey(0)

# Create Block matching object.
window_size = 0
min_disp = 0
nDispFactor = 1  # adjust this (14 is good)
num_disp = 16*nDispFactor-min_disp

stereo = cv2.StereoSGBM.create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=window_size,
                               P1=5*3*window_size**2,
                               P2=32*3*window_size**2,
                               disp12MaxDiff=10,
                               uniquenessRatio=1,
                               speckleWindowSize=1,
                               speckleRange=2,
                               preFilterCap=63,
                               mode=cv2.STEREO_SGBM_MODE_SGBM)

disparity_map = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

# # WLS Filtering for smoother disparity maps
# right_matcher = cv2.ximgproc.createRightMatcher(stereo)
# disparity_left = np.float32(stereo.compute(imgL, imgR))
# disparity_right = np.float32(right_matcher.compute(imgR, imgL))
#
# # Disparity WLS Filter
# wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
# wls_filter.setLambda(1000.0)  # Adjust these parameters based on your specific application
# wls_filter.setSigmaColor(1.5)
#
# filtered_disparity = wls_filter.filter(disparity_left, imgL, disparity_map_right=disparity_right)
#
# # Post-processing
# filtered_disparity = cv2.normalize(src=filtered_disparity, dst=filtered_disparity, beta=0, alpha=255,
#                                    norm_type=cv2.NORM_MINMAX)

# Show disparity map before generating 3D cloud to verify that point cloud will be usable.
# Display the disparity map
plt.imshow(disparity_map, 'gray')
plt.colorbar()
plt.show()

# Generate  point cloud.
print("\nGenerating the 3D map...")

# Get new down_sampled width and height
h, w = imgR.shape[:2]

# Reproject points into 3D
points_3D = cv2.reprojectImageTo3D(disparity_map, Q)

# Get rid of points with value 0 (i.e. no depth)
output_points = points_3D[disparity_map > disparity_map.min()]

# Get color points
colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
output_colors = colors[disparity_map > disparity_map.min()]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(output_points)
pcd.colors = o3d.utility.Vector3dVector(output_colors / 255.0)

# Visualize the point cloud
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd], window_name="Stereo point cloud", width=1280, height=720)

o3d.io.write_point_cloud("Data/Output/PointClouds/Stereo/Stereo_PointCloud.ply", pcd)

# # Define name for output file
# output_file = 'Data/Output/PointClouds/Stereo/point_cloud_stereo.ply'
#
# # Generate point cloud
# print("\n Creating the output file... \n")
# create_output(output_points, output_colors, output_file)
