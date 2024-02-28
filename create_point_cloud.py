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

imgL = cv2.imread('Data/Output/Color_image/Color_image25.jpg', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('Data/Output/RGB_CAM/RGB_image25.jpg', cv2.IMREAD_GRAYSCALE)

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
imgL = down_sample_image(imgL, 2)
imgR = down_sample_image(imgR, 2)

# stereo = cv2.StereoBM_create(numDisparities=32, blockSize=9)
# For each pixel algorithm will find the best disparity from 0
# Larger block size implies smoother, though less accurate disparity map
# disparity = stereo.compute(imgL, imgR)

# Set disparity parameters
# Note: disparity range is tuned according to specific parameters obtained through trial and error.
win_size = 5
min_disp = -1
max_disp = 31  # min_disp * 9
num_disp = max_disp - min_disp  # Needs to be divisible by 16

# Create Block matching object.
stereo = cv2.StereoSGBM.create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=5,
                               uniquenessRatio=5,
                               speckleWindowSize=5,
                               speckleRange=5,
                               disp12MaxDiff=2,
                               P1=8 * 3 * win_size ** 2,  # 8*3*win_size**2,
                               P2=32 * 3 * win_size ** 2)  # 32*3*win_size**2)

# Compute disparity map
print("\nComputing the disparity  map...")
disparity_map = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

# Show disparity map before generating 3D cloud to verify that point cloud will be usable.
plt.imshow(disparity_map, 'gray')
# cv2.imshow('Disparity Map', (disparity_map - min_disp) / num_disp)
plt.show()

# Generate  point cloud.
print("\nGenerating the 3D map...")

# Get new down_sampled width and height
h, w = imgR.shape[:2]
# focal_length = 0.8*w
#
# # Perspective transformation matrix
# Q = np.float32([[1, 0, 0, -w/2.0],
#                 [0, -1, 0,  h/2.0],
#                 [0, 0, 0, -focal_length],
#                 [0, 0, 1, 0]])

# Reproject points into 3D
points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
# Get color points
colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

# Get rid of points with value 0 (i.e. no depth)
mask_map = disparity_map > disparity_map.min()

# Mask colors and points.
output_points = points_3D[mask_map]
output_colors = colors[mask_map]

# Define name for output file
output_file = 'Data/Output/PointClouds/Stereo/point_cloud_stereo.ply'

# Generate point cloud
print("\n Creating the output file... \n")
create_output(output_points, output_colors, output_file)

pcd = o3d.io.read_point_cloud("Data/Output/PointClouds/Stereo/point_cloud_stereo.ply", remove_nan_points=True,
                              remove_infinite_points=True)
o3d.visualization.draw_geometries([pcd])
cloudPoints = np.asarray(pcd.points)
