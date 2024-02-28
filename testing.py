import cv2
import open3d as o3d

# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('Data/Input/stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

img_left = cv2.imread('Data/Output/Color_image/Color_image20.jpg')
img_right = cv2.imread('Data/Output/RGB_CAM/RGB_image20.jpg')

# img_left = cv2.imread('Data/Input/Camera_Calibration_Images/stereoLeft/imageL0.png')
# img_right = cv2.imread('Data/Input/Camera_Calibration_Images/stereoRight/imageR0.png')

# img_left = cv2.imread('Data/Input/Deformed/Im_L_1.png')
# img_right = cv2.imread('Data/Input/Deformed/Im_R_1.png')

# cropped_img = img_right[0:1280, 0:720]
cv2.imshow("frame right", img_right)
cv2.imshow("frame left", img_left)

# pcd = o3d.io.read_point_cloud("Data/Output/PointClouds/3D_Cam/point_cloud1.ply", remove_nan_points=True,
#                               remove_infinite_points=True)
# pc = o3d.io.read_point_cloud("Data/Output/PointClouds/Stereo/3D.ply", remove_nan_points=False,
#                              remove_infinite_points=False)
#
# o3d.visualization.draw_geometries([pcd])

# Undistorted and rectify images
frame_right = cv2.remap(img_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
frame_left = cv2.remap(img_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

# Show the frames
# cv2.imshow("image right", img_right)
# cv2.imshow("image left", img_left)
# cv2.imshow("frame right", frame_right)
# cv2.imshow("frame left", frame_left)
cv2.imwrite(f"Data/Output/rectified_img/RIR_0.jpg", frame_right)
cv2.imwrite(f"Data/Output/rectified_img/RIL_0.jpg", frame_left)

# here it should be the pause
k = cv2.waitKey(0)
if k == 27:  # wait for ESC key to exit
    cv2.destroyAllWindows()
