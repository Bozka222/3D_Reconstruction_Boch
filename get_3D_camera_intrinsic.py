import pyrealsense2 as rs
import cv2 as cv
import numpy as np

# Create pipeline and config
pipeline = rs.pipeline()
config = rs.config()

# Set configuration parameters for streams,  Open CV supports BGRA formats!!!!
config.enable_stream(rs.stream.color, 1280, 720, format=rs.format.bgr8, framerate=30)
config.enable_stream(rs.stream.depth, 1280, 720, format=rs.format.z16, framerate=30)

# Start pipeline with configuration
profile = pipeline.start(config)

# Get intrinsic
depth_stream = profile.get_stream(rs.stream.depth)
color_stream = profile.get_stream(rs.stream.color)
depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

print("\n Depth intrinsics: " + str(depth_intrinsics))
print("\n Color intrinsics: " + str(color_intrinsics))

intrinsic_mat = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx], [0, depth_intrinsics.fy, depth_intrinsics.ppy], [0, 0, 1]])
print(intrinsic_mat)

print("Saving parameters!")
cv_file = cv.FileStorage('Data/Input/3D_intrinsic.xml', cv.FILE_STORAGE_WRITE)
cv_file.write('Intrinsic', intrinsic_mat)

cv_file.release()
