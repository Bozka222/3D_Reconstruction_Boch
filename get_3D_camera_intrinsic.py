import pyrealsense2 as rs
import cv2 as cv

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
intrinsic = depth_stream.as_video_stream_profile().get_intrinsics()
print(intrinsic)

print("Saving parameters!")
cv_file = cv.FileStorage('Data/Input/3D_intrinsic.xml', cv.FILE_STORAGE_WRITE)
cv_file.write('Intrinsic', intrinsic)

cv_file.release()
