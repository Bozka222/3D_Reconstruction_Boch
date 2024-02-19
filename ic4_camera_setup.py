# import the opencv library
import cv2

# define a video capture object
vid = cv2.VideoCapture(3, cv2.CAP_DSHOW)
width = vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
height = vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
vid.set(cv2.CAP_PROP_FPS, 15)
print(width, height)
i = 0
while True:

    # Capture the video frame by frame
    ret, frame = vid.read()

    if frame.any():
        # Display the resulting frame
        cv2.imshow('RGB_CAM', frame)
        cv2.imwrite(f"Data/Output/RGB_CAM/RGB_image{i}.jpg", frame)
        i += 1

    fps = vid.get(cv2.CAP_PROP_FPS)
    print('Video frame rate={0}'.format(fps))
    print('Resolution: {0}x{1}'.format(frame.shape[1], frame.shape[0]))

    key = cv2.waitKey(1)
    if key == ord("\x1b"):  # End stream when pressing ESC
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
