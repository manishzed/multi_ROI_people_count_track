
# initialize video capture from your camera (0 is typically the default camera)
path =r"C:/Users/manish.kumar/Desktop/person_tracker_id_ft_age_gender/CCTV Recording.MP4"

import cv2
import numpy as np

# Global variables
drawing = False  # True if mouse is pressed
ix, iy = -1, -1  # Starting coordinates of the bounding box
fx, fy = -1, -1  # Ending coordinates of the bounding box

# Mouse callback function
def draw_rect(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        fx, fy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y

# Read the video file
video_capture = cv2.VideoCapture(path)

# Create a window and set the callback function
cv2.namedWindow('ROI Selection')
cv2.setMouseCallback('ROI Selection', draw_rect)

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    if ix != -1 and fx != -1:
        # Draw the rectangle on the frame
        cv2.rectangle(frame, (ix, iy), (fx, fy), (0, 255, 0), 2)

    cv2.imshow('ROI Selection', frame)
    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to quit and save the bounding box coordinates
    if key == ord('q'):
        break

# Release the video capture and destroy the window
video_capture.release()
cv2.destroyAllWindows()

# Print the selected ROI coordinates
if ix != -1 and fx != -1:
    print(f"ROI coordinates: Top-left: ({ix}, {iy}), Bottom-right: ({fx}, {fy})")
