from ultralytics import YOLO
import cv2

VIDEO_PATH = "./videos/cat_video1.mp4"

# Load the model
model = YOLO("yolov8x.pt") 

# Video object
cap = cv2.VideoCapture(VIDEO_PATH)

# Reads through frames of video object
ret = True
while ret:
    ret, frame = cap.read()

    # Detects and tracks objects
    results = model.track(frame, persist=True)

    # Plots the results
    frame_ = results[0].plot()

    # Visualize frames
    cv2.imshow("frame", frame_)

    # Exits loop with 'q' key
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break