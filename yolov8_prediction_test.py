from ultralytics import YOLO

# Load the model, the bigger the model, the more accurate the prediction
model = YOLO("yolov8n.pt") 

# Run classification prediction on source using the model
results = model.predict("./images/bus.jpg")

# Process results
for result in results:
    boxes = result.boxes                # Boxes object for bounding box outputs
    masks = result.masks                # Masks object for segmentation masks outputs
    keypoints = result.keypoints        # Keypoints object for pose outputs
    probs = result.probs                # Probs object for classification outputs
    obb = result.obb                    # Oriented boxes object for OBB outputs
    result.show()                       # Display to screen
    result.save(filename="result.jpg")  # Save to file

