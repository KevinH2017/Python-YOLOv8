from ultralytics import YOLO

# Load the model, the bigger the model, the more accurate the prediction
model = YOLO("yolov8x.pt") 

# Run inference on source using the model, "0" is the webcam
results = model.predict(source="0", show=True)

# Prints how many of each object and the confidence of it
print(results)