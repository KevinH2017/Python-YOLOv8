from ultralytics import YOLO

# Load the model, the bigger the model, the more accurate the prediction
model = YOLO("yolov8n.pt") 

# Run classification prediction on source using the model
results = model.predict("./images/cat_dog.jpg")
result = results[0]             # Takes the first image

# print(result.names)       # Object IDs with their associated names

for box in result.boxes:
    cords = box.xyxy[0].tolist()                        # Gets coordinates of boxes within image
    cords = [round(x) for x in cords]                   # Rounded to nearest whole number
    class_id = result.names[box.cls[0].item()]          # Checks what classification ID of what they are
    conf = round(box.conf[0].item(), 2)                 # How confident the classifications are, rounded to 2 decimal places

    print("Object type:", class_id)
    print("Coordinates:", cords)
    print("Probability:", conf)
    print("--------------------------")
