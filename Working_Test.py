print('hello world')

import cv2
print("OpenCV version:", cv2.__version__)

from ultralytics import YOLO
#import cv2

# Load the YOLOv8 model
model = YOLO('yolov8x.pt')  # Use 'yolov8n.pt', 'yolov8l.pt', etc., for different versions

# Path to video or live stream
#capture = '/Users/gabri/School/Senior/cattle_vid.mp4'  # /Users/gabri/School/Senior/cattle_vid.mp4 Replace with your video path or use 0 for webcam

# Open video capture
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use DirectShow instead of MSMF
#cap = cv2.VideoCapture(0, cv2.CAP_MSMF)  # Try MSMF if DirectShow fails


cattle_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model(frame, conf=0.5)  # Confidence threshold

    cattle_count = 0

    # Loop through detections in results
    for box in results[0].boxes:
        label = model.names[int(box.cls[0])]  # Get the class name
        if label == "cow":  # Check if the detected object is a cow
            cattle_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Save cattle count
    cattle_data.append(cattle_count)

    # Display cattle count on frame
    cv2.putText(frame, f"Cattle Count: {cattle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the annotated frame
    cv2.waitKey(1)

    cv2.imshow('YOLOv8 Cow Detection', frame)
    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()