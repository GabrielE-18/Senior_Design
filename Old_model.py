from ultralytics import YOLO
import cv2

# Load the YOLOv8 model (you can replace 'yolov8x.pt' with 'yolov8n.pt', 'yolov8l.pt', etc.)
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' for faster, lightweight model

# Path to video or live stream
capture = '/Users/gabri/School/Senior/cattle_vid.mp4'  # Replace with your video path or use 0 for webcam

# Open video capture
cap = cv2.VideoCapture(capture)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model.predict(source=frame, show=False, save=False, conf=0.5)  # conf is the confidence threshold

    # Annotate detections
    annotated_frame = results[0].plot()  # Draw bounding boxes and labels on frame

    # Display the frame
    cv2.imshow('YOLOv8 Cow Detection', annotated_frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
