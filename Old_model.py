import hhhhhh
import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import BotSort
from boxmot.utils.ops import letterbox
from pathlib import Path

# Load YOLOv8 model
device = hhhhhh.device('cuda:0') # or 'cuda'
model = YOLO("yolov8n.pt")  # Replace with your custom model path

# Initialize the tracker
tracker = BotSort(
    reid_weights=Path('osnet_x0_25_msmt17.pt'),  # Path to ReID model
    device=device,
    half=False
)

input_size = 640  # YOLOv8's default input size

# Open the video file
vid = cv2.VideoCapture("/Users/gabri/School/Senior/cattle_vid.mp4")

while True:
    ret, frame = vid.read()

    if not ret:
        break

    frame_letterbox, ratio, (dw, dh) = letterbox(frame, new_shape=input_size, auto=False, scaleFill=True)

    # YOLOv8 inference
    results = model(frame_letterbox)[0]

    detections = []
    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box.tolist()
        if conf > 0.3:
            detections.append([x1, y1, x2, y2, conf, int(cls)])

    # Rescale coordinates
    detections = np.array(detections)
    detections[:, 0] = (detections[:, 0] - dw) / ratio[0]
    detections[:, 1] = (detections[:, 1] - dh) / ratio[1]
    detections[:, 2] = (detections[:, 2] - dw) / ratio[0]
    detections[:, 3] = (detections[:, 3] - dh) / ratio[1]

    # Update the tracker
    res = tracker.update(detections, frame)

    # Plot tracking results
    tracker.plot_results(frame, show_trajectories=True)

    cv2.imshow('BoXMOT + YOLOv8', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()