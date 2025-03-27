from ultralytics import YOLO
import cv2
import torch
import numpy as np
from boxmot import BotSort
from boxmot.utils.ops import letterbox
from pathlib import Path
import firebase_admin
from firebase_admin import credentials

cred = credentials.Certificate("C:\\Users\\gabri\\School\\Senior\\Chris_T_Firebase.json")
firebase_admin.initialize_app(cred)

def count_cows_and_detect_intruders():
    # Load YOLOv8 model
    device = torch.device('cuda:0')  # or 'cuda'
    model = YOLO('runs/detect/train32/weights/best.pt')  # Load the trained model
    #model = YOLO('yolov8x.pt')  # Load the trained model

    # Initialize the tracker
    tracker = BotSort(
        reid_weights=Path('osnet_x0_25_msmt17.pt'),  # Path to ReID model
        device=device,
        half=False
    )

    input_size = 640  # YOLOv8's default input size

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow instead of MSMF

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_letterbox, ratio, (dw, dh) = letterbox(frame, new_shape=input_size, auto=False, scaleFill=True)

        # YOLOv8 inference
        results = model(frame_letterbox)[0]

        detections = []
        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            if conf > 0.8:  # Use the same confidence threshold as before
                detections.append([x1, y1, x2, y2, conf, int(cls)])

        # Rescale coordinates
        detections = np.array(detections)
        if detections.size > 0:  # added this check
            detections[:, 0] = (detections[:, 0] - dw) / ratio[0]
            detections[:, 1] = (detections[:, 1] - dh) / ratio[1]
            detections[:, 2] = (detections[:, 2] - dw) / ratio[0]
            detections[:, 3] = (detections[:, 3] - dh) / ratio[1]

        # Update the tracker
        res = tracker.update(detections, frame)

        cow_count = 0
        intruder_detected = False

        # Plot tracking results
        for i, track in enumerate(res):
            x1, y1, x2, y2, track_id, _, conf, *extra = track
            class_id_int = int(detections[i][5])
            if 0 <= class_id_int < len(results.names):
                label = results.names[class_id_int]
            else:
                print(f"Warning: class_id {class_id_int} out of range.")
                label = "Unknown"

            if label ==  "cow" :  #"Black_Cow_1" or label == "Spotted_Cow_1" or label == "Brown_Cow_1" or label == "Spotted_Cow_2" or label == "Spotted_Cow_2 " or label == "Spotted_Calve_1" or label == "Spotted_Calve_2" or label == "Friendly_Farmer" or label == "Brown_Cow_2" or label == "Spotted_Cow_3" or label =="Baby_Calve":
                cow_count += 1
                color = (255, 0, 0)  # Blue for cows
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{label} {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if label == "person":
                intruder_detected = True
                color = (0, 0, 255)  # Red for intruders
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{label} {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
   #         else:
 #                   color = (10, 10, 50)  # Green for other objects.
  #                  cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
 #                   cv2.putText(frame, f"Unknown {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
        # Display count
        cv2.putText(frame, f"Cows: {cow_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Detection", frame)

        # Send alert if intruder detected
        if intruder_detected:
            print("Intruder detected!")

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    count_cows_and_detect_intruders()