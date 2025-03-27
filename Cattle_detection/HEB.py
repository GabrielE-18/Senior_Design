from ultralytics import YOLO
import cv2
import torch
import numpy as np
from boxmot import BotSort
from boxmot.utils.ops import letterbox
from pathlib import Path
import firebase_admin
from firebase_admin import credentials
from collections import defaultdict

# ================================
# ✅ Firebase Initialization
# ================================
cred = credentials.Certificate("C:\\Users\\gabri\\School\\Senior\\Chris_T_Firebase.json")
firebase_admin.initialize_app(cred)

# ================================
# ✅ Model and Tracker Setup
# ================================
device = torch.device('cuda:0')
model = YOLO('runs/detect/train29/weights/best.pt')  # Your trained model path

# BotSort Tracker with ReID
tracker = BotSort(
    reid_weights=Path('osnet_x0_25_msmt17.pt'),  
    device=device,
    half=False
)

# YOLO input size
input_size = 640  

# ================================
# ✅ Cow Tracking Data   Talk to group 14 over for better training 
# ================================
known_cows = {}   # {cow_id: {"features": [feature1, feature2, ...], "last_seen": frame_count}}
cow_type_count = defaultdict(int)  # To count each cow type separately
frame_count = 0

# ================================
# ✅ Feature Extraction Function
# ================================
# ✅ Updated feature extraction
def extract_features(frame, x1, y1, x2, y2):
    """Extracts color, size, and position features as a list of numerical values."""
    roi = frame[int(y1):int(y2), int(x1):int(x2)]

    # Average color
    avg_color = np.mean(roi, axis=(0, 1))

    # Size (area)
    width = x2 - x1
    height = y2 - y1
    size = width * height

    # Position (center coordinates)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    position = (center_x, center_y)

    # Return features as a list of numerical values
    return [
        avg_color[0], avg_color[1], avg_color[2],  # RGB values
        size,
        position[0], position[1]                   # X and Y positions
    ]


# ✅ Updated cow matching function

def calculate_similarity(features1, features2):
    """Calculates the Euclidean distance between two feature vectors."""
    return np.linalg.norm(np.array(features1) - np.array(features2))
def match_cow(features, threshold=150):
    """Finds the best matching cow based on visual similarity."""
    best_match = None
    best_score = float('inf')

    for cow_id, data in known_cows.items():
        if len(data["features"]) > 0:
            feature_array = np.array(data["features"])
            
            # Calculate the average features
            avg_features = np.mean(feature_array, axis=0)
        else:
            # Fallback to the input features if no existing features
            avg_features = np.array(features)

        # Calculate similarity score
        score = calculate_similarity(features, avg_features)

        if score < best_score and score < threshold:
            best_score = score
            best_match = cow_id

    return best_match



# ✅ Fixed cow dictionary storage
def count_cows_and_detect_intruders():
    global frame_count

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use webcam or video path

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Preprocessing for YOLO inference
        frame_letterbox, ratio, (dw, dh) = letterbox(frame, new_shape=input_size, auto=False, scaleFill=True)

        # YOLO Inference
        results = model(frame_letterbox)[0]

        # Prepare detections for BotSort
        detections = []
        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            if conf > 0.85:  # Confidence threshold
                detections.append([x1, y1, x2, y2, conf, int(cls)])

        # Rescale coordinates
        detections = np.array(detections)
        if detections.size > 0:
            detections[:, 0] = (detections[:, 0] - dw) / ratio[0]
            detections[:, 1] = (detections[:, 1] - dh) / ratio[1]
            detections[:, 2] = (detections[:, 2] - dw) / ratio[0]
            detections[:, 3] = (detections[:, 3] - dh) / ratio[1]

        # Update the tracker
        res = tracker.update(detections, frame)

        # Initialize counters
        intruder_detected = False

        # Plot and track results
        for i, track in enumerate(res):
            x1, y1, x2, y2, track_id, _, conf, *extra = track
            class_id_int = int(detections[i][5])

            if 0 <= class_id_int < len(results.names):
                label = results.names[class_id_int]
            else:
                print(f"Warning: class_id {class_id_int} out of range.")
                label = "Unknown"

            # Extract visual features (now numerical values)
            features = extract_features(frame, x1, y1, x2, y2)

            # Re-ID logic
            matched_cow_id = match_cow(features)

            if matched_cow_id:
                cow_id = matched_cow_id
                known_cows[cow_id]["features"].append(features)
                known_cows[cow_id]["last_seen"] = frame_count
            else:
                cow_id = f"Cow_{len(known_cows) + 1}"
                known_cows[cow_id] = {"features": [features], "last_seen": frame_count}

            # Count cows by type
            if label == "cow":
                cow_type_count[label] += 1

                color = (255, 0, 0)  # Blue for cows
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{label} {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            elif label == "person":
                intruder_detected = True
                color = (0, 0, 255)  # Red for intruders
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, "INTRUDER", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ================================
        # ✅ Display Counts
        # ================================
        y_offset = 30
        for cow_type, count in cow_type_count.items():
            cv2.putText(frame, f"{cow_type}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 25

        cv2.imshow("Detection", frame)

        # Send Firebase alert
        if intruder_detected:
            print("Intruder detected!")

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ================================
# ✅ Main Execution
# ================================
if __name__ == "__main__":
    count_cows_and_detect_intruders()
