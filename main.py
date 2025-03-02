# main.py
from ultralytics import YOLO
import cv2
from notifications import send_fcm_alert
from config import DEVICE_TOKEN, SERVICE_ACCOUNT_PATH

def count_cows_and_detect_intruders():
    model = YOLO('yolov8x.pt') # Load the trained model
    #cap = cv2.VideoCapture("http://192.168.164.109:8080/video")#r video source 
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use DirectShow instead of MSMF
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.5)
        boxes = results[0].boxes
        names = results[0].names

        cow_count = 0
        intruder_detected = False

        for box in boxes:
            class_id = int(box.cls)
            label = names[class_id]
            if label == "cow":
                cow_count += 1
            else:
                intruder_detected = True

        # Display count
        cv2.putText(frame, f"Cows: {cow_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Detection", frame)

        # Send alert if intruder detected
        if intruder_detected:
            send_fcm_alert(
                "Intruder detected!",
                DEVICE_TOKEN
            )

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    count_cows_and_detect_intruders()