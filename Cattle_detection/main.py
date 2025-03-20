# main.py
from ultralytics import YOLO
import cv2
from notifications import send_fcm_alert
from config import DEVICE_TOKEN, SERVICE_ACCOUNT_PATH
import firebase_admin
from firebase_admin import credentials

cred = credentials.Certificate("C:\\Users\\gabri\\School\\Senior\\Chris_T_Firebase.json")
firebase_admin.initialize_app(cred)

def count_cows_and_detect_intruders():
    #model = YOLO('runs/detect/train9/weights/best.pt') # Load the trained model
    model = YOLO('yolov8x.pt') # Load the trained model
    #cap = cv2.VideoCapture("http://192.168.164.109:8080/video")#r video source 
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow instead of MSMF
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
            #if label == "Black_Cow_1" or label == "Spotted_Cow_1":
            if label == "cow":
                cow_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            if label == "person":  # Detect human as intruder
                intruder_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Display count
        cv2.putText(frame, f"Cows: {cow_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Detection", frame)

        # Send alert if intruder detected
        if intruder_detected:
            print("Intruder detected!")
            ''' Uncomment the following line to send an alert
                Note: Make sure you have set up Firebase and obtained the DEVICE_TOKEN
                from your Android app. '''
            send_fcm_alert(
                "Intruder detected!",
                DEVICE_TOKEN,
                SERVICE_ACCOUNT_PATH
            )
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    count_cows_and_detect_intruders()