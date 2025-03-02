# config.py

# 1. Path to your Firebase Service Account JSON file (downloaded from Firebase Console)
SERVICE_ACCOUNT_PATH = "C:\Users\gabri\School\Senior\cattle-notification-firebase-adminsdk-fbsvc-2e120248b2.json"

# 2. Path to your YOLOv8 model (trained on your dataset)
MODEL_PATH = "yolov8x.pt" # or "yolov8n.pt" for faster inference
# 3. Path to your dataset YAML file (contains class names and paths)
DATASET_YAML_PATH = "C:\Users\gabri\School\Senior\data.yaml"


# 3. Path to your video source (can be a video file or a camera stream)
# VIDEO_SOURCE = "https://"

# 4. Device Token from your Android app (retrieved via Logcat)
DEVICE_TOKEN = "your_android_device_token_here"