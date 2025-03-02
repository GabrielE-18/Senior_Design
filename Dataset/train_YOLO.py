from ultralytics import YOLO

# Load a pre-trained model (better starting point)
model = YOLO('yolov8x.pt')  # Use 'yolov8n.pt' for faster training

# Train the model
results = model.train(
    data='Dataset/data.yaml',
    epochs=100,             # Increase if underfitting
    batch=16,               # Adjust based on GPU RAM
    imgsz=640,              # Match your video resolution
    device='0',             # Use GPU (e.g., '0' for CUDA)
    name='cow_detector_v1'
)