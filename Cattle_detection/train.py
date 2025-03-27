import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8x.pt')  # Replace with your model

# Define training parameters
epochs = 50  # Total number of epochs
data = 'data.yaml'  # Path to your dataset configuration

# Start timing
start_time = time.time()

# Train the model
model.train(data=data, epochs=1)  # Train for one epoch to measure time

# Calculate time for one epoch
time_per_epoch = time.time() - start_time
estimated_total_time = time_per_epoch * epochs

print(f"Time per epoch: {time_per_epoch:.2f} seconds")
print(f"Estimated total training time: {estimated_total_time / 60:.2f} minutes")
