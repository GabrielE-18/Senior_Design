from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load your YOLOv8 model
model = YOLO("yolov8x.pt")  # Replace with your model path


@app.get("/predict/health")
def health_check():
    return {"status": "ok"}

# Setup endpoint (for initializing or configuring the system/model)
@app.get("/predict/setup")
def setup():
    # Example: Return a setup status message
    return {"status": "model setup complete"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    results = model(image)

    detections = []
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            detections.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": conf,
                "class": int(cls)
            })

    return {"detections": detections}
