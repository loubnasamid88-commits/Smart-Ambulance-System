from ultralytics import YOLO
import cv2
import time

class YOLOEngine:
    def __init__(self, model_path="yolov8n.pt", conf=0.5):
        self.model = YOLO(model_path)
        self.conf = conf

    def predict(self, frame):
        results = self.model(frame, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = self.model.names[cls]

            if conf >= self.conf:
                detections.append({
                    "label": label,
                    "confidence": conf
                })

        return detections