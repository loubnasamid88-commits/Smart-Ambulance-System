from ultralytics import YOLO

# Charger modèle pré-entraîné
model = YOLO("yolov8s.pt")

# Entraînement
model.train(
    data="dataset/data/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device=0
)