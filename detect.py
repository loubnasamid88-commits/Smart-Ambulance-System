import requests
from ultralytics import YOLO
import numpy as np
import cv2

URL_ACCIDENT = "http://127.0.0.1:5000/vision_accident"
URL_CONGESTION = "http://127.0.0.1:5000/vision_congestion"

model = YOLO("yolov8n.pt")

accident_sent = False
congestion_sent = False

while True:

    # ==============================
    # 1. RÉCUPÉRER IMAGE DE SUMO
    # ==============================
    try:
        response = requests.get("http://127.0.0.1:5000/sumo_frame")
        frame_array = np.frombuffer(response.content, np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

    except:
        continue

    # ==============================
    # 2. YOLO SUR FRAME SUMO
    # ==============================
    results = model(frame)

    vehicle_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]

            if name in ["car", "truck", "bus", "motorbike"]:
                vehicle_count += 1

    # ==============================
    # 3. CONGESTION
    # ==============================
    if vehicle_count > 8 and not congestion_sent:
        print("🚦 Congestion SUMO détectée")
        requests.post(URL_CONGESTION)
        congestion_sent = True

    # ==============================
    # 4. ACCIDENT
    # ==============================
    if vehicle_count < 2 and not accident_sent:
        print("🚑 Accident SUMO détecté")
        requests.post(URL_ACCIDENT)
        accident_sent = True

    # debug
    cv2.imshow("YOLO + SUMO", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()