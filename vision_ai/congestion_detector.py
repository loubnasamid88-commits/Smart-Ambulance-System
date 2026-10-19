class CongestionDetector:
    def detect(self, detections):
        vehicles = [d for d in detections if d["label"] in ["car", "bus", "truck"]]

        if len(vehicles) > 12:
            return {
                "event": "congestion",
                "level": "high",
                "confidence": 0.9
            }

        return None