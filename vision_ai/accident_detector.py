import time
import numpy as np

class AccidentDetector:
    def __init__(self):
        self.vehicle_memory = {}  # tracking simple

    def detect_accident(self, detections):
        """
        Détection intelligente :
        - densité élevée
        - immobilité prolongée (approximation)
        - présence multi-classes
        """

        vehicles = [
            d for d in detections
            if d["label"] in ["car", "truck", "bus"]
        ]

        pedestrians = [
            d for d in detections
            if d["label"] == "person"
        ]

        vehicle_count = len(vehicles)
        person_count = len(pedestrians)

        # 🔥 1. CAS CONGESTION FORTE (possible accident)
        if vehicle_count >= 6 and person_count >= 3:
            confidence = min(0.95, (vehicle_count + person_count) / 15)

            return {
                "event": "accident",
                "confidence": float(confidence)
            }

        # 🔥 2. CAS URBAIN DENSE CRITIQUE
        if vehicle_count >= 10:
            return {
                "event": "accident",
                "confidence": 0.80
            }

        # 🔥 3. CAS SUSPECT (option PFE bonus)
        if vehicle_count >= 5 and person_count >= 5:
            return {
                "event": "accident",
                "confidence": 0.75
            }

        return None