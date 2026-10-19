import threading
import time
import numpy as np
import cv2
import mss

from ultralytics import YOLO


class VisionWorker:
    def __init__(self, model_path, callback=None, backend_url=None):
        """
        YOLO Vision Worker
        Compatible avec :
        - Flask
        - SUMO
        - TraCI
        """

        # =========================
        # CONFIG
        # =========================
        self.model_path = model_path
        self.callback = callback
        self.backend_url = backend_url

        # =========================
        # YOLO MODEL
        # =========================
        self.model = YOLO(model_path)

        print("📦 Classes YOLO :", self.model.names)

        # =========================
        # THREAD SYSTEM
        # =========================
        self.running = False
        self.thread = None

        # =========================
        # SCREEN CAPTURE
        # =========================
        self.sct = mss.mss()

        # écran principal
        self.monitor = self.sct.monitors[1]

        # =========================
        # TRACI FLAG
        # =========================
        self.traci_ready = False

        # =========================
        # ANTI-SPAM
        # =========================
        self.last_event_time = 0
        self.cooldown = 3  # secondes

        print("🚀 VisionWorker initialisé avec succès")

    # ==================================================
    # SUMO / TRACI READY
    # ==================================================
    def set_traci_ready(self, value: bool):
        """
        Appelé depuis app.py quand SUMO est prêt
        """
        self.traci_ready = value
        print(f"🔗 TraCI ready = {value}")

    # ==================================================
    # START
    # ==================================================
    def start(self):
        if self.running:
            return

        self.running = True

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True
        )

        self.thread.start()

        print("👁 YOLO Vision loop STARTED")

    # ==================================================
    # STOP
    # ==================================================
    def stop(self):
        self.running = False
        print("🛑 YOLO Vision STOPPED")

    # ==================================================
    # MAIN LOOP
    # ==================================================
    def _loop(self):
        print("📷 YOLO SCREEN CAPTURE LOOP")

        while self.running:

            try:
                frame = self._capture_screen()

                if frame is None:
                    continue

                # inference YOLO
                results = self.model(frame, verbose=False)

                # analyse
                self._process_results(results)

                # ~20 FPS
                time.sleep(0.05)

            except Exception as e:
                print(f"❌ Erreur YOLO LOOP : {e}")
                time.sleep(1)

    # ==================================================
    # SCREEN CAPTURE
    # ==================================================
    def _capture_screen(self):

        try:
            img = self.sct.grab(self.monitor)

            frame = np.array(img)

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGRA2BGR
            )

            return frame

        except Exception as e:
            print(f"❌ Erreur capture écran : {e}")
            return None

    # ==================================================
    # YOLO PROCESSING
    # ==================================================
    def _process_results(self, results):

        if not results:
            return

        r = results[0]

        if r.boxes is None:
            return

        for box in r.boxes:

            try:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                label = self.model.names.get(
                    cls_id,
                    str(cls_id)
                )

                # debug
                print(f"🎯 DETECTION: {label} ({conf:.2f})")

                label_lower = label.lower()

                # =========================
                # TES CLASSES DATASET
                # =========================
                is_accident = (
                    "accident" in label_lower
                    or "flip" in label_lower
                    or "collision" in label_lower
                    or "crash" in label_lower
                )

                if is_accident and conf > 0.5:
                    self._trigger_event(conf, box)

            except Exception as e:
                print(f"❌ Erreur analyse box YOLO : {e}")

    # ==================================================
    # EVENT SYSTEM
    # ==================================================
    def _trigger_event(self, conf, box):

        now = time.time()

        # anti-spam
        if now - self.last_event_time < self.cooldown:
            return

        self.last_event_time = now

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        event = {
            "type": "ACCIDENT_YOLO",
            "confidence": float(conf),
            "bbox": [x1, y1, x2, y2],
            "timestamp": now
        }

        print(
            f"💥 YOLO ACCIDENT DETECTÉ "
            f"(conf={conf:.2f})"
        )

        # callback Flask/backend
        if self.callback:
            try:
                self.callback(event)

            except Exception as e:
                print(f"❌ Callback erreur : {e}")

    # ==================================================
    # DEBUG VIEW
    # ==================================================
    def debug_view(self):

        try:
            frame = self._capture_screen()

            if frame is None:
                return

            results = self.model(
                frame,
                verbose=False
            )

            annotated = results[0].plot()

            cv2.imshow(
                "YOLO SUMO VIEW",
                annotated
            )

            cv2.waitKey(1)

        except Exception as e:
            print(f"❌ Debug view erreur : {e}")