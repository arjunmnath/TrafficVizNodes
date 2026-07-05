from ultralytics import YOLO
import numpy as np


class YOLOTracker:
    def __init__(self, model_path: str = "yolov8s.pt", conf: float = 0.5):
        self.model = YOLO(model_path)
        self.conf = conf
        # COCO classes: 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.classes = [0, 2, 3, 5, 7]

    def track(self, frame: np.ndarray):
        # We use persist=True to keep tracks across frames
        results = self.model.track(
            frame,
            classes=self.classes,
            conf=self.conf,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        return results[0] if len(results) > 0 else None
