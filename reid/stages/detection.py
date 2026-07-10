from typing import Any
from reid.stages.base import PipelineStage
from reid.utils import ReIDPipelineListener, FrameData
from reid.tracking.detector import Detector


class YoloDetectionStage(PipelineStage):
    """Stage 1: Performs YOLO detection on the input frame. Owns the Detector model instance."""

    def __init__(self, yolo_path: str):
        """Constructor.

        Args:
            yolo_path (str): Path to YOLO detector weight file (.pt).
        """
        self.yolo_path = yolo_path
        self.detector = None

    def initialize(self, listener: ReIDPipelineListener = None) -> None:
        if listener:
            listener.on_init_status("Loading YOLOv8 Detector model weights...")
        self.detector = Detector(self.yolo_path)

    def process(self, data: FrameData, pipeline: Any) -> FrameData:
        if data.skip or data.end_of_stream:
            return data

        frame = data.frame
        # Run detection. Target COCO classes: person(0), car(2), motorcycle(3), bus(5), truck(7).
        dets = self.detector.detect(frame, conf=0.25, classes=[0, 2, 3, 5, 7])

        data.boxes = dets["boxes"]
        data.scores = dets["scores"]
        data.classes = dets["classes"]
        return data
