import os
from typing import Any
import cv2
from reid.stages.base import PipelineStage
from reid.utils import ReIDPipelineListener, FrameData


class VideoFeederStage(PipelineStage):
    """Stage 0: Minimal synchronous video frame feeder stage."""

    def __init__(self, video_path: str = ""):
        """Constructor.

        Args:
            video_path (str): Initial video file path or RTSP stream link.
        """
        self.video_path = video_path
        self.cap = None
        self.fps = 30.0
        self.total_frames = 0
        self.video_name = ""
        self.frame_count = 0

    def set_video_path(self, video_path: str) -> None:
        """Update the video path for the next stream ingestion.

        Args:
            video_path (str): Video file or RTSP link.
        """
        self.video_path = video_path

    def initialize(self, listener: ReIDPipelineListener = None) -> None:
        """Open the video stream and reset the frame counter."""
        if not self.video_path:
            return

        if listener:
            listener.on_init_status(f"Initializing VideoFeeder for {self.video_path}...")

        # If a capture object is already open, release it
        if self.cap and self.cap.isOpened():
            self.cap.release()

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_name = os.path.basename(self.video_path)
        self.frame_count = 0

    def process(self, data: FrameData, pipeline: Any) -> FrameData:
        """Fetch the next frame synchronously from cv2.VideoCapture."""
        if self.cap is None or not self.cap.isOpened():
            data.skip = True
            data.end_of_stream = True
            return data

        ret, frame = self.cap.read()
        if not ret:
            data.skip = True
            data.end_of_stream = True
            return data

        self.frame_count += 1
        timestamp = self.frame_count / self.fps

        # Populate FrameData properties
        data.frame = frame
        data.frame_count = self.frame_count
        data.feed_name = self.video_name
        data.total_frames = self.total_frames
        data.timestamp = timestamp
        data.fps = self.fps

        data.skip = False
        data.end_of_stream = False
        return data

    def stop(self) -> None:
        """Release the VideoCapture resources."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
