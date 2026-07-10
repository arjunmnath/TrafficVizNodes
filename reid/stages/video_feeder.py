import os
import time
import cv2
import queue
import threading
from typing import Any
from reid.stages.base import PipelineStage
from reid.utils import ReIDPipelineListener, FrameData


class VideoFeederStage(PipelineStage):
    """Stage 0: Real-time asynchronous video frame feeder stage.

    Reads frames from a video file or stream in a background thread and pushes them
    to a bounded queue to prevent input lag in real-time systems.
    """

    def __init__(self, video_path: str = "", max_queue_size: int = 2):
        """Constructor.

        Args:
            video_path (str): Initial video file path or RTSP stream link.
            max_queue_size (int): Max capacity of the thread-safe queue.
        """
        self.video_path = video_path
        self.max_queue_size = max_queue_size
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.thread = None
        self.cap = None
        self.fps = 30.0
        self.total_frames = 0
        self.video_name = ""

    def set_video_path(self, video_path: str) -> None:
        """Update the video path for the next stream ingestion.

        Args:
            video_path (str): Video file or RTSP link.
        """
        self.video_path = video_path

    def initialize(self, listener: ReIDPipelineListener = None) -> None:
        """Open the video stream and start the asynchronous reader thread."""
        if not self.video_path:
            return

        if listener:
            listener.on_init_status(f"Initializing VideoFeeder for {self.video_path}...")

        # Clear queue from any leftover frame data
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_name = os.path.basename(self.video_path)

        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self) -> None:
        """Continuous ingestion loop running in a background thread."""
        frame_count = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                # Signal end of stream
                self.frame_queue.put(None)
                break

            frame_count += 1
            timestamp = frame_count / self.fps

            # Package frame data context
            frame_data = {
                "frame": frame,
                "frame_count": frame_count,
                "video_name": self.video_name,
                "total_frames": self.total_frames,
                "timestamp": timestamp,
                "fps": self.fps,
            }

            # Put frame to queue. If queue is full, discard the oldest frame
            # to preserve real-time low latency.
            while self.running:
                try:
                    self.frame_queue.put(frame_data, timeout=0.1)
                    break
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass

        if self.cap:
            self.cap.release()

    def process(self, data: FrameData, pipeline: Any) -> FrameData:
        """Fetch the latest frame context from the queue.

        Blocks until a frame is available or stream ends.
        """
        try:
            # Block until frame is retrieved or timeout
            frame_data = self.frame_queue.get(timeout=5.0)
        except queue.Empty:
            data.skip = True
            data.end_of_stream = True
            return data

        if frame_data is None:
            data.skip = True
            data.end_of_stream = True
            return data

        # Populate FrameData properties
        data.frame = frame_data["frame"]
        data.frame_count = frame_data["frame_count"]
        data.feed_name = frame_data["video_name"]
        data.total_frames = frame_data["total_frames"]
        data.timestamp = frame_data["timestamp"]
        data.fps = frame_data["fps"]

        data.skip = False
        data.end_of_stream = False
        return data

    def stop(self) -> None:
        """Cleanly terminate the background thread and release resources."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        if self.cap and self.cap.isOpened():
            self.cap.release()
