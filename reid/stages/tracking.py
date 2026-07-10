import time
from typing import Any
from reid.stages.base import PipelineStage
from reid.utils import ReIDPipelineListener, FrameData
from reid.tracking.tracker import Tracker


class TrackingStage(PipelineStage):
    """Stage 3: Performs manual track association. Owns the Tracker.

    Registry updates happen only on track termination via the on_track_terminated hook,
    not on every frame.
    """

    def __init__(self, tracker_config: str):
        """Constructor.

        Args:
            tracker_config (str): Path to tracker configuration YAML.
        """
        self.tracker_config = tracker_config
        self.manual_tracker = None

    def initialize(self, listener: ReIDPipelineListener = None) -> None:
        if listener:
            listener.on_init_status("Loading manual Tracker and configuration...")
        self.manual_tracker = Tracker(self.tracker_config)

    def _wire_termination_hook(self, pipeline: Any) -> None:
        """Wire the on_track_terminated hook to register tracks into the pipeline registry."""

        def _on_terminated(track):
            # The Tracker augments the track with .embedding from its internal store
            embedding = getattr(track, "embedding", None)
            if embedding is None:
                return

            class_id = int(track.cls)

            # Look up the class label from YOLO detector
            from reid.stages.detection import YoloDetectionStage
            yolo_stage = next(
                (s for s in pipeline.stages if isinstance(s, YoloDetectionStage)), None
            )
            class_label = "unknown"
            if yolo_stage and yolo_stage.detector:
                class_label = yolo_stage.detector.model.names.get(class_id, "unknown")

            # Resolve feed_name from the VideoFeederStage
            from reid.stages.video_feeder import VideoFeederStage
            feeder_stage = next(
                (s for s in pipeline.stages if isinstance(s, VideoFeederStage)), None
            )
            feed_name = feeder_stage.video_name if feeder_stage else ""

            global_id, similarity = pipeline.registry.register_track(
                local_track_id=track.track_id,
                embedding=embedding,
                class_label=class_label,
                feed_name=feed_name,
            )

        self.manual_tracker.on_track_terminated = _on_terminated

    def process(self, data: FrameData, pipeline: Any) -> FrameData:
        # Wire hook on first call (lazy, after pipeline is fully set up)
        if self.manual_tracker and not getattr(self.manual_tracker, "_hook_wired", False):
            self._wire_termination_hook(pipeline)
            self.manual_tracker._hook_wired = True

        if data.skip or data.end_of_stream:
            listener = data.listener
            if listener and not data.end_of_stream:
                listener.on_frame_processed(
                    video_name=data.feed_name,
                    video_idx=data.feed_idx,
                    total_videos=data.total_videos,
                    frame_count=data.frame_count,
                    total_frames=data.total_frames,
                    elapsed_time=data.elapsed_time,
                    fps=data.fps,
                    registry=pipeline.registry,
                )
            return data

        # Run manual tracker update
        tracks = self.manual_tracker.update(
            boxes=data.boxes,
            scores=data.scores,
            classes=data.classes,
            features=data.features
        )

        listener = data.listener

        # Progress update with active track count
        if listener:
            active_ids = [int(t[4]) for t in tracks] if len(tracks) > 0 else []
            log_line = None
            if len(active_ids) > 0:
                t_str = time.strftime("%H:%M:%S")
                log_line = f"[{t_str}] Active tracks: {active_ids}"

            listener.on_frame_processed(
                video_name=data.feed_name,
                video_idx=data.feed_idx,
                total_videos=data.total_videos,
                frame_count=data.frame_count,
                total_frames=data.total_frames,
                elapsed_time=data.elapsed_time,
                fps=data.fps,
                registry=pipeline.registry,
                log_message=log_line,
            )

        return data
