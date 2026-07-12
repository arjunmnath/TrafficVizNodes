import time
from typing import Any, Optional
from reid.stages.base import PipelineStage
from reid.utils import ReIDPipelineListener, FrameData
from reid.tracking.tracker import Tracker


class TrackingStage(PipelineStage):
    """Performs manual track association. Owns the Tracker.

    When a track is terminated the ``on_track_terminated`` hook fires.
    If a ``postprocessing_pipeline`` is provided it is executed immediately
    on that track before any other downstream logic.
    """

    def __init__(
        self,
        tracker_config: str,
        postprocessing_pipeline: Optional[Any] = None,
    ):
        """Constructor.

        Args:
            tracker_config (str): Path to tracker configuration YAML.
            postprocessing_pipeline: Optional PostProcessingPipeline to run on
                each terminated track. If None, no postprocessing is performed.
        """
        self.tracker_config: str = tracker_config
        self.manual_tracker: Tracker | None = None
        self.postprocessing_pipeline: Optional[Any] = postprocessing_pipeline

    def initialize(self, listener: ReIDPipelineListener | None = None) -> None:
        if listener:
            listener.on_init_status("Loading manual Tracker and configuration...")
        self.manual_tracker = Tracker(self.tracker_config)

    def _wire_termination_hook(self, pipeline: Any) -> None:
        """Wire the on_track_terminated hook to run the postprocessing pipeline."""
        from reid.postprocessing.pipeline import TerminatedTrack

        postprocessing_pipeline = self.postprocessing_pipeline

        def _on_terminated(track: Any) -> None:
            # Resolve class label from track history if available
            class_label = "unknown"
            feed_name = ""

            # Pull occurrence_embeddings from the registry for this track
            occurrence_embeddings = None
            if hasattr(pipeline, "registry") and pipeline.registry is not None:
                entry = pipeline.registry.identities.get(track.track_id)
                if entry is not None:
                    occ_list = entry.get("occurrence_embeddings", [])
                    if occ_list:
                        import numpy as np

                        occurrence_embeddings = np.array(occ_list, dtype=np.float32)
                    # Pull feed_name and class_label from last occurrence record
                    occs = entry.get("occurrences", [])
                    if occs:
                        feed_name = occs[-1].get("feed_name", "")
                        class_label = occs[-1].get("class_label", "unknown")

            terminated = TerminatedTrack(
                track_id=track.track_id,
                class_label=class_label,
                feed_name=feed_name,
                occurrence_embeddings=occurrence_embeddings,
                smooth_embedding=getattr(track, "embedding", None),
                history=getattr(track, "history", None),
            )

            if postprocessing_pipeline is not None:
                terminated = postprocessing_pipeline.run(terminated)

            # Store the postprocessed track back so downstream stages can read it
            track.postprocessed = terminated

        self.manual_tracker.on_track_terminated = _on_terminated

    def process(self, data: FrameData, pipeline: Any) -> FrameData:
        assert self.manual_tracker is not None, "tracker not initialized."

        # Wire hook on first call (lazy, after pipeline is fully set up)
        if not self.manual_tracker.hook_wired:
            self._wire_termination_hook(pipeline)
            self.manual_tracker.set_hook_wired(True)

        # Calculate dynamic processing speed (frames per second) instead of static video frame rate
        processing_fps = data.frame_count / data.elapsed_time if data.elapsed_time > 0.0 else 0.0

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
                    fps=processing_fps,
                    registry=pipeline.registry,
                )
            return data

        assert data.boxes is not None and data.scores is not None and data.classes is not None, (
            "boxes, scores, and classes must not be None"
        )
        # Run manual tracker update
        tracks = self.manual_tracker.update(
            boxes=data.boxes,
            scores=data.scores,
            classes=data.classes,
            features=data.features,
            frame_count=data.frame_count,
            timestamp=data.timestamp,
        )
        data.tracks = tracks

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
                fps=processing_fps,
                registry=pipeline.registry,
                log_message=log_line,
            )

        return data

    def finalize(self, pipeline: Any) -> None:
        if self.manual_tracker:
            self.manual_tracker.terminate_all_tracks()
            self.manual_tracker.reset()
