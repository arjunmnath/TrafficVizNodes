from reid.stages.base import PipelineStage
from reid.stages.tracking import TrackingStage
from reid.utils import FrameData
from typing import Any
import numpy as np


class OfflineAddToRegistryStage(PipelineStage):
    """Offline registry stage that registers active tracks frame-by-frame.

    For each active track, passes both:
      - The smoothed appearance embedding maintained by the tracker (moving average).
      - The raw per-frame detection embedding from FrameData.features (occurrence embedding).
    """

    def process(self, data: FrameData, pipeline: Any) -> FrameData:
        if not hasattr(pipeline, "registry") or pipeline.registry is None:
            return data

        if data.skip or data.end_of_stream:
            return data

        if data.tracks is None or len(data.tracks) == 0:
            return data

        # Access TrackingStage to retrieve smoothed embeddings
        tracking_stage = next((s for s in pipeline.stages if isinstance(s, TrackingStage)), None)
        if not tracking_stage or not tracking_stage.manual_tracker:
            return data

        # Resolve feed name from the feeder stage
        from reid.stages.video_feeder import VideoFeederStage
        from reid.stages.live_feeder import LiveFootageFeedStage

        feeder_stage = next(
            (s for s in pipeline.stages if isinstance(s, (VideoFeederStage, LiveFootageFeedStage))),
            None,
        )
        feed_name = (
            feeder_stage.video_name if feeder_stage and hasattr(feeder_stage, "video_name") else ""
        )

        # Resolve class label lookup from detector stage
        from reid.stages.detection import YoloDetectionStage

        yolo_stage = next((s for s in pipeline.stages if isinstance(s, YoloDetectionStage)), None)

        # Fallback feature dimension
        feat_dim = (
            data.features.shape[1] if data.features is not None and len(data.features) > 0 else 2048
        )

        for t in data.tracks:
            # track layout: [x1, y1, x2, y2, track_id, score, class_id, detection_idx]
            bbox = t[0:4].tolist()
            track_id = int(t[4])
            class_id = int(t[6])
            det_idx = int(t[7])

            class_label = "unknown"
            if yolo_stage and yolo_stage.detector:
                class_label = yolo_stage.detector.model.names.get(class_id, "unknown")

            # Smoothed embedding from the tracker's moving average store
            smooth_embedding = tracking_stage.manual_tracker.track_embeddings.get(track_id)
            if smooth_embedding is None:
                smooth_embedding = np.zeros(feat_dim, dtype=np.float32)

            # Raw occurrence embedding from FrameData.features for this specific detection
            if data.features is not None and det_idx < len(data.features):
                occurrence_embedding = data.features[det_idx].copy()
            else:
                occurrence_embedding = np.zeros(feat_dim, dtype=np.float32)

            pipeline.registry.update_track(
                local_track_id=track_id,
                smooth_embedding=smooth_embedding,
                occurrence_embedding=occurrence_embedding,
                class_label=class_label,
                feed_name=feed_name,
                frame_number=data.frame_count,
                timestamp=data.timestamp,
                bbox=bbox,
            )

        return data
