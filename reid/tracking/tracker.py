import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from ultralytics.utils import IterableSimpleNamespace
from .tracker_factory import TrackerFactory


class Detections:
    """Mock detections object compatible with Ultralytics tracker requirements."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        """Initialize mock detections.

        Args:
            xywh (np.ndarray): Center x, center y, width, height format bounding boxes (N, 4).
            conf (np.ndarray): Confidence scores (N,).
            cls (np.ndarray): Class labels (N,).
        """
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.xywh)

    def __getitem__(self, idx: Any) -> "Detections":
        return Detections(self.xywh[idx], self.conf[idx], self.cls[idx])


class Tracker:
    """Manual tracking API wrapping custom and standard ByteTrack/EnhancedByteTracker algorithms.

    Maintains an internal mapping of track_id -> latest embedding so that the
    on_track_terminated hook can access the appearance vector regardless of
    which tracker backend (BYTETracker vs EnhancedByteTracker) is used.
    """

    def __init__(self, tracker_config: Union[str, Path, dict]):
        """Initialize the tracker with a YAML configuration file or dict settings.

        Args:
            tracker_config (Union[str, Path, dict]): Tracker configuration settings.
        """
        if isinstance(tracker_config, (str, Path)):
            path = str(tracker_config)
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(__file__), "config", os.path.basename(path))

            with open(path, "r") as f:
                cfg_dict = yaml.safe_load(f) or {}
        elif isinstance(tracker_config, dict):
            cfg_dict = tracker_config
        else:
            raise TypeError("tracker_config must be a path to a YAML file or a dictionary.")

        # Map to IterableSimpleNamespace as required by Ultralytics trackers
        self.args = IterableSimpleNamespace(**cfg_dict)
        self.args.device = "cpu"  # Keep tracking logic on CPU

        # Create tracker backend via Factory
        self.tracker = TrackerFactory.create(self.args)

        # Internal store: track_id -> latest appearance embedding
        self.track_embeddings: Dict[int, np.ndarray] = {}

    def on_track_terminated(self, track: Any) -> None:
        """Hook called when a track is terminated/removed.

        The track object is augmented with an `embedding` attribute containing
        the last known appearance vector from the internal store.

        Can be overridden by subclasses or set as a callback function on instance.
        """
        pass

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Manually update track states using new frame detections and features.

        Args:
            boxes (np.ndarray): Bounding boxes in xyxy format (N, 4).
            scores (np.ndarray): Confidence scores (N,).
            classes (np.ndarray): Class label indices (N,).
            features (np.ndarray, optional): Appearance embeddings (N, D).

        Returns:
            np.ndarray: Tracked targets array of shape (M, 8) where each row is:
                        [x1, y1, x2, y2, track_id, score, class_id, detection_idx]
        """
        prev_removed_ids = {t.track_id for t in getattr(self.tracker, "removed_stracks", [])}

        if len(boxes) == 0:
            # Handle frame with zero detections
            empty_detections = Detections(
                xywh=np.empty((0, 4), dtype=np.float32),
                conf=np.empty((0,), dtype=np.float32),
                cls=np.empty((0,), dtype=np.int32)
            )
            tracks = self.tracker.update(empty_detections, feats=np.empty((0, 0)))
        else:
            # Convert detections from xyxy to center x, center y, width, height (xywh)
            xywh = np.empty_like(boxes, dtype=np.float32)
            xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0  # Center x
            xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0  # Center y
            xywh[:, 2] = boxes[:, 2] - boxes[:, 0]          # Width
            xywh[:, 3] = boxes[:, 3] - boxes[:, 1]          # Height

            # Wrap into mock detection object
            results = Detections(xywh=xywh, conf=scores, cls=classes)

            # Call tracker update method, forwarding optional pre-extracted features
            tracks = self.tracker.update(results, feats=features)

        # Update the internal track_id -> embedding mapping from active tracks
        if features is not None and len(tracks) > 0:
            for t in tracks:
                track_id = int(t[4])
                det_idx = int(t[7])
                if det_idx < len(features):
                    self.track_embeddings[track_id] = features[det_idx].copy()

        # Trigger hook for any newly terminated tracks
        newly_removed = [
            t for t in getattr(self.tracker, "removed_stracks", [])
            if t.track_id not in prev_removed_ids
        ]
        for track in newly_removed:
            # Augment the track object with the stored embedding
            track.embedding = self.track_embeddings.pop(track.track_id, None)
            self.on_track_terminated(track)

        return tracks

    def reset(self) -> None:
        """Reset the tracker state and IDs."""
        self.tracker.reset()
        self.track_embeddings.clear()
