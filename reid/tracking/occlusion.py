import numpy as np
from typing import Any, Dict, List, Optional


def bbox_iou(box1: np.ndarray[Any, Any], box2: np.ndarray[Any, Any]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes in xyxy format.

    Args:
        box1 (np.ndarray): First box [x1, y1, x2, y2].
        box2 (np.ndarray): Second box [x1, y1, x2, y2].

    Returns:
        float: IoU value between 0.0 and 1.0.
    """
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    xi1 = max(x11, x21)
    yi1 = max(y11, y21)
    xi2 = min(x12, x22)
    yi2 = min(y12, y22)

    inter_area = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)

    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)

    union_area = box1_area + box2_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


class OcclusionManager:
    """Manager responsible for caching lost tracks and querying candidate tracks for recall."""

    def __init__(
        self,
        timeout: int = 30,
        similarity_threshold: float = 0.5,
        spatial_threshold: Optional[float] = None,
    ):
        """Initialize the occlusion manager.

        Args:
            timeout (int): Maximum disappearance duration in frames before a track is evicted.
            similarity_threshold (float): Minimum appearance cosine similarity required for association.
            spatial_threshold (float, optional): Optional spatial gating IoU threshold.
        """
        self.timeout = timeout
        self.similarity_threshold = similarity_threshold
        self.spatial_threshold = spatial_threshold
        self.lost_tracks: Dict[int, Any] = {}

    def add_lost_track(self, track: Any) -> None:
        """Add a lost track to the temporary occlusion cache.

        Args:
            track (Any): The track object that has transitioned to Lost.
        """
        self.lost_tracks[track.track_id] = track

    def remove(self, track_id: int) -> None:
        """Remove a track from the occlusion cache.

        Args:
            track_id (int): ID of the track to remove.
        """
        self.lost_tracks.pop(track_id, None)

    def cleanup(self, current_frame: int) -> None:
        """Evict lost tracks that have exceeded the timeout.

        Args:
            current_frame (int): The current frame ID.
        """
        expired_ids = [
            tid
            for tid, track in self.lost_tracks.items()
            if current_frame - track.end_frame > self.timeout
        ]
        for tid in expired_ids:
            self.lost_tracks.pop(tid, None)

    def query(self, track: Any, frame_id: int) -> List[Any]:
        """Query candidate lost tracks that match the query detection track.

        Args:
            track (Any): The new unmatched detection track.
            frame_id (int): ID of the current frame.

        Returns:
            List[Any]: Filtered candidate tracks, sorted by similarity descending.
        """
        candidates = []
        for tid, t in self.lost_tracks.items():
            # Class compatibility check
            if getattr(t, "cls", None) != getattr(track, "cls", None):
                continue

            # Check maximum disappearance duration
            duration = frame_id - t.end_frame
            if duration > self.timeout:
                continue

            # Resolve appearance representations
            feat_query = getattr(track, "curr_feat", None)
            if feat_query is None:
                feat_query = getattr(track, "smooth_feat", None)

            feat_cand = getattr(t, "smooth_feat", None)
            if feat_cand is None:
                feat_cand = getattr(t, "curr_feat", None)

            if feat_query is None or feat_cand is None:
                continue

            # Compute cosine similarity
            norm_q = np.linalg.norm(feat_query)
            norm_c = np.linalg.norm(feat_cand)
            if norm_q == 0.0 or norm_c == 0.0:
                continue

            similarity = float(np.dot(feat_query, feat_cand) / (norm_q * norm_c))
            if similarity < self.similarity_threshold:
                continue

            # Optional spatial gating check
            if self.spatial_threshold is not None and self.spatial_threshold > 0:
                iou = bbox_iou(track.xyxy, t.xyxy)
                if iou < self.spatial_threshold:
                    continue

            candidates.append((t, similarity))

        # Sort candidates by similarity descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates]
