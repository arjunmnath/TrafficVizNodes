import numpy as np
from typing import Dict, List, Optional, Any


class SimpleRegistry:
    """Identity registry that maps local track IDs to appearance vectors and occurrences.

    No embedding matching is performed. Each local_track_id is used directly as the identity key.
    The registry stores per track:
      - smooth_embeddings: per-frame tracker moving-average appearance vectors
      - occurrence_embeddings: per-frame raw detection embeddings from FrameData.features
      - occurrences: list of occurrence metadata records (frame, timestamp, bbox, class, feed)
    """

    def __init__(self) -> None:
        # local_track_id -> {
        #   "smooth_embeddings": list[ndarray],     # tracker moving-average per frame
        #   "occurrence_embeddings": list[ndarray], # raw detection feature per frame
        #   "occurrences": list[dict]               # metadata records per observation
        # }
        self.identities: Dict[int, Dict[str, Any]] = {}

    def update_track(
        self,
        local_track_id: int,
        smooth_embedding: np.ndarray[Any, Any],
        occurrence_embedding: np.ndarray[Any, Any],
        class_label: str = "unknown",
        feed_name: str = "",
        frame_number: int = 0,
        timestamp: float = 0.0,
        bbox: Optional[List[float]] = None,
    ) -> int:
        """Register or update a track with a new frame observation.

        Args:
            local_track_id: The tracker-assigned track ID, used directly as identity key.
            smooth_embedding: The moving-average appearance vector maintained by the tracker.
            occurrence_embedding: The raw detection embedding from FrameData.features for this frame.
            class_label: Detected class name.
            feed_name: Source video feed identifier.
            frame_number: Current frame index.
            timestamp: Current timestamp in seconds.
            bbox: Bounding box [x1, y1, x2, y2].

        Returns:
            The local_track_id (identity key).
        """
        if local_track_id not in self.identities:
            self.identities[local_track_id] = {
                "smooth_embeddings": [],
                "occurrence_embeddings": [],
                "occurrences": [],
            }

        entry = self.identities[local_track_id]
        entry["smooth_embeddings"].append(smooth_embedding)
        entry["occurrence_embeddings"].append(occurrence_embedding)
        entry["occurrences"].append(
            {
                "class_label": class_label,
                "feed_name": feed_name,
                "frame": int(frame_number),
                "timestamp_seconds": float(timestamp),
                "bbox": list(map(float, bbox)) if bbox is not None else [],
            }
        )

        return local_track_id

    def get_results_summary(self) -> List[Dict[str, Any]]:
        """Return a JSON-serialisable summary of all track identities and their occurrences."""
        return [
            {
                "track_id": track_id,
                "occurrences": data["occurrences"],
            }
            for track_id, data in self.identities.items()
        ]

    def get_embeddings_dict(self) -> Dict[str, np.ndarray[Any, Any]]:
        """Return per-track stacked embeddings suitable for np.savez.

        Keys follow the pattern:
          - ``occ_{track_id}``    — stacked raw occurrence embeddings, shape (N, D)
          - ``smooth_{track_id}`` — stacked tracker moving-average embeddings, shape (N, D)

        Returns:
            Flat dict of str -> ndarray ready for ``np.savez(**result)``.
        """
        result: Dict[str, np.ndarray[Any, Any]] = {}
        for track_id, data in self.identities.items():
            result[f"occ_{track_id}"] = np.array(data["occurrence_embeddings"], dtype=np.float32)
            result[f"smooth_{track_id}"] = np.array(data["smooth_embeddings"], dtype=np.float32)
        return result
