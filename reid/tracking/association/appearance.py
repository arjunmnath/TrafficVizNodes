import numpy as np
from ultralytics.trackers.utils import matching
from .base import AssociationCost


class AppearanceCost(AssociationCost):
    """Computes association cost based on appearance feature embedding distance."""

    def compute(self, tracks: list, detections: list) -> np.ndarray:
        """Compute the appearance cost matrix between tracks and detections.

        Args:
            tracks (list): List of track instances containing ReID features.
            detections (list): List of detection instances containing ReID features.

        Returns:
            np.ndarray: (N x M) appearance cosine cost matrix.
        """
        if not tracks or not detections:
            return np.empty((len(tracks), len(detections)), dtype=np.float32)

        # Calculates cosine distance: 1 - cosine_similarity.
        # This returns values in [0, 2] (or 2.0 if missing features).
        return matching.embedding_distance(tracks, detections)
