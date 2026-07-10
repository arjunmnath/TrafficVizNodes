from abc import ABC, abstractmethod
import numpy as np
from ultralytics.trackers.utils import matching


class AssociationCost(ABC):
    """Base interface for tracking association cost metrics."""

    @abstractmethod
    def compute(self, tracks: list, detections: list) -> np.ndarray:
        """Compute the cost matrix between tracks and detections.

        Args:
            tracks (list): A list of tracked object instances.
            detections (list): A list of current detection object instances.

        Returns:
            np.ndarray: An (N x M) cost matrix where N is the number of tracks
                        and M is the number of detections.
        """
        pass


class IoUCost(AssociationCost):
    """Computes association cost based on Intersection over Union (IoU) distance."""

    def compute(self, tracks: list, detections: list) -> np.ndarray:
        """Compute the IoU cost matrix between tracks and detections.

        Args:
            tracks (list): List of track instances.
            detections (list): List of detection instances.

        Returns:
            np.ndarray: (N x M) IoU cost matrix (1 - IoU).
        """
        if not tracks or not detections:
            return np.empty((len(tracks), len(detections)), dtype=np.float32)
        return matching.iou_distance(tracks, detections)
