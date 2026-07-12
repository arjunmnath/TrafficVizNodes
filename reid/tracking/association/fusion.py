import numpy as np
from typing import Dict, List, Optional, Any


class CostFusion:
    """Combines multiple association cost matrices using a weighted sum or official score-appearance-IoU fusion."""

    @staticmethod
    def combine(
        costs: Dict[str, np.ndarray[Any, Any]],
        weights: Dict[str, float],
        detections: Optional[List[Any]] = None,
        alpha: float = 0.8,
        missing_mask: Optional[np.ndarray[Any, Any]] = None,
        fuse_score: bool = False,
    ) -> np.ndarray[Any, Any]:
        """Combine multiple cost matrices using the score-appearance-IoU fusion formula.

        Formula:
            S = IoU_similarity * (alpha * score + (1 - alpha) * appearance_similarity)
            C = 1 - S

        If detections is not provided, falls back to a simple weighted sum of cost matrices.

        Args:
            costs (Dict[str, np.ndarray[Any, Any]]): Mapping of cost component names to cost matrices.
            weights (Dict[str, float]): Mapping of cost component names to their weights.
            detections (List[Any], optional): List of detection objects containing detection scores.
            alpha (float, optional): Weighting factor between detection score and appearance. Defaults to 0.8.
            missing_mask (np.ndarray[Any, Any], optional): Boolean mask indicating where appearance features are missing.
            fuse_score (bool, optional): Whether to apply detection score fusion on fallback costs.

        Returns:
            np.ndarray[Any, Any]: The combined cost matrix.
        """
        if not costs:
            raise ValueError("No cost matrices provided for combination.")

        iou_cost = costs.get("iou")
        appearance_cost = costs.get("appearance")
        appearance_weight = weights.get("appearance", 0.0)

        # Fallback to simple weighted combine if inputs for official fusion are missing or appearance is disabled
        if (
            iou_cost is None
            or appearance_cost is None
            or detections is None
            or appearance_weight <= 0.0
        ):
            sample_matrix = next(iter(costs.values()))
            fused_cost = np.zeros_like(sample_matrix, dtype=np.float32)

            for name, cost_matrix in costs.items():
                weight = weights.get(name, 0.0)
                if weight > 0.0:
                    fused_cost += weight * cost_matrix

            if fuse_score and detections is not None and iou_cost is not None and iou_cost.size > 0:
                iou_sim = 1.0 - fused_cost
                det_scores = np.array([det.score for det in detections], dtype=np.float32)
                det_scores = np.expand_dims(det_scores, axis=0).repeat(fused_cost.shape[0], axis=0)
                fused_cost = (1.0 - iou_sim * det_scores).astype(np.float32)

            return fused_cost

        if iou_cost.size == 0:
            return iou_cost

        # Official score-appearance-IoU fusion logic:
        # S = IoU * (alpha * score + (1 - alpha) * appearance_similarity)
        # C = 1 - S
        iou_sim = 1.0 - iou_cost
        appearance_sim = 1.0 - appearance_cost

        det_scores = np.array([det.score for det in detections], dtype=np.float32)
        det_scores = np.expand_dims(det_scores, axis=0).repeat(iou_cost.shape[0], axis=0)

        fused_sim = iou_sim * (alpha * det_scores + (1.0 - alpha) * appearance_sim)
        fused_cost = (1.0 - fused_sim).astype(np.float32)

        # Handle missing appearance features fallback if mask is provided
        if missing_mask is not None:
            if fuse_score:
                fallback_cost = (1.0 - iou_sim * det_scores).astype(np.float32)
            else:
                fallback_cost = iou_cost
            fused_cost = np.where(missing_mask, fallback_cost, fused_cost)

        return fused_cost
