import math
import numpy as np
from typing import Dict, List, Tuple

from tracking.domain.track import CompressedTrack
from tracking.compression.reconstruction import BBoxReconstructor


def compute_iou(
    bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]
) -> float:
    """Compute the Intersection-Over-Union (IoU) of two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Intersection coordinates
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = box1_area + box2_area - inter_area
    if union_area < 1e-8:
        return 0.0

    return inter_area / union_area


class CompressionMetrics:
    """Evaluates compression efficiency and reconstruction accuracy."""

    @staticmethod
    def evaluate(
        track: CompressedTrack,
        raw_frames: List[int],
        raw_timestamps: List[float],
        raw_bboxes: List[Tuple[float, float, float, float]],
    ) -> Dict[str, float]:
        """Compare compressed track against raw detections to evaluate error metrics."""
        n = len(raw_timestamps)
        if n == 0:
            return {}

        position_errors = []
        width_errors = []
        height_errors = []
        ious = []

        for f, t, bbox in zip(raw_frames, raw_timestamps, raw_bboxes):
            # Original values
            x1, y1, x2, y2 = bbox
            orig_cx = (x1 + x2) / 2.0
            orig_cy = (y1 + y2) / 2.0
            orig_w = x2 - x1
            orig_h = y2 - y1

            # Reconstructed values
            rec_cx, rec_cy = track.position(t)
            rec_w = track.width(t)
            rec_h = track.height(t)
            rec_bbox = BBoxReconstructor.reconstruct(track, t)

            # Errors
            pos_err = math.sqrt((orig_cx - rec_cx) ** 2 + (orig_cy - rec_cy) ** 2)
            w_err = abs(orig_w - rec_w)
            h_err = abs(orig_h - rec_h)
            iou = compute_iou(bbox, rec_bbox)

            position_errors.append(pos_err)
            width_errors.append(w_err)
            height_errors.append(h_err)
            ious.append(iou)

        # Compression ratio computation
        # Calculate parameters stored:
        # Each constant segment: t0 (1), t1 (1), cx (1), cy (1) = 4 floats
        # Each linear segment: t0 (1), t1 (1), a (1), b (1), c (1), d (1) = 6 floats
        # Each spline segment: t0 (1), t1 (1), len(pts)*3 = 2 + 3 * len(pts) floats
        # Plus metadata & time model (we can count float values)
        # Original size: n * 5 (frame, t, x1, y1, x2, y2)
        param_count = 0
        for seg in track.trajectory.segments:
            if seg.serialize()["type"] == "constant":
                param_count += 4
            elif seg.serialize()["type"] == "linear":
                param_count += 6
            elif seg.serialize()["type"] == "spline":
                param_count += 2 + 3 * len(seg.serialize()["control_points"])

        # Size model parameters
        sz_serialized = track.size_model.serialize()
        if sz_serialized["type"] == "linear":
            param_count += 4
        elif sz_serialized["type"] == "constant":
            param_count += 2
        elif sz_serialized["type"] == "spline":
            param_count += len(sz_serialized["control_points"]) * 3
        elif sz_serialized["type"] == "polynomial":
            param_count += len(sz_serialized["parameters"]["w_coeffs"]) + len(
                sz_serialized["parameters"]["h_coeffs"]
            )

        raw_size = n * 6  # (frame, timestamp, x1, y1, x2, y2)
        compression_ratio = float(raw_size) / max(1.0, float(param_count))

        return {
            "mean_position_error_px": float(np.mean(position_errors)),
            "max_position_error_px": float(np.max(position_errors)),
            "mean_width_error_px": float(np.mean(width_errors)),
            "mean_height_error_px": float(np.mean(height_errors)),
            "mean_iou": float(np.mean(ious)),
            "min_iou": float(np.min(ious)),
            "compression_ratio": compression_ratio,
        }
