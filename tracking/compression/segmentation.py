import math
import numpy as np
from typing import List, Tuple

from tracking.domain.interfaces import SegmentationStrategy


class AdaptiveSegmentation(SegmentationStrategy):
    """Adaptive trajectory segmentation using heading, speed, and fitting error thresholds."""

    def __init__(
        self,
        heading_threshold: float = 0.5,  # ~30 degrees in radians
        speed_change_threshold: float = 0.4,  # 40% change
        max_fitting_error: float = 8.0,  # Max pixel distance error
        min_segment_length: int = 3,  # Min points per segment (at least 3 for spline fitting)
    ):
        self.heading_threshold = heading_threshold
        self.speed_change_threshold = speed_change_threshold
        self.max_fitting_error = max_fitting_error
        self.min_segment_length = max(2, min_segment_length)

    def _angle_diff(self, a1: float, a2: float) -> float:
        """Compute the shortest angular difference between two angles in radians."""
        diff = (a1 - a2 + math.pi) % (2 * math.pi) - math.pi
        return abs(diff)

    def segment(
        self,
        timestamps: List[float],
        positions: List[Tuple[float, float]],
        velocities: List[Tuple[float, float]],
        headings: List[float],
    ) -> List[Tuple[int, int]]:
        """Divide the raw trajectory into segments based on change thresholds."""
        n = len(timestamps)
        if n < self.min_segment_length:
            return [(0, n - 1)] if n >= 1 else []

        segments: List[Tuple[int, int]] = []
        i_start = 0

        # Pre-calculate speeds
        speeds = [math.sqrt(vx**2 + vy**2) for vx, vy in velocities]

        j = i_start + 1
        while j < n:
            # Check length constraint
            seg_len = j - i_start + 1
            if seg_len <= self.min_segment_length:
                j += 1
                continue

            # 1. Heading change check
            # Check diff between current heading and initial segment heading
            h_diff = self._angle_diff(headings[j], headings[i_start])
            if h_diff > self.heading_threshold:
                segments.append((i_start, j - 1))
                i_start = j - 1
                j = i_start + 1
                continue

            # 2. Speed change check
            initial_speed = speeds[i_start]
            curr_speed = speeds[j]
            speed_change = abs(curr_speed - initial_speed) / max(1.0, initial_speed)
            if speed_change > self.speed_change_threshold:
                segments.append((i_start, j - 1))
                i_start = j - 1
                j = i_start + 1
                continue

            # 3. Linear fitting error check
            times = np.array(timestamps[i_start : j + 1])
            xs = np.array([p[0] for p in positions[i_start : j + 1]])
            ys = np.array([p[1] for p in positions[i_start : j + 1]])

            # Fit linear models for x and y
            x_coeffs = np.polyfit(times, xs, 1)
            y_coeffs = np.polyfit(times, ys, 1)

            x_fitted = np.polyval(x_coeffs, times)
            y_fitted = np.polyval(y_coeffs, times)

            errors = np.sqrt((xs - x_fitted) ** 2 + (ys - y_fitted) ** 2)
            max_err = float(np.max(errors))

            if max_err > self.max_fitting_error:
                # If error is too high, split at j-1
                segments.append((i_start, j - 1))
                i_start = j - 1
                j = i_start + 1
                continue

            j += 1

        # Add the final segment
        if i_start < n - 1:
            segments.append((i_start, n - 1))

        # Merge segments that are too short into their neighbors
        final_segments: List[Tuple[int, int]] = []
        for s in segments:
            if not final_segments:
                final_segments.append(s)
            else:
                prev_start, prev_end = final_segments[-1]
                curr_len = s[1] - s[0] + 1
                # If current segment is too small, merge with previous
                if curr_len < self.min_segment_length:
                    final_segments[-1] = (prev_start, s[1])
                else:
                    final_segments.append(s)

        return final_segments
