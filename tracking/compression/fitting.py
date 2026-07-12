import math
import numpy as np
from typing import List, Tuple

from tracking.domain.interfaces import TrajectoryFitter, TrajectorySegment
from tracking.domain.segments import ConstantSegment, LinearSegment, CubicSplineSegment


class ConstantFitter(TrajectoryFitter):
    """Fits a ConstantSegment to the given trajectory data by computing the mean position."""

    def fit(
        self, timestamps: List[float], positions: List[Tuple[float, float]]
    ) -> TrajectorySegment:
        if not positions:
            raise ValueError("Cannot fit ConstantSegment to empty data.")

        t0, t1 = timestamps[0], timestamps[-1]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        cx = float(np.mean(xs))
        cy = float(np.mean(ys))

        # Calculate max error
        errors = np.sqrt((np.array(xs) - cx) ** 2 + (np.array(ys) - cy) ** 2)
        max_err = float(np.max(errors))

        return ConstantSegment(t0=t0, t1=t1, cx=cx, cy=cy, max_err=max_err)


class LinearFitter(TrajectoryFitter):
    """Fits a LinearSegment to the given trajectory data using linear regression."""

    def fit(
        self, timestamps: List[float], positions: List[Tuple[float, float]]
    ) -> TrajectorySegment:
        if len(timestamps) < 2:
            raise ValueError("LinearFitter requires at least 2 points.")

        t0, t1 = timestamps[0], timestamps[-1]
        times = np.array(timestamps)
        xs = np.array([p[0] for p in positions])
        ys = np.array([p[1] for p in positions])

        # Perform linear regression: x = a*t + b, y = c*t + d
        a, b = np.polyfit(times, xs, 1)
        c, d = np.polyfit(times, ys, 1)

        # Compute maximum L2 error
        x_fitted = a * times + b
        y_fitted = c * times + d
        errors = np.sqrt((xs - x_fitted) ** 2 + (ys - y_fitted) ** 2)
        max_err = float(np.max(errors))

        return LinearSegment(
            t0=t0, t1=t1, a=float(a), b=float(b), c=float(c), d=float(d), max_err=max_err
        )


class SplineFitter(TrajectoryFitter):
    """Fits a CubicSplineSegment to the given trajectory data."""

    def __init__(self, downsample_factor: int = 1):
        """
        downsample_factor: if > 1, downsamples points to select a subset of control points.
        """
        self.downsample_factor = max(1, downsample_factor)

    def fit(
        self, timestamps: List[float], positions: List[Tuple[float, float]]
    ) -> TrajectorySegment:
        n = len(timestamps)
        if n < 2:
            raise ValueError("SplineFitter requires at least 2 points.")

        # Reconstruct spline control points. If n is small, we must keep at least 2 points.
        # SciPy CubicSpline requires at least 2 points, but 3+ is better.
        indices = list(range(0, n, self.downsample_factor))
        if indices[-1] != n - 1:
            indices.append(n - 1)

        # Ensure unique control point indices
        indices = sorted(list(set(indices)))

        control_points = [
            [timestamps[idx], positions[idx][0], positions[idx][1]] for idx in indices
        ]

        # Fit segment to evaluate maximum error over all original points
        segment = CubicSplineSegment(control_points=control_points)

        # Compute max error on all points
        errors = []
        for t, pos in zip(timestamps, positions):
            rec_pos = segment.position(t)
            err = math.sqrt((pos[0] - rec_pos[0]) ** 2 + (pos[1] - rec_pos[1]) ** 2)
            errors.append(err)
        max_err = float(np.max(errors))

        segment._max_err = max_err
        return segment
