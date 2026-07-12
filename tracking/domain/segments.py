from typing import Any, Dict, List, Tuple
from scipy.interpolate import CubicSpline  # type: ignore[import-untyped]

from tracking.domain.interfaces import TrajectorySegment


class ConstantSegment(TrajectorySegment):
    """A segment representing a stationary object (constant position)."""

    def __init__(self, t0: float, t1: float, cx: float, cy: float, max_err: float = 0.0):
        self.t0 = t0
        self.t1 = t1
        self.cx = cx
        self.cy = cy
        self._max_err = max_err

    def __call__(self, t: float) -> Tuple[float, float]:
        return self.position(t)

    def position(self, t: float) -> Tuple[float, float]:
        # Clip time to segment boundaries
        return self.cx, self.cy

    def velocity(self, t: float) -> Tuple[float, float]:
        return 0.0, 0.0

    def duration(self) -> float:
        return self.t1 - self.t0

    def max_error(self) -> float:
        return self._max_err

    def serialize(self) -> Dict[str, Any]:
        return {
            "type": "constant",
            "t0": float(self.t0),
            "t1": float(self.t1),
            "parameters": {"cx": float(self.cx), "cy": float(self.cy)},
            "max_error": float(self._max_err),
        }


class LinearSegment(TrajectorySegment):
    """A segment representing linear motion: x(t) = a*t + b, y(t) = c*t + d."""

    def __init__(
        self,
        t0: float,
        t1: float,
        a: float,
        b: float,
        c: float,
        d: float,
        max_err: float = 0.0,
    ):
        self.t0 = t0
        self.t1 = t1
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self._max_err = max_err

    def __call__(self, t: float) -> Tuple[float, float]:
        return self.position(t)

    def position(self, t: float) -> Tuple[float, float]:
        return self.a * t + self.b, self.c * t + self.d

    def velocity(self, t: float) -> Tuple[float, float]:
        return self.a, self.c

    def duration(self) -> float:
        return self.t1 - self.t0

    def max_error(self) -> float:
        return self._max_err

    def serialize(self) -> Dict[str, Any]:
        return {
            "type": "linear",
            "t0": float(self.t0),
            "t1": float(self.t1),
            "parameters": {
                "a": float(self.a),
                "b": float(self.b),
                "c": float(self.c),
                "d": float(self.d),
            },
            "max_error": float(self._max_err),
        }


class CubicSplineSegment(TrajectorySegment):
    """A segment representing smooth motion using cubic splines."""

    def __init__(self, control_points: List[List[float]], max_err: float = 0.0):
        """control_points is a list of [t, x, y] points."""
        self.control_points = sorted(control_points, key=lambda pt: pt[0])
        self.t0 = self.control_points[0][0]
        self.t1 = self.control_points[-1][0]
        self._max_err = max_err

        # Fit splines
        times = [pt[0] for pt in self.control_points]
        xs = [pt[1] for pt in self.control_points]
        ys = [pt[2] for pt in self.control_points]

        # Use BcType if needed, but standard not-a-knot is good
        self.cs_x = CubicSpline(times, xs)
        self.cs_y = CubicSpline(times, ys)

    def __call__(self, t: float) -> Tuple[float, float]:
        return self.position(t)

    def position(self, t: float) -> Tuple[float, float]:
        # Clip time to segment boundaries for stability in interpolation
        t_clipped = max(self.t0, min(self.t1, t))
        return float(self.cs_x(t_clipped)), float(self.cs_y(t_clipped))

    def velocity(self, t: float) -> Tuple[float, float]:
        t_clipped = max(self.t0, min(self.t1, t))
        # Evaluate first derivative
        vx = float(self.cs_x(t_clipped, 1))
        vy = float(self.cs_y(t_clipped, 1))
        return vx, vy

    def duration(self) -> float:
        return self.t1 - self.t0

    def max_error(self) -> float:
        return self._max_err

    def serialize(self) -> Dict[str, Any]:
        return {
            "type": "spline",
            "t0": float(self.t0),
            "t1": float(self.t1),
            "control_points": [
                [float(pt[0]), float(pt[1]), float(pt[2])] for pt in self.control_points
            ],
            "max_error": float(self._max_err),
        }
