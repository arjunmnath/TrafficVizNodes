import numpy as np
from typing import Any, Dict, List, Tuple
from scipy.interpolate import CubicSpline  # type: ignore[import-untyped]

from tracking.domain.interfaces import SizeModel


class ConstantModel(SizeModel):
    """A size model representing constant width and height over time."""

    def __init__(self, w0: float, h0: float):
        self.w0 = w0
        self.h0 = h0

    def __call__(self, t: float) -> Tuple[float, float]:
        return self.width(t), self.height(t)

    def width(self, t: float) -> float:
        return self.w0

    def height(self, t: float) -> float:
        return self.h0

    def serialize(self) -> Dict[str, Any]:
        return {"type": "constant", "parameters": {"w0": float(self.w0), "h0": float(self.h0)}}


class LinearModel(SizeModel):
    """A size model representing linear change over time: w(t) = a*t + b, h(t) = c*t + d."""

    def __init__(self, a: float, b: float, c: float, d: float):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __call__(self, t: float) -> Tuple[float, float]:
        return self.width(t), self.height(t)

    def width(self, t: float) -> float:
        # Prevent negative size
        return max(1.0, self.a * t + self.b)

    def height(self, t: float) -> float:
        # Prevent negative size
        return max(1.0, self.c * t + self.d)

    def serialize(self) -> Dict[str, Any]:
        return {
            "type": "linear",
            "parameters": {
                "a": float(self.a),
                "b": float(self.b),
                "c": float(self.c),
                "d": float(self.d),
            },
        }


class PolynomialModel(SizeModel):
    """A size model representing polynomial change: w(t) = sum(w_coeffs[i] * t^(N-i))."""

    def __init__(self, w_coeffs: List[float], h_coeffs: List[float]):
        self.w_coeffs = w_coeffs
        self.h_coeffs = h_coeffs

    def __call__(self, t: float) -> Tuple[float, float]:
        return self.width(t), self.height(t)

    def width(self, t: float) -> float:
        val = float(np.polyval(self.w_coeffs, t))
        return max(1.0, val)

    def height(self, t: float) -> float:
        val = float(np.polyval(self.h_coeffs, t))
        return max(1.0, val)

    def serialize(self) -> Dict[str, Any]:
        return {
            "type": "polynomial",
            "parameters": {
                "w_coeffs": [float(c) for c in self.w_coeffs],
                "h_coeffs": [float(c) for c in self.h_coeffs],
            },
        }


class SplineModel(SizeModel):
    """A size model using 1D cubic splines to interpolate width and height."""

    def __init__(self, control_points: List[List[float]]):
        """control_points is a list of [t, w, h] points."""
        self.control_points = sorted(control_points, key=lambda pt: pt[0])
        self.t0 = self.control_points[0][0]
        self.t1 = self.control_points[-1][0]

        times = [pt[0] for pt in self.control_points]
        ws = [pt[1] for pt in self.control_points]
        hs = [pt[2] for pt in self.control_points]

        self.cs_w = CubicSpline(times, ws)
        self.cs_h = CubicSpline(times, hs)

    def __call__(self, t: float) -> Tuple[float, float]:
        return self.width(t), self.height(t)

    def width(self, t: float) -> float:
        t_clipped = max(self.t0, min(self.t1, t))
        val = float(self.cs_w(t_clipped))
        return max(1.0, val)

    def height(self, t: float) -> float:
        t_clipped = max(self.t0, min(self.t1, t))
        val = float(self.cs_h(t_clipped))
        return max(1.0, val)

    def serialize(self) -> Dict[str, Any]:
        return {
            "type": "spline",
            "control_points": [
                [float(pt[0]), float(pt[1]), float(pt[2])] for pt in self.control_points
            ],
        }
