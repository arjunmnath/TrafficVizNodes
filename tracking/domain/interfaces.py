from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class Interpolator(ABC):
    """Interface for time-based interpolation of scalar or vector quantities."""

    @abstractmethod
    def __call__(self, t: float) -> Any:
        """Evaluate the interpolated value at time t."""
        pass


class TrajectorySegment(Interpolator, ABC):
    """Interface for a single temporal segment of a piecewise trajectory."""

    t0: float
    t1: float

    @abstractmethod
    def position(self, t: float) -> Tuple[float, float]:
        """Evaluate the 2D center position (cx, cy) at time t."""
        pass

    @abstractmethod
    def velocity(self, t: float) -> Tuple[float, float]:
        """Evaluate the 2D velocity (vx, vy) at time t."""
        pass

    @abstractmethod
    def duration(self) -> float:
        """Return the temporal duration of this segment."""
        pass

    @abstractmethod
    def max_error(self) -> float:
        """Return the maximum fitting error of this segment."""
        pass

    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """Serialize segment to a dictionary."""
        pass


class TrajectoryModel(Interpolator, ABC):
    """Interface for a complete multi-segment trajectory."""

    @abstractmethod
    def position(self, t: float) -> Tuple[float, float]:
        """Evaluate the 2D position (cx, cy) at time t."""
        pass

    @abstractmethod
    def velocity(self, t: float) -> Tuple[float, float]:
        """Evaluate the 2D velocity (vx, vy) at time t."""
        pass

    @abstractmethod
    def direction(self, t: float) -> float:
        """Evaluate the heading direction (radians) at time t."""
        pass

    @abstractmethod
    def duration(self) -> float:
        """Return total duration of the trajectory."""
        pass


class SizeModel(Interpolator, ABC):
    """Interface for modeling bounding box width and height over time."""

    @abstractmethod
    def width(self, t: float) -> float:
        """Evaluate width at time t."""
        pass

    @abstractmethod
    def height(self, t: float) -> float:
        """Evaluate height at time t."""
        pass

    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """Serialize size model parameters to a dictionary."""
        pass


class TrajectoryFitter(ABC):
    """Interface for fitting a trajectory model to a sequence of points."""

    @abstractmethod
    def fit(
        self, timestamps: List[float], positions: List[Tuple[float, float]]
    ) -> TrajectorySegment:
        """Fit a trajectory segment to the given timestamp and position data."""
        pass


class SegmentationStrategy(ABC):
    """Interface for detecting segment boundaries in trajectory data."""

    @abstractmethod
    def segment(
        self,
        timestamps: List[float],
        positions: List[Tuple[float, float]],
        velocities: List[Tuple[float, float]],
        headings: List[float],
    ) -> List[Tuple[int, int]]:
        """Divide raw trajectory indices into segment start/end index boundaries."""
        pass


class Serializer(ABC):
    """Interface for serializing/deserializing tracks."""

    @abstractmethod
    def serialize(self, track: Any) -> Any:
        """Serialize the track object."""
        pass

    @abstractmethod
    def deserialize(self, data: Any) -> Any:
        """Deserialize representation back to track object."""
        pass
