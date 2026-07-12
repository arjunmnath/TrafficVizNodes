import cv2
import numpy as np
from typing import List, Tuple, Optional, Any

from tracking.domain.track import CompressedTrack
from tracking.compression.reconstruction import BBoxReconstructor


class TrajectoryRenderer:
    """Renders track trajectories, segment boundaries, and bounding boxes onto images."""

    @staticmethod
    def draw_trajectory_path(
        image: np.ndarray[Any, Any],
        track: CompressedTrack,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        draw_segment_boundaries: bool = True,
        boundary_color: Tuple[int, int, int] = (0, 0, 255),
    ) -> np.ndarray[Any, Any]:
        """Draw the continuous center-point path of the compressed track onto the image."""
        img = image.copy()
        t0, t1 = track.trajectory.t0, track.trajectory.t1
        if t1 <= t0:
            return img

        # Sample points along the trajectory
        times = np.linspace(t0, t1, 200)
        pts = [track.position(t) for t in times]

        # Draw continuous path
        for i in range(len(pts) - 1):
            p1 = (int(round(pts[i][0])), int(round(pts[i][1])))
            p2 = (int(round(pts[i + 1][0])), int(round(pts[i + 1][1])))
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

        # Draw segment boundaries
        if draw_segment_boundaries:
            for seg in track.trajectory.segments:
                # Start boundary
                p_start = track.position(seg.t0)
                cp_start = (int(round(p_start[0])), int(round(p_start[1])))
                cv2.circle(img, cp_start, 5, boundary_color, -1)

                # End boundary
                p_end = track.position(seg.t1)
                cp_end = (int(round(p_end[0])), int(round(p_end[1])))
                cv2.circle(img, cp_end, 5, boundary_color, -1)

        return img

    @staticmethod
    def draw_reconstructed_bbox(
        image: np.ndarray[Any, Any],
        track: CompressedTrack,
        t: float,
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
        label: Optional[str] = None,
    ) -> np.ndarray[Any, Any]:
        """Draw the reconstructed bounding box at timestamp t onto the image."""
        img = image.copy()
        x1, y1, x2, y2 = BBoxReconstructor.reconstruct(track, t)

        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))

        cv2.rectangle(img, p1, p2, color, thickness)

        if label:
            cv2.putText(
                img,
                label,
                (p1[0], max(15, p1[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        return img

    @staticmethod
    def plot_reconstruction_comparison(
        track: CompressedTrack,
        raw_times: List[float],
        raw_positions: List[Tuple[float, float]],
        output_image_path: str,
    ) -> None:
        """Generate a matplotlib plot comparing original vs reconstructed coordinates and save it."""
        try:
            import matplotlib.pyplot as plt

            times = np.array(raw_times)
            raw_xs = np.array([p[0] for p in raw_positions])
            raw_ys = np.array([p[1] for p in raw_positions])

            rec_pts = [track.position(t) for t in raw_times]
            rec_xs = np.array([p[0] for p in rec_pts])
            rec_ys = np.array([p[1] for p in rec_pts])

            plt.figure(figsize=(10, 6))

            plt.subplot(2, 1, 1)
            plt.plot(times, raw_xs, "ro-", label="Original X", alpha=0.6)
            plt.plot(times, rec_xs, "b-", label="Reconstructed X")
            plt.title(f"Track {track.metadata.track_id} Coordinate Comparison")
            plt.ylabel("X coordinate")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot(times, raw_ys, "ro-", label="Original Y", alpha=0.6)
            plt.plot(times, rec_ys, "b-", label="Reconstructed Y")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Y coordinate")
            plt.legend()
            plt.grid(True)

            # Highlight segment boundary times
            for seg in track.trajectory.segments:
                plt.subplot(2, 1, 1)
                plt.axvline(x=seg.t0, color="gray", linestyle="--", alpha=0.5)
                plt.subplot(2, 1, 2)
                plt.axvline(x=seg.t0, color="gray", linestyle="--", alpha=0.5)

            plt.tight_layout()
            plt.savefig(output_image_path, dpi=150)
            plt.close()
        except ImportError:
            # Matplotlib is not installed, fail silently/gracefully
            pass
