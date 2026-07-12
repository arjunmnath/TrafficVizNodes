"""
reid/postprocessing/stages/trajectory_fusion.py

TrajectoryFusionStage: aggregates per-frame detection embeddings for a terminated
track into a single representative feature vector using either mean fusion or
scaled dot-product self-attention fusion.
"""

from __future__ import annotations

from typing import Literal, Any

import numpy as np

from reid.postprocessing.base import PostProcessingStage
from reid.postprocessing.pipeline import TerminatedTrack


# ──────────────────────────────────────────────────────────────────────────────
# Fusion functions
# ──────────────────────────────────────────────────────────────────────────────


def _l2_normalize(v: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """L2-normalize along the last axis."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + 1e-8)  # type: ignore[no-any-return]


def mean_fusion(embeddings: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Average all frame embeddings into a single prototype.

    Args:
        embeddings: Shape (N, D).

    Returns:
        L2-normalized mean vector, shape (D,).
    """
    return _l2_normalize(embeddings.mean(axis=0))


def attention_fusion(embeddings: np.ndarray[Any, Any], temperature: float = 1.0) -> np.ndarray[Any, Any]:
    """Aggregate embeddings via scaled dot-product self-attention.

    Each frame embedding attends over all others to produce a context-weighted
    value, then the attended values are mean-pooled into a single prototype.

    Steps:
        1. Normalize embeddings (queries / keys / values are all the same matrix).
        2. Compute attention scores: A = softmax(Q @ K^T / sqrt(D) / temperature).
        3. Attended output for each position: O_i = sum_j A_{ij} * V_j.
        4. Mean-pool over positions → single vector.
        5. L2-normalize result.

    Args:
        embeddings: Shape (N, D). Raw per-frame occurrence embeddings.
        temperature: Softmax temperature. Lower = sharper attention.

    Returns:
        L2-normalized attended prototype, shape (D,).
    """
    N, D = embeddings.shape

    if N == 1:
        # Single frame — nothing to attend over
        return _l2_normalize(embeddings[0])

    # Normalize embeddings before computing dot-product scores for numerical stability
    normed = _l2_normalize(embeddings)  # (N, D)

    # Scaled dot-product attention scores (N, N)
    scale = np.sqrt(D) * temperature
    scores = (normed @ normed.T) / scale  # (N, N)

    # Softmax over key dimension (axis=-1) per query
    scores -= scores.max(axis=-1, keepdims=True)  # numerical stability
    attn = np.exp(scores)
    attn /= attn.sum(axis=-1, keepdims=True)  # (N, N)

    # Weighted sum of values (using unnormalized embeddings as values)
    attended = attn @ embeddings  # (N, D)

    # Mean-pool attended outputs and normalize
    return _l2_normalize(attended.mean(axis=0))


# ──────────────────────────────────────────────────────────────────────────────
# Stage
# ──────────────────────────────────────────────────────────────────────────────

FusionMode = Literal["mean", "attention"]


class TrajectoryFusionStage(PostProcessingStage):
    """Fuse per-frame occurrence embeddings of a terminated track into one feature vector.

    Two modes are supported:
      - ``"mean"``: Simple mean pooling, fast and robust.
      - ``"attention"``: Scaled dot-product self-attention over the trajectory,
        allowing frames with more discriminative features to contribute more.

    The fused vector is stored in ``track.fused_embedding`` for use by
    downstream postprocessing stages (e.g. cross-camera matching).

    Args:
        mode: Fusion strategy, one of ``"mean"`` or ``"attention"``.
        temperature: Softmax temperature for attention mode (lower → sharper).
            Has no effect in ``"mean"`` mode.
        fallback_to_smooth: If True and ``occurrence_embeddings`` is None or empty,
            fall back to ``track.smooth_embedding`` as the fused output.
    """

    def __init__(
        self,
        mode: FusionMode = "attention",
        temperature: float = 1.0,
        fallback_to_smooth: bool = True,
    ) -> None:
        if mode not in ("mean", "attention"):
            raise ValueError(f"Unknown fusion mode {mode!r}. Choose 'mean' or 'attention'.")
        self.mode = mode
        self.temperature = temperature
        self.fallback_to_smooth = fallback_to_smooth

    def process(self, track: TerminatedTrack) -> TerminatedTrack:
        """Fuse trajectory embeddings and store result in ``track.fused_embedding``.

        Args:
            track: The TerminatedTrack entering this stage.

        Returns:
            The same track with ``fused_embedding`` populated.
        """
        embeddings = track.occurrence_embeddings

        # Validate input
        if embeddings is None or len(embeddings) == 0:
            if self.fallback_to_smooth and track.smooth_embedding is not None:
                track.fused_embedding = _l2_normalize(track.smooth_embedding.astype(np.float32))
            # Otherwise leave fused_embedding as None
            return track

        # Ensure float32 numpy array of shape (N, D)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]

        if self.mode == "mean":
            track.fused_embedding = mean_fusion(embeddings)
        else:
            track.fused_embedding = attention_fusion(embeddings, self.temperature)

        return track

    def __repr__(self) -> str:
        return (
            f"TrajectoryFusionStage(mode={self.mode!r}, "
            f"temperature={self.temperature}, "
            f"fallback_to_smooth={self.fallback_to_smooth})"
        )
