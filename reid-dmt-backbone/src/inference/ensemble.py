import torch
import numpy as np
from typing import List, Dict, Union


def fuse_embeddings(embeddings_list: List[torch.Tensor], method: str = "concat") -> torch.Tensor:
    """Fuses a list of embeddings from multiple models.

    Expected inputs are already L2-normalized.
    Methods:
    - 'concat': Concatenates the embeddings and L2-normalizes the result.
    - 'mean': Averages the embeddings and L2-normalizes the result.
    """
    if not embeddings_list:
        raise ValueError("embeddings_list cannot be empty")

    if method == "concat":
        fused = torch.cat(embeddings_list, dim=-1)
    elif method == "mean":
        fused = torch.stack(embeddings_list, dim=0).mean(dim=0)
    else:
        raise ValueError(f"Unknown fusion method: {method}")

    # Re-normalize to ensure the final output is L2-normalized
    fused = torch.nn.functional.normalize(fused, p=2, dim=-1)
    return fused


def compute_euclidean_distance(qf: torch.Tensor, gf: torch.Tensor) -> torch.Tensor:
    """Computes pairwise Euclidean distance between query and gallery features in PyTorch."""
    m = qf.shape[0]
    n = gf.shape[0]
    # dist = q^2 + g^2 - 2 * q * g.T
    q_pow = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n)
    g_pow = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat = q_pow + g_pow
    dist_mat = torch.addmm(dist_mat, qf, gf.t(), beta=1, alpha=-2)
    # Ensure no negative values due to floating-point precision
    dist_mat = torch.clamp(dist_mat, min=0.0)
    return dist_mat


def compute_cosine_distance(qf: torch.Tensor, gf: torch.Tensor) -> torch.Tensor:
    """Computes pairwise Cosine distance (1 - cosine_similarity) in PyTorch."""
    # First, make sure they are L2-normalized
    qf_norm = torch.nn.functional.normalize(qf, p=2, dim=1)
    gf_norm = torch.nn.functional.normalize(gf, p=2, dim=1)
    similarity = torch.mm(qf_norm, gf_norm.t())
    return 1.0 - similarity


def fuse_distance_matrices(
    query_embeddings_list: List[torch.Tensor],
    gallery_embeddings_list: List[torch.Tensor],
    metric: str = "euclidean",
) -> np.ndarray:
    """Fuses distance matrices from multiple models by summing them.

    Matches the ensembling technique in the original ensemble.py.
    """
    if len(query_embeddings_list) != len(gallery_embeddings_list):
        raise ValueError("Query and gallery lists must have the same length")

    fused_dist = None

    for qf, gf in zip(query_embeddings_list, gallery_embeddings_list):
        if metric == "euclidean":
            dist = compute_euclidean_distance(qf, gf)
        elif metric == "cosine":
            dist = compute_cosine_distance(qf, gf)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        dist_np = dist.cpu().numpy()
        if fused_dist is None:
            fused_dist = dist_np
        else:
            fused_dist += dist_np

    return fused_dist
