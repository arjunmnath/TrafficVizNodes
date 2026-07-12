import torch
from typing import List


def fuse_embeddings(embeddings_list: List[torch.Tensor]) -> torch.Tensor:
    """Fuses a list of embeddings from multiple models by taking their mean (centroid) and L2-normalizing the result."""
    if not embeddings_list:
        raise ValueError("embeddings_list cannot be empty")

    fused = torch.stack(embeddings_list, dim=0).mean(dim=0)

    # Re-normalize to ensure the final output is L2-normalized
    fused = torch.nn.functional.normalize(fused, p=2, dim=-1)
    return fused
