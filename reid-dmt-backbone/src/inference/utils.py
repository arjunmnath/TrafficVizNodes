import torch
import numpy as np

def get_default_device() -> str:
    """Returns 'cuda' if GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def compute_distance_matrix(qf: torch.Tensor, gf: torch.Tensor, metric: str = "euclidean") -> np.ndarray:
    """Helper to compute a pairwise distance matrix between query and gallery embeddings.
    
    Supports both PyTorch Tensors and NumPy arrays.
    Returns a NumPy array.
    """
    if isinstance(qf, np.ndarray):
        qf = torch.from_numpy(qf)
    if isinstance(gf, np.ndarray):
        gf = torch.from_numpy(gf)
        
    if metric == "euclidean":
        m = qf.shape[0]
        n = gf.shape[0]
        q_pow = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n)
        g_pow = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist_mat = q_pow + g_pow
        dist_mat = torch.addmm(dist_mat, qf, gf.t(), beta=1, alpha=-2)
        dist_mat = torch.clamp(dist_mat, min=0.0)
        return dist_mat.cpu().numpy()
    elif metric == "cosine":
        qf_norm = torch.nn.functional.normalize(qf, p=2, dim=1)
        gf_norm = torch.nn.functional.normalize(gf, p=2, dim=1)
        similarity = torch.mm(qf_norm, gf_norm.t())
        return (1.0 - similarity).cpu().numpy()
    else:
        raise ValueError(f"Unknown metric: {metric}")
