import numpy as np
from typing import Dict


class CostFusion:
    """Combines multiple association cost matrices using a weighted sum approach."""

    @staticmethod
    def combine(costs: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        """Combine multiple cost matrices using their configured weights.

        Args:
            costs (Dict[str, np.ndarray]): Mapping of cost component names to cost matrices.
            weights (Dict[str, float]): Mapping of cost component names to their weights.

        Returns:
            np.ndarray: The combined cost matrix.
        """
        if not costs:
            raise ValueError("No cost matrices provided for combination.")

        # Resolve output matrix shape using the first available cost matrix
        sample_matrix = next(iter(costs.values()))
        fused_cost = np.zeros_like(sample_matrix, dtype=np.float32)

        for name, cost_matrix in costs.items():
            weight = weights.get(name, 0.0)
            if weight > 0.0:
                fused_cost += weight * cost_matrix

        return fused_cost
