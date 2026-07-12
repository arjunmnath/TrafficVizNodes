import torch
import torch.nn as nn
from ..config import InferenceConfig


class EnsembleModel(nn.Module):
    """Unified neural network model that ensembles 3 ReID submodels and provides a single forward execution point."""

    def __init__(self, config):
        """Constructor.

        Args:
            config (EnsembleConfig): Configuration for the ensemble model.
        """
        super().__init__()
        self.config = config
        self.flip_feats = config.flip_feats

        # Import model factory builder inside init to avoid circular dependency
        from ..model_factory import build_model_from_config
        from reid.utils import resolve_model_weights

        self.submodels = nn.ModuleList()
        for backbone, checkpoint in zip(config.backbones, config.checkpoint_paths):
            resolved_checkpoint = resolve_model_weights(checkpoint)
            sub_cfg = InferenceConfig(
                backbone=backbone,
                checkpoint_path=resolved_checkpoint,
                device=config.device,
                fp16=config.fp16,
                image_size=config.image_size,
                flip_feats=config.flip_feats,
            )
            submodel = build_model_from_config(sub_cfg)
            self.submodels.append(submodel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the EnsembleModel.

        Extracts features from each submodule, applies horizontal flip augmentation
        (if enabled), L2-normalizes each submodel's output, and computes the mean
        centroid of features across the 3 submodels before applying final L2-normalization.

        Args:
            x (torch.Tensor): Preprocessed batch of images of shape (N, C, H, W).

        Returns:
            torch.Tensor: Fused and normalized features of shape (N, D).
        """
        model_embeddings = []

        for model in self.submodels:
            if self.flip_feats:
                # Forward original
                f2 = model(x)
                # Flip horizontal: dimension 3 is width
                inv_idx = torch.arange(x.size(3) - 1, -1, -1, device=x.device).long()
                flipped_x = x.index_select(3, inv_idx)
                f1 = model(flipped_x)
                feat = f2 + f1
            else:
                feat = model(x)

            # L2 normalize each model's feature representation
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            model_embeddings.append(feat)

        # Fusing: mean centroid across models
        fused = torch.stack(model_embeddings, dim=0).mean(dim=0)

        # Re-normalize to ensure the final output is L2-normalized
        fused = torch.nn.functional.normalize(fused, p=2, dim=-1)
        return fused
