import torch
import torch.nn as nn

from .vecsetx import autoencoder as vecsetx

ENCODERS = "vecsetx"

class TouchEncoder(nn.Module):
    """Global point-cloud touch encoder."""

    def __init__(
        self,
        encoder_name="vecsetx",
        model_name="learnable_vec1024x32_dim1024_depth24_nb",
        checkpoint_path=None,
    ):
        super().__init__()

        if encoder_name != "vecsetx":
            raise ValueError(f"Unknown touch encoder: {encoder_name}")

        model_factory = getattr(vecsetx, model_name)
        self.encoder = model_factory()

        if checkpoint_path is not None:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            state_dict = checkpoint.get("model", checkpoint)
            self.encoder.load_state_dict(state_dict, strict=True)

    def forward(self, points):
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(
                f"Expected [B, N, 3], received {tuple(points.shape)}"
            )

        return self.encoder.encode(points)["x"]