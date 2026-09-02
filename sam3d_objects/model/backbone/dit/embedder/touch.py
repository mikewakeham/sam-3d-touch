import torch
import torch.nn as nn

from .vecsetx import autoencoder as vecsetx

ENCODERS = {
    "vecsetx": {
        "constructor": vecsetx.learnable_vec1024x32_dim1024_depth24_nb,
        "checkpoint": "/path/to/learnable_vec1024x32_dim1024_depth24_sdf_nb.pth",
    },
}

# These are the only VecSetX parameters used by encoder.encode().
VECSETX_ENCODE_PARAMETER_PREFIXES = (
    "latents.",
    "point_embed.",
    "cross_attend_blocks.",
    "bottleneck.pre_bottleneck_proj.",
)

class TouchEncoder(nn.Module):
    def __init__(self, encoder_name="vecsetx", trainable=False):
        super().__init__()

        if encoder_name not in ENCODERS:
            raise ValueError(f"Unknown encoder {encoder_name!r},available encoders: {tuple(ENCODERS)}")

        config = ENCODERS[encoder_name]

        self.encoder_name = encoder_name
        self.encoder = config["constructor"]()

        checkpoint = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        self.encoder.load_state_dict(state_dict, strict=True)

        self.set_trainable(trainable)

    def set_trainable(self, trainable):
        self.encoder_trainable = bool(trainable)

        self.encoder.requires_grad_(False)

        if self.encoder_trainable:
            for name, parameter in self.encoder.named_parameters():
                if name.startswith(VECSETX_ENCODE_PARAMETER_PREFIXES):
                    parameter.requires_grad_(True)

    def get_trainable_parameters(self):
        return (
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def get_config(self):
        return {
            "encoder_name": self.encoder_name,
            "trainable": self.encoder_trainable,
        }

    def forward(self, points):
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"Expected points shaped [B, N, 3], got {tuple(points.shape)}")

        return self.encoder.encode(points)["x"]