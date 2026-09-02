import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.ops import sample_farthest_points

from huggingface_hub import hf_hub_download

from .vecsetx import autoencoder as vecsetx

ENCODERS = {
    "vecsetx": {
        "constructor": vecsetx.learnable_vec1024x32_dim1024_depth24_nb,
        "repo_id": "Zbalpha/VecSetX",
        "filename": (
            "learnable_vec1024x32_dim1024_depth24_sdf_nb/"
            "checkpoint-125.pth"
        ),
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
        self.num_points = getattr(self.encoder, "num_inputs", None)

        checkpoint_path = hf_hub_download(repo_id=config["repo_id"], filename=config["filename"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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

        points, point_mask = self.prepare_points(points)
        return self.encoder.encode(points, point_mask)["x"]

    def prepare_points(self, points):
        batch_size, current_count, _ = points.shape

        point_mask = torch.ones(
            batch_size,
            current_count,
            dtype=torch.bool,
            device=points.device,
        )

        if self.num_points is None:
            return points, point_mask

        if current_count > self.num_points:
            points, _ = sample_farthest_points(
                points,
                K=self.num_points,
                random_start_point=False,
            )

            point_mask = torch.ones(
                batch_size,
                self.num_points,
                dtype=torch.bool,
                device=points.device,
            )

        elif current_count < self.num_points:
            padding = self.num_points - current_count

            points = F.pad(
                points,
                (0, 0, 0, padding),
                value=0,
            )

            point_mask = F.pad(
                point_mask,
                (0, padding),
                value=False,
            )

        return points, point_mask