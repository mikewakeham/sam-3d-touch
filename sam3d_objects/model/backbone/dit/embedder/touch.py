import torch
import torch.nn as nn

from pytorch3d.ops import sample_farthest_points

from huggingface_hub import hf_hub_download

from .vecsetx import autoencoder as vecsetx
from sam3d_objects.model.layers.llama3.ff import FeedForward

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
    def __init__(self, encoder_name="vecsetx", output_dim=1024, trainable=False):
        super().__init__()

        if encoder_name not in ENCODERS:
            raise ValueError(f"Unknown encoder {encoder_name!r},available encoders: {tuple(ENCODERS)}")

        config = ENCODERS[encoder_name]

        self.encoder_name = encoder_name
        self.output_dim = output_dim
        self.encoder = config["constructor"]()
        self.num_points = getattr(self.encoder, "num_inputs", None)

        checkpoint_path = hf_hub_download(repo_id=config["repo_id"], filename=config["filename"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        self.encoder.load_state_dict(state_dict, strict=True)

        latent_dim = self.encoder.bottleneck.pre_bottleneck_proj.out_features
        self.output_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            FeedForward(
                dim=latent_dim,
                hidden_dim=4 * output_dim,
                output_dim=output_dim,
            ),
        )

        self.touch_embedding = nn.Parameter(torch.empty(1, 1, output_dim))
        nn.init.normal_(self.touch_embedding, mean=0.0, std=1.0 / output_dim**0.5)

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
            "output_dim": self.output_dim,
            "trainable": self.encoder_trainable,
        }

    def forward(self, points, point_mask=None):
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"Expected points shaped [B, N, 3], got {tuple(points.shape)}")

        points, point_mask = self.prepare_points(points, point_mask)
        tokens = self.encoder.encode(points, point_mask)["x"]
        tokens = self.output_projection(tokens)
        return tokens + self.touch_embedding

    def prepare_points(self, points, point_mask=None):
        batch_size, point_count, _ = points.shape

        if point_mask is None:
            point_mask = torch.ones(batch_size, point_count, dtype=torch.bool, device=points.device)
        else:
            if point_mask.shape != points.shape[:2]:
                raise ValueError("Point mask shape does not match points")

            point_mask = point_mask.to(device=points.device,dtype=torch.bool)

        if (point_mask[:, 1:] & ~point_mask[:, :-1]).any():
            raise ValueError("point_mask must be packed to the front")

        lengths = point_mask.sum(dim=1)

        if (lengths == 0).any():
            raise ValueError("Point cloud contains no valid points")

        if self.num_points is None or point_count == self.num_points:
            return points, point_mask

        points, indices = sample_farthest_points(
            points,
            lengths=lengths,
            K=self.num_points,
            random_start_point=False,
        )

        return points, indices >= 0