import torch
import torch.nn as nn

from sam3d_objects.model.layers.llama3.ff import FeedForward


def make_vecsetx():
    from .vecsetx import autoencoder as vecsetx
    return vecsetx.learnable_vec1024x32_dim1024_depth24_nb()


def make_craftsman():
    from .craftsman import CraftsManEncoder
    return CraftsManEncoder()


def make_triposg():
    from .triposg import TripoSGPointEncoder
    return TripoSGPointEncoder()


ENCODERS = {
    "vecsetx": {
        "constructor": make_vecsetx,
        "repo_id": "Zbalpha/VecSetX",
        "filename": (
            "learnable_vec1024x32_dim1024_depth24_sdf_nb/"
            "checkpoint-125.pth"
        ),
    },
    "craftsman": {"constructor": make_craftsman},
    "triposg": {"constructor": make_triposg},
}

# These are the only VecSetX parameters used by encoder.encode().
VECSETX_ENCODE_PARAMETER_PREFIXES = (
    "latents.",
    "point_embed.",
    "cross_attend_blocks.",
    "bottleneck.pre_bottleneck_proj.",
)

class TouchEncoder(nn.Module):
    def __init__(
        self,
        encoder_name="vecsetx",
        output_dim=1024,
        trainable=False,
        use_position=True,
        position_scale="raw",
        use_learn=False,
        pretrained=True,
        encoder_checkpoint=None,
    ):
        super().__init__()

        if encoder_name not in ENCODERS:
            raise ValueError(f"Unknown encoder {encoder_name!r},available encoders: {tuple(ENCODERS)}")

        config = ENCODERS[encoder_name]

        self.encoder_name = encoder_name
        self.output_dim = output_dim
        self.use_learn = bool(use_learn)
        self.pretrained = bool(pretrained)
        self.requires_normals = encoder_name != "vecsetx"
        self.encoder_checkpoint = str(encoder_checkpoint) if encoder_checkpoint is not None else None
        if not self.pretrained and (not trainable or self.use_learn):
            raise ValueError("Scratch initialization requires a trainable encoder without frozen decoder features")
        if self.requires_normals and self.use_learn:
            raise ValueError("Decoder features are only supported for VecSetX")
        if self.encoder_checkpoint and not self.pretrained:
            raise ValueError("An encoder checkpoint cannot be used with random initialization")
        self.use_position = bool(use_position)
        self.position_scale = position_scale
        if self.position_scale not in ("raw", "log"):
            raise ValueError(f"Unknown position scale {self.position_scale!r}")
        self.encoder = config["constructor"]()
        self.num_points = getattr(self.encoder, "num_inputs", None)

        if self.pretrained and self.requires_normals:
            from .surface_encoder_utils import load_surface_encoder_weights
            load_surface_encoder_weights(self.encoder, encoder_name, self.encoder_checkpoint)
        elif self.pretrained:
            checkpoint_path = self.encoder_checkpoint or "/n/home12/mwakeham/.cache/huggingface/hub/models--Zbalpha--VecSetX/snapshots/5fb84917189d2bee8392404f833f42ca5c067e0b/learnable_vec1024x32_dim1024_depth24_sdf_nb/checkpoint-125.pth"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("model", checkpoint)
            self.encoder.load_state_dict(state_dict, strict=True)

        latent_dim = (self.encoder.latent_dim if self.requires_normals
                      else self.encoder.bottleneck.pre_bottleneck_proj.out_features)
        if self.use_learn:
            latent_dim = self.encoder.bottleneck.post_bottleneck_proj.out_features
        self.output_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            FeedForward(
                dim=latent_dim,
                hidden_dim=4 * output_dim,
                output_dim=output_dim,
            ),
        )

        self.position_projection = FeedForward(
            dim=4,
            hidden_dim=4 * output_dim,
            output_dim=output_dim,
        )
        nn.init.zeros_(self.position_projection.w2.weight)
        self.position_projection.requires_grad_(self.use_position)

        self.touch_embedding = nn.Parameter(torch.empty(1, 1, output_dim))
        nn.init.normal_(self.touch_embedding, mean=0.0, std=1.0 / output_dim**0.5)

        self.set_trainable(trainable)

    def set_trainable(self, trainable):
        self.encoder_trainable = bool(trainable)

        self.encoder.requires_grad_(False)

        if self.encoder_trainable:
            for name, parameter in self.encoder.named_parameters():
                # These embeddings are shared by encode() and learn().
                if self.use_learn and name.startswith("latents."):
                    continue
                if self.requires_normals or name.startswith(VECSETX_ENCODE_PARAMETER_PREFIXES):
                    parameter.requires_grad_(True)
        self.encoder.train(self.training and self.encoder_trainable)

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen encoders' query sampling deterministic during adapter training.
        self.encoder.train(mode and self.encoder_trainable)
        return self

    def get_trainable_parameters(self):
        return (
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def get_config(self):
        config = {
            "encoder_name": self.encoder_name,
            "output_dim": self.output_dim,
            "trainable": self.encoder_trainable,
        }
        if not self.use_position:
            config["use_position"] = False
        elif self.position_scale == "log":
            config["position_scale"] = "log"
        if self.use_learn:
            config["use_learn"] = True
        if not self.pretrained:
            config["pretrained"] = False
        if self.encoder_checkpoint is not None:
            config["encoder_checkpoint"] = self.encoder_checkpoint
        return config

    def forward(self, points, point_mask=None):
        channels = 6 if self.requires_normals else 3
        if points.ndim != 3 or points.shape[-1] != channels:
            raise ValueError(f"Expected points shaped [B, N, {channels}], got {tuple(points.shape)}")

        if self.requires_normals:
            points, shifts, scales = self.prepare_surface(points, point_mask)
            with torch.set_grad_enabled(torch.is_grad_enabled() and self.encoder_trainable):
                tokens = self.encoder.encode(points)
        else:
            points, point_mask, shifts, scales = self.prepare_points(points, point_mask)
            tokens = self.encoder.encode(points, point_mask)["x"]
        if self.use_learn:
            tokens = self.encoder.learn(tokens)
        tokens = self.output_projection(tokens)
        if self.use_position:
            position_scale = scales.log() if self.position_scale == "log" else scales
            position = self.position_projection(
                torch.cat((shifts, position_scale), dim=-1)
            )
            tokens = tokens + position.unsqueeze(1)
        return tokens + self.touch_embedding

    def prepare_surface(self, surface, point_mask=None):
        if point_mask is not None and (point_mask.shape != surface.shape[:2] or not point_mask.all()):
            raise ValueError("CraftsMan/TripoSG require equally sized, unpadded full-surface clouds")
        if surface.shape[1] < self.encoder.num_latents or not torch.isfinite(surface).all():
            raise ValueError("Surface must be finite and contain at least as many points as encoder queries")
        points, normals = surface[..., :3], surface[..., 3:]
        if not torch.allclose(normals.norm(dim=-1), torch.ones_like(normals[..., 0]), atol=1e-4, rtol=0):
            raise ValueError("Surface normals must be unit vectors paired with the XYZ points")
        # Adapt camera-frame clouds to the [-1, 1] object cube used by the VAEs.
        # CraftsMan supplementary §1.1 / data/base.py. TripoSG exposes this domain
        # in scripts/inference_vae.py, but does not release its full training preprocessing.
        # Fit the sampled cloud, preserving aspect ratio and camera orientation.
        lower, upper = points.amin(dim=1), points.amax(dim=1)
        shifts = (lower + upper) / 2
        extent = (upper - lower).amax(dim=-1, keepdim=True)
        if (extent <= 0).any():
            raise ValueError("Surface must have a positive extent")
        scales = 2 / extent
        points = (points - shifts[:, None]) * scales[:, None]
        return torch.cat((points, normals), dim=-1), shifts, scales

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

        points, shifts, scales = self.normalize_points_for_vecsetx(points, point_mask)

        if self.num_points is None or point_count == self.num_points:
            return points, point_mask, shifts, scales

        from pytorch3d.ops import sample_farthest_points
        points, indices = sample_farthest_points(
            points,
            lengths=lengths,
            K=self.num_points,
            random_start_point=False,
        )

        return points, indices >= 0, shifts, scales

    def normalize_points_for_vecsetx(self, points, point_mask):
        normalized = torch.zeros_like(points)
        shifts = []
        scales = []

        for index in range(len(points)):
            surface = points[index, point_mask[index]]
            shift = (surface.max(dim=0).values + surface.min(dim=0).values) / 2
            surface = surface - shift
            distances = torch.linalg.vector_norm(surface, dim=1)
            radius = distances.max()
            if not torch.isfinite(radius) or radius <= 0:
                raise ValueError(f"Point cloud {index} must have a positive finite radius")
            scale = 1 / radius
            normalized[index, point_mask[index]] = surface * scale
            shifts.append(shift)
            scales.append(scale)

        return normalized, torch.stack(shifts), torch.stack(scales).unsqueeze(1)
