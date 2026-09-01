import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if self.probability == 0.0 or not self.training:
            return x
        keep_probability = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        keep = x.new_empty(shape).bernoulli_(keep_probability)
        return x * keep / keep_probability


class MLP(nn.Module):
    def __init__(self, width, ratio=4.0):
        super().__init__()
        hidden_width = int(width * ratio)
        self.fc1 = nn.Linear(width, hidden_width)
        self.fc2 = nn.Linear(hidden_width, width)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class SelfAttention(nn.Module):
    def __init__(self, width, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_width = width // num_heads
        self.qkv = nn.Linear(width, 3 * width, bias=True)
        self.proj = nn.Linear(width, width)

    def forward(self, x, point_mask=None):
        batch_size, sequence_length, width = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, sequence_length, 3, self.num_heads, self.head_width
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention_mask = point_mask[:, None, None, :] if point_mask is not None else None
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        attended = attended.transpose(1, 2).reshape(batch_size, sequence_length, width)
        return self.proj(attended)


class LocalCPE(nn.Module):
    """Continuous local positional encoding replacing PTv3's sparse 3D CPE."""

    def __init__(self, width):
        super().__init__()
        self.position_mlp = nn.Sequential(
            nn.Linear(3, width // 2), nn.GELU(), nn.Linear(width // 2, width)
        )
        self.proj = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x, xyz, neighbor_indices, neighbor_mask):
        batch_indices = torch.arange(x.shape[0], device=x.device)[:, None, None]
        neighbor_features = x[batch_indices, neighbor_indices]
        neighbor_xyz = xyz[batch_indices, neighbor_indices]
        relative_xyz = neighbor_xyz - xyz[:, :, None, :]
        relative_xyz = relative_xyz.masked_fill(~neighbor_mask[..., None], 0)
        local_features = neighbor_features + self.position_mlp(relative_xyz)
        local_features = local_features * neighbor_mask[..., None]
        local_features = local_features.sum(dim=2) / neighbor_mask.sum(dim=2, keepdim=True).clamp_min(1)
        return x + self.norm(self.proj(local_features))


class PointTransformerBlock(nn.Module):
    """PTv3 block order: CPE, pre-norm attention, pre-norm MLP."""

    def __init__(self, width, num_heads, mlp_ratio, drop_path):
        super().__init__()
        self.cpe = LocalCPE(width)
        self.norm1 = nn.LayerNorm(width)
        self.attention = SelfAttention(width, num_heads)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = MLP(width, mlp_ratio)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, xyz, point_mask, neighbor_indices, neighbor_mask):
        x = self.cpe(x, xyz, neighbor_indices, neighbor_mask)
        x = x + self.drop_path(self.attention(self.norm1(x), point_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x * point_mask[..., None]


class ContactTransformerBlock(nn.Module):
    def __init__(self, width, num_heads, mlp_ratio, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = SelfAttention(width, num_heads)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = MLP(width, mlp_ratio)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.attention(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TouchEncoder(nn.Module):
    """Encode camera-coordinate surface neighborhoods into touch tokens."""

    def __init__(
        self,
        output_dim,
        width=128,
        point_depth=4,
        contact_depth=2,
        num_heads=8,
        mlp_ratio=4.0,
        num_neighbors=16,
        max_drop_path=0.1,
        tokens_per_contact=1,
        architecture_version="center_v1",
    ):
        super().__init__()
        if architecture_version not in {"center_v1", "query_pool_v1"}:
            raise ValueError(f"Unknown touch encoder architecture: {architecture_version}")
        if architecture_version == "center_v1" and tokens_per_contact != 1:
            raise ValueError("center_v1 requires tokens_per_contact=1")
        if architecture_version == "query_pool_v1" and tokens_per_contact < 2:
            raise ValueError("query_pool_v1 requires at least 2 tokens per contact")

        self.output_dim = output_dim
        self.width = width
        self.point_depth = point_depth
        self.contact_depth = contact_depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_neighbors = num_neighbors
        self.max_drop_path = max_drop_path
        self.tokens_per_contact = tokens_per_contact
        self.architecture_version = architecture_version
        self.absolute_embedding = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.relative_embedding = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.input_norm = nn.LayerNorm(width)

        total_depth = point_depth + contact_depth
        drop_paths = torch.linspace(0, max_drop_path, total_depth).tolist()
        self.point_blocks = nn.ModuleList(
            PointTransformerBlock(width, num_heads, mlp_ratio, drop_paths[index])
            for index in range(point_depth)
        )
        self.contact_blocks = nn.ModuleList(
            ContactTransformerBlock(width, num_heads, mlp_ratio, drop_paths[point_depth + index])
            for index in range(contact_depth)
        )
        if architecture_version == "query_pool_v1":
            self.summary_queries = nn.Parameter(
                torch.empty(1, tokens_per_contact - 1, width)
            )
            self.summary_query_norm = nn.LayerNorm(width)
            self.summary_point_norm = nn.LayerNorm(width)
            self.summary_attention = nn.MultiheadAttention(
                width, num_heads, batch_first=True
            )
            self.summary_output_norm = nn.LayerNorm(width)
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, output_dim)
        self.modality_embedding = nn.Parameter(torch.empty(1, 1, output_dim))

        self.apply(self._initialize_module)
        if architecture_version == "query_pool_v1":
            nn.init.normal_(self.summary_queries, std=width**-0.5)
        nn.init.normal_(self.modality_embedding, std=output_dim**-0.5)

    def get_config(self):
        return {
            "output_dim": self.output_dim,
            "width": self.width,
            "point_depth": self.point_depth,
            "contact_depth": self.contact_depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "num_neighbors": self.num_neighbors,
            "max_drop_path": self.max_drop_path,
            "tokens_per_contact": self.tokens_per_contact,
            "architecture_version": self.architecture_version,
        }

    @staticmethod
    def _initialize_module(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _neighbors(self, relative_xyz, point_mask):
        neighbor_count = min(self.num_neighbors, relative_xyz.shape[1])
        with torch.no_grad():
            distances = torch.cdist(relative_xyz.float(), relative_xyz.float())
            distances = distances.masked_fill(~point_mask[:, None, :], float("inf"))
            indices = distances.topk(neighbor_count, dim=-1, largest=False).indices
            batch_indices = torch.arange(point_mask.shape[0], device=point_mask.device)[:, None, None]
            return indices, point_mask[batch_indices, indices]

    def forward(self, xyz, point_mask=None):
        if not torch.isfinite(xyz).all():
            raise ValueError("Touch input contains non-finite coordinates")

        batch_size, contact_count, point_count, _ = xyz.shape
        if point_mask is None:
            point_mask = torch.ones(xyz.shape[:-1], dtype=torch.bool, device=xyz.device)
        centers = xyz[:, :, :1]
        relative_xyz = xyz - centers

        x = self.absolute_embedding(xyz) + self.relative_embedding(relative_xyz)
        x = self.input_norm(x)

        x = x.reshape(batch_size * contact_count, point_count, -1)
        flat_relative_xyz = relative_xyz.reshape(batch_size * contact_count, point_count, 3)
        point_mask = point_mask.reshape(batch_size * contact_count, point_count)
        x = x * point_mask[..., None]
        neighbor_indices, neighbor_mask = self._neighbors(flat_relative_xyz, point_mask)
        for block in self.point_blocks:
            x = block(x, flat_relative_xyz, point_mask, neighbor_indices, neighbor_mask)

        # The data loader guarantees that point zero is the physical contact center.
        center = x[:, :1]
        if self.architecture_version == "query_pool_v1":
            queries = self.summary_queries.to(x).expand(x.shape[0], -1, -1)
            points = self.summary_point_norm(x)
            summaries, _ = self.summary_attention(
                self.summary_query_norm(queries),
                points,
                points,
                key_padding_mask=~point_mask,
                need_weights=False,
            )
            summaries = self.summary_output_norm(queries + summaries)
            x = torch.cat((center, summaries), dim=1)
        else:
            x = center

        x = x.reshape(batch_size, contact_count * self.tokens_per_contact, -1)
        for block in self.contact_blocks:
            x = block(x)

        x = self.output_projection(self.output_norm(x))
        return x + self.modality_embedding
