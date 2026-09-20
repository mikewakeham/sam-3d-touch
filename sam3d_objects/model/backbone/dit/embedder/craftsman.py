# Encoder-only port of HKUST-SAIL/CraftsMan3D, f87d6a54ef7e818581b1c97a2aa962dc731306a0.
# Sources: craftsman/models/autoencoders/michelangelo_autoencoder.py and
# craftsman/models/transformers/{attention,perceiver_1d,utils}.py.
# Released v1.5 config: craftsman3d/craftsman@df4ddf7544cc2e75c5d24cb8605d8e91f0fa4abc.
# Decoder/framework code omitted; FPS uses the existing PyTorch3D dependency.
import math
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from einops import repeat
from .surface_encoder_utils import checkpoint, farthest_point_indices

def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class MLP(nn.Module):
    def __init__(self, *,
                 width: int,
                 init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
        qkv_bias: bool,
        qk_norm: bool = False,
        qkv_fuse: bool = True,
        norm_layer=nn.LayerNorm,
        use_flash: bool = False
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.qkv_fuse = qkv_fuse
        if qkv_fuse:
            self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        else:
            self.c_q = nn.Linear(width, width, bias=qkv_bias)
            self.c_k = nn.Linear(width, width, bias=qkv_bias)
            self.c_v = nn.Linear(width, width, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(
            heads=heads, 
            n_ctx=n_ctx, 
            width=width, 
            norm_layer=norm_layer, 
            qk_norm=qk_norm, 
            use_flash=use_flash
        )
        if qkv_fuse:
            init_linear(self.c_qkv, init_scale)
        else:
            init_linear(self.c_q, init_scale)
            init_linear(self.c_k, init_scale)
            init_linear(self.c_v, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        if self.qkv_fuse:
            x = self.c_qkv(x)
            x = checkpoint(self.attention.forward_qkv_fuse, (x,), (), True)
        else:
            q = self.c_q(x)
            k = self.c_k(x)
            v = self.c_v(x)
            x = checkpoint(self.attention, (q,k,v,), (), True)
        x = self.c_proj(x)
        return x


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, heads: int, n_ctx: int, width=None, qk_norm: bool = False, norm_layer=nn.LayerNorm, use_flash: bool = False):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx
        self.use_flash = use_flash

        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, q, k, v):
        bs, n_ctx, width = q.shape
        attn_ch = width // self.heads
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        q = q.view(bs, n_ctx, self.heads, -1)
        k = k.view(bs, n_ctx, self.heads, -1)
        v = v.view(bs, n_ctx, self.heads, -1)

        if self.use_flash:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(bs, n_ctx, -1)
        else:
            weight = torch.einsum(
                "bthc,bshc->bhts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            wdtype = weight.dtype
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
            out = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return out

    def forward_qkv_fuse(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)
        
        if self.use_flash:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(bs, n_ctx, -1)
        else:
            weight = torch.einsum(
                "bthc,bshc->bhts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            wdtype = weight.dtype
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
            out = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        use_flash: bool = False,
        use_checkpoint: bool = False
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.attn = MultiheadAttention(
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
            use_flash=use_flash
        )
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width)

    def _forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward(self, x: torch.Tensor):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)


class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        init_scale: float,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        qkv_fuse: bool = True, 
        use_flash: bool = False,
        n_data: Optional[int] = None,
        data_width: Optional[int] = None,
    ):
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.qkv_fuse = qkv_fuse
        if qkv_fuse:
            self.c_q = nn.Linear(width, width, bias=qkv_bias)
            self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias)
        else:
            self.c_q = nn.Linear(width, width, bias=qkv_bias)
            self.c_k = nn.Linear(width, width, bias=qkv_bias)
            self.c_v = nn.Linear(width, width, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadCrossAttention(
            heads=heads, n_data=n_data, width=width, norm_layer=norm_layer, qk_norm=qk_norm, use_flash=use_flash
        )
        if qkv_fuse:
            init_linear(self.c_q, init_scale)
            init_linear(self.c_kv, init_scale)
        else:
            init_linear(self.c_q, init_scale)
            init_linear(self.c_k, init_scale)
            init_linear(self.c_v, init_scale)

        init_linear(self.c_proj, init_scale)

    def forward(self, x, data):
        if self.qkv_fuse:
            x = self.c_q(x)
            data = self.c_kv(data)
            x = checkpoint(self.attention.forward_qkv_fuse, (x, data), (), True)
        else:
            q = self.c_q(x)
            k = self.c_k(data)
            v = self.c_v(data)
            x = checkpoint(self.attention, (q, k, v,), (), True)
        x = self.c_proj(x)
        return x


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(
        self, 
        *, 
        heads: int, 
        n_data: Optional[int] = None, 
        width=None, 
        norm_layer=nn.LayerNorm, 
        qk_norm: bool = False, 
        use_flash: bool = False
    ):

        super().__init__()
        self.heads = heads
        self.n_data = n_data
        self.use_flash = use_flash
        
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, q, k, v):
        _, n_ctx, _ = q.shape
        bs, n_data, width = k.shape
        attn_ch = width // self.heads
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        q = q.view(bs, n_ctx, self.heads, -1)
        k = k.view(bs, n_data, self.heads, -1)
        v = v.view(bs, n_data, self.heads, -1)

        if self.use_flash:
            
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(bs, n_ctx, -1)
        else:
            weight = torch.einsum(
                "bthc,bshc->bhts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            wdtype = weight.dtype
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
            out = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return out

    def forward_qkv_fuse(self, q, kv):
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.use_flash:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(bs, n_ctx, -1)
        else:
            weight = torch.einsum(
                "bthc,bshc->bhts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            wdtype = weight.dtype
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
            out = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return out


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_data: Optional[int] = None,
        width: int,
        heads: int,
        data_width: Optional[int] = None,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        use_flash: bool = False
    ):
        super().__init__()

        if data_width is None:
            data_width = width

        self.attn = MultiheadCrossAttention(
            n_data=n_data,
            width=width,
            heads=heads,
            data_width=data_width,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            use_flash=use_flash,
        )
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(data_width)
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_3 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x

class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        use_flash: bool = False,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    use_flash=use_flash,
                    use_checkpoint=use_checkpoint
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x

class FourierEmbedder(nn.Module):
    def __init__(self,
                 num_freqs: int = 6,
                 logspace: bool = True,
                 input_dim: int = 3,
                 include_input: bool = True,
                 include_pi: bool = True) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                num_freqs,
                dtype=torch.float32
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (num_freqs - 1),
                num_freqs,
                dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x


class PerceiverCrossAttentionEncoder(nn.Module):
    def __init__(self,
                 use_downsample: bool,
                 num_latents: int,
                 embedder: FourierEmbedder,
                 point_feats: int,
                 embed_point_feats: bool,
                 width: int,
                 heads: int,
                 layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 use_ln_post: bool = False,
                 use_flash: bool = False,
                 use_checkpoint: bool = False,
                 use_multi_reso: bool = False,
                 resolutions: list = [],
                 sampling_prob: list = []):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents
        self.use_downsample = use_downsample
        self.embed_point_feats = embed_point_feats
        self.use_multi_reso = use_multi_reso
        self.resolutions = resolutions
        self.sampling_prob = sampling_prob

        if not self.use_downsample:
            self.query = nn.Parameter(torch.randn((num_latents, width)) * 0.02)

        self.embedder = embedder
        if self.embed_point_feats:
            self.input_proj = nn.Linear(self.embedder.out_dim * 2, width)
        else:
            self.input_proj = nn.Linear(self.embedder.out_dim + point_feats, width)

        self.cross_attn = ResidualCrossAttentionBlock(
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
        )

        self.self_attn = Perceiver(
            n_ctx=num_latents,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_checkpoint=False
        )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width)
        else:
            self.ln_post = None

    def _forward(self, pc, feats):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:

        """

        bs, N, D = pc.shape
        
        data = self.embedder(pc)
        if feats is not None:
            if self.embed_point_feats:
                feats = self.embedder(feats)
            data = torch.cat([data, feats], dim=-1)
        data = self.input_proj(data)

        if self.use_downsample:
            indices = farthest_point_indices(pc, self.num_latents, random_start=self.training)
            query = data.gather(1, indices[..., None].expand(-1, -1, data.shape[-1]))
        else:
            query = self.query
            query = repeat(query, "m c -> b m c", b=bs)

        latents = self.cross_attn(query, data)
        latents = self.self_attn(latents)

        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents

    def forward(self, pc: torch.FloatTensor, feats: Optional[torch.FloatTensor] = None):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:
            dict
        """

        return checkpoint(self._forward, (pc, feats), self.parameters(), self.use_checkpoint)


class CraftsManEncoder(nn.Module):
    def __init__(self, width=768, heads=12, layers=8, num_latents=768, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.embedder = FourierEmbedder(num_freqs=8, include_pi=False)
        self.encoder = PerceiverCrossAttentionEncoder(
            use_downsample=True, num_latents=num_latents, embedder=self.embedder,
            point_feats=3, embed_point_feats=False, width=width, heads=heads, layers=layers,
            init_scale=0.25 * math.sqrt(1.0 / width), qkv_bias=False,
            use_ln_post=True, use_flash=True, use_checkpoint=True,
        )
        self.pre_kl = nn.Linear(width, latent_dim * 2)

    def encode(self, surface):
        latents = self.encoder(surface[..., :3], surface[..., 3:])
        # Upstream encode(surface, sample_posterior=False): posterior.mode().
        return self.pre_kl(latents).chunk(2, dim=-1)[0]
