# License for the adapted CPU FPS reference below:
# BSD License
# 
# For PyTorch3D software
# 
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
#  * Neither the name Meta nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#    prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Small shared helpers for the copied CraftsMan and TripoSG encoders."""

from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint


def checkpoint(function, inputs, parameters, enabled):
    # Same call signature as CraftsMan; use PyTorch's maintained implementation.
    if enabled and torch.is_grad_enabled():
        return torch_checkpoint(function, *inputs, use_reentrant=False)
    return function(*inputs)


@torch.no_grad()
def farthest_point_indices(points, count, random_start=False):
    if count > points.shape[1]:
        raise ValueError(f"Need at least {count} points for encoder queries")
    if points.is_cuda:
        from pytorch3d.ops import sample_farthest_points
        return sample_farthest_points(points.float(), K=count, random_start_point=random_start)[1]

    # CPU reference adapted from PyTorch3D's sample_farthest_points_naive.
    # https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/sample_farthest_points.py
    # Copyright (c) Meta Platforms, Inc. and affiliates. BSD-3-Clause.
    # For small local checks; cluster runs use the CUDA implementation above.
    points = points.float()
    batch_size, size, _ = points.shape
    indices = torch.empty(batch_size, count, dtype=torch.long, device=points.device)
    distance = points.new_full((batch_size, size), float("inf"))
    selected = torch.randint(size, (batch_size,), device=points.device) if random_start else indices.new_zeros(batch_size)
    batch = torch.arange(batch_size, device=points.device)
    for index in range(count):
        indices[:, index] = selected
        squared = (points - points[batch, selected, None]).square().sum(-1)
        distance = torch.minimum(distance, squared)
        selected = distance.argmax(-1)
    return indices


def load_surface_encoder_weights(encoder, name, checkpoint_path=None):
    if checkpoint_path is None:
        from huggingface_hub import hf_hub_download
        if name == "craftsman":
            checkpoint_path = hf_hub_download(
                "craftsman3d/craftsman", "model.ckpt",
                revision="df4ddf7544cc2e75c5d24cb8605d8e91f0fa4abc",
            )
        else:
            checkpoint_path = hf_hub_download(
                "VAST-AI/TripoSG", "vae/diffusion_pytorch_model.safetensors",
                revision="2c1c516d22d58db486a058d98d31bb6177344e06",
            )
    checkpoint_path = Path(checkpoint_path).expanduser()
    if name == "craftsman":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        state = checkpoint.get("state_dict", checkpoint)
        if any(key.startswith("shape_model.") for key in state):
            state = {key.removeprefix("shape_model."): value for key, value in state.items()
                     if key.startswith("shape_model.")}
        state = {key: value for key, value in state.items() if key.startswith(("encoder.", "pre_kl."))}
    else:
        from safetensors import safe_open
        with safe_open(checkpoint_path, framework="pt", device="cpu") as saved:
            state = {key: saved.get_tensor(key) for key in saved.keys()
                     if key.startswith(("encoder.", "quant."))}
    # Only decoder weights are excluded. Every encoder key/shape must match exactly.
    encoder.load_state_dict(state, strict=True)
