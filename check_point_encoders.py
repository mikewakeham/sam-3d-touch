"""Check installed dependencies and full-size encoder forwards before a cluster run."""
import argparse
from importlib.metadata import version
import os
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import torch
from packaging.requirements import Requirement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--pretrained", action="store_true", help="Download/cache and strictly load the official weights")
    parser.add_argument("--backward", action="store_true", help="Also exercise trainable encoders and check gradients")
    parser.add_argument("--bf16", action="store_true", help="Exercise the training job's bfloat16 autocast")
    args = parser.parse_args()

    print(f"Torch {torch.__version__}; CUDA build {torch.version.cuda}; device {args.device}", flush=True)
    requirements = Path(__file__).with_name("requirements.point-encoders.txt").read_text().splitlines()
    for line in requirements:
        if not line.strip() or line.startswith("#"):
            continue
        requirement = Requirement(line)
        installed = version(requirement.name)
        if installed not in requirement.specifier:
            raise RuntimeError(f"Installed {requirement.name}=={installed}; expected {requirement}")
        print(f"{requirement.name} {installed}", flush=True)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable; run --device cuda inside a GPU allocation")
        from pytorch3d.ops import sample_farthest_points
        print(f"PyTorch3D {version('pytorch3d')}; GPU {torch.cuda.get_device_name()}", flush=True)
    else:
        torch.set_num_threads(2)
        print("CPU check only; CUDA/PyTorch3D kernels need --device cuda on a GPU node.", flush=True)

    from sam3d_objects.model.backbone.dit.embedder.touch import TouchEncoder
    for name, count, tokens in (("craftsman", 16384, 768), ("triposg", 20480, 2048)):
        torch.manual_seed(29)
        points = torch.rand(1, count, 3, device=args.device) * 2 - 1
        normals = torch.nn.functional.normalize(torch.randn_like(points), dim=-1)
        model = TouchEncoder(name, pretrained=args.pretrained,
                             trainable=args.backward or not args.pretrained, use_position=False).to(args.device)
        model.train(args.backward)
        with torch.set_grad_enabled(args.backward), torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.bf16):
            output = model(torch.cat((points, normals), dim=-1))
            if output.shape != (1, tokens, 1024) or not torch.isfinite(output).all():
                raise RuntimeError(f"{name}: invalid output")
            loss = output.float().square().mean()
        if args.backward:
            loss.backward()
            for key, parameter in model.named_parameters():
                if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f"{name}: missing/nonfinite gradient for {key}")
        print(f"{name}: {count} XYZ+normal points -> {tuple(output.shape)}; OK", flush=True)
        del model, output, loss


if __name__ == "__main__":
    main()
