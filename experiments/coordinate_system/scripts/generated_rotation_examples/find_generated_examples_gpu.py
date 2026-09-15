"""Generate actual Stage-1 latents; shortlist rotation-related mismatches. No decoding."""
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from experiments.coordinate_system.scripts.generated_rotation_examples.candidate_selection import Shortlist


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--checkpoint', default='best.pt')
    p.add_argument('--data-config', type=Path, default=REPO/'configs/data_full_surface.yaml')
    p.add_argument('--pipeline-config', type=Path, default=REPO/'checkpoints/hf/pipeline.yaml')
    p.add_argument('--encoder-checkpoint', type=Path, default=REPO/'checkpoints/hf/ss_encoder.ckpt')
    p.add_argument('--val-objects', type=int, default=32)
    p.add_argument('--views', type=int, default=4)
    p.add_argument('--draws', type=int, default=2)
    p.add_argument('--top', type=int, default=12)
    p.add_argument('--seed', type=int, default=29)
    p.add_argument('--inference-steps', type=int, default=25)
    p.add_argument('--cfg-strength', type=float, default=0.)
    p.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.top < 1:
        p.error('--top must be positive')
    args.run_dirs = [args.run_dir]
    args.train_objects = 0
    args.wrong_surface = False
    shortlist = Shortlist(args.top)
    from experiments.coordinate_system.scripts.rotation_loss.evaluate_stage1_latents_gpu import run
    run(args, on_prediction=shortlist.add, endpoints_only=True)
    report = shortlist.save(args.output_dir/'candidates')
    print(f"Saved {len(report['examples'])} candidates from {report['total_predictions']} predictions; no decoding.")


if __name__ == '__main__':
    main()
