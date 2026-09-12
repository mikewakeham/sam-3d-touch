#!/bin/bash
#SBATCH --job-name=full_surface_frame
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_qianqian_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

# HISTORICAL ONLY: recommendation withdrawn after user reported prior oracle results.
# This is not a requested next experiment.
# Matches jobs/stage1_full_surface_full_cross_attention.sh.
# Default adds the known camera-to-object transform. Pass camera for a fresh control.
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true
arm="${1:-oracle}"
frame_args=(--no-touch-position)
case "$arm" in
 oracle) frame_args+=(--oracle-point-frame) ;;
 camera) ;;
 *) echo 'Expected oracle or camera' >&2; exit 2 ;;
esac
run_dir="outputs/conditioning_investigation/full_dataset_frames/stage1_full_surface_${arm}_full_cross_attention_${SLURM_JOB_ID:-manual}"
if [ -e "$run_dir" ]; then
 echo "Refusing to reuse existing output directory: $run_dir" >&2
 exit 1
fi
/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python - "$run_dir" "$arm" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
paths = ['train.py', 'dataloader.py',
         'sam3d_objects/model/backbone/dit/embedder/touch.py',
         'configs/data_full_surface.yaml', 'checkpoints/hf/pipeline.yaml',
         'checkpoints/hf/ss_generator.yaml',
         'doc/conditioning_investigation_2026-09-11/run_full_dataset_frame.sh']
record = {'arm': sys.argv[2],
          'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
(out / 'investigation_provenance.json').write_text(json.dumps(record, indent=2) + '\n')
PY
/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/torchrun --standalone --nproc_per_node=4 train.py \
 --pipeline-config checkpoints/hf/pipeline.yaml \
 --data-config configs/data_full_surface.yaml \
 --output-dir "$run_dir" \
 --batch-size 4 --workers 8 --val-workers 2 --epochs 20 \
 --learning-rate 1e-4 --cross-attention-learning-rate 1e-5 \
 --gradient-clip 1 --precision bf16 \
 --cross-attention-scope full \
 "${frame_args[@]}"
