#!/bin/bash
# Run from the repository root: bash experiments/coordinate_system/scripts/camera_frame_formal/submit.sh PREPARATION_DIR
set -e

PREP=$1
RUN=experiments/coordinate_system/outputs/camera_frame_formal/quick_fit_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"

for arm in object_stock camera_stock camera_shared; do
  sbatch --job-name="camera_frame_${arm}" --output="$RUN/${arm}.log" <<EOF
#!/bin/bash
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_qianqian_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -e
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python \\
  experiments/coordinate_system/scripts/camera_frame_formal/fit_camera_frame_gpu.py \\
  --preparation-dir "$PREP" \\
  --arm "$arm" \\
  --output-dir "$RUN/$arm" \\
  --steps 1000 \\
  --batch-size 4 \\
  --global-batch-size 4 \\
  --validate-every 100 \\
  --workers 4 \\
  --train-scope shape_cross_attention \\
  --visual-dropout 0.5
EOF
done

echo "Results and logs: $RUN"
