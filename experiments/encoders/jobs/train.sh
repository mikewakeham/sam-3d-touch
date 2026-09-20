#!/bin/bash
#SBATCH --job-name=encoder_test
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_qianqian_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/experiments/encoders/logs/%x-%j.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/experiments/encoders/logs/%x-%j.err

set -e
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# sbatch experiments/encoders/jobs/train.sh craftsman_frozen overfit
# sbatch experiments/encoders/jobs/train.sh triposg_scratch small
variant=${1:?Specify image, vecsetx_frozen, craftsman_frozen, triposg_frozen, or *_scratch}
stage=${2:?Specify overfit or small}
shift 2
case "$variant" in
  image) flags=(--no-touch) ;;
  vecsetx_frozen|craftsman_frozen|triposg_frozen)
    flags=(--point-encoder "${variant%_frozen}") ;;
  vecsetx_scratch|craftsman_scratch|triposg_scratch)
    flags=(--point-encoder "${variant%_scratch}" --train-point-encoder --point-encoder-from-scratch) ;;
  *) echo "Unknown variant: $variant" >&2; exit 1 ;;
esac
case "$stage" in
  overfit) steps=200; epochs=200 ;;
  small) steps=1024; epochs=8 ;;
  *) echo "Unknown stage: $stage" >&2; exit 1 ;;
esac

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/torchrun --standalone --nproc_per_node=4 experiments/encoders/run.py \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --data-config "experiments/encoders/data/$stage.yaml" \
  --output-dir "experiments/encoders/outputs/${stage}_${variant}" \
  --batch-size 4 \
  --workers 8 \
  --val-workers 2 \
  --epochs "$epochs" \
  --max-steps "$steps" \
  --learning-rate 1e-4 \
  --cross-attention-learning-rate 1e-5 \
  --log-every 10 \
  --precision bf16 \
  --no-touch-position \
  --train-scope shape_cross_attention "${flags[@]}" "$@"
