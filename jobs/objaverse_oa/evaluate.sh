#!/bin/bash
#SBATCH --job-name=oa_evaluate
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_qianqian_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/logs/objaverse_oa/%x-%j.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/logs/objaverse_oa/%x-%j.err

set -e
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/torchrun --standalone --nproc_per_node=4 -m evaluation.evaluate \
  --run-dirs \
  outputs/objaverse_oa/pointmap_camframe \
  outputs/objaverse_oa/pointmap_touch_32x256_joint_camframe \
  outputs/objaverse_oa/pointmap_surface_camframe \
  --data-root /n/lab_storage/qianqian_lab/Lab/mwakeham/datasets/objaverse_oa/sam3d_8views_sampled \
  --pipeline-config checkpoints/hf/pipeline.yaml \
  --selection-data-config configs/data_objaverse_oa_8views_sampled_full_surface_camframe.yaml \
  --output-dir outputs/objaverse_oa/evaluation_camframe \
  --max-samples 100 \
  --split val \
  --selection random \
  --selection-seed 29 \
  --seed 29 \
  --workers 4 \
  --inference-steps 25 \
  --stage2-inference-steps 25 "$@"
