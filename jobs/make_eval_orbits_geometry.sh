#!/bin/bash
#SBATCH --job-name=eval_orbits_geometry
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_qianqian_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/logs/%x-%j.out
#SBATCH --error=/n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch/logs/%x-%j.err

set -e
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

sample_ids=(
  05d9d1c2ae9d4edf8f8c7d96ae2f34aa_003
  0716cddd4b14421fb4db6eaedf59375f_015
  0838b73f246f4768aa0671e7b2ba7c30_004
  1c89a18e18074a2cbf67a6d26d8d0e11_000
  1cf32d0ef332402c81cdf8ea6d6030b3_004
  1f4fa3b947474d4ead7d9d054a8032cd_004
  2bdc5f50dd19477c8c472acbdd08692d_003
  364e260a1bf942e9b3cde9874621cd13_004
  6141e7ddae4e4e67bfc522b586c0c551_005
  645136cc19154a3d8e045bf641a3b8d8_004
  68a31a8eb07d4b84aeb3b13484d4c9e8_003
  696101f22bf84430987160c67ea5d762_009
  6f05bcf4cd4c42218cb7abca0d1e492f_003
  9335eebb9e3f4cdebe30fbe0782ad8ad_004
  a61f6441d23e4e86a3cb00e2a3840eb3_001
  a657f8db0ba34262b2a1dc4338ef7630_000
  beebcf86bb35455aaaa18b8de071daac_003
  d69aea51817f4409a2fcd6f1f68c45e6_007
  d87e82071e24487481bf90c8f2682fb6_004
  df2eaa6838734e05b97ead25a39c9be0_004
  e03c939a9d9b4e689f2e15707b05c0d4_010
  e610ab728a124989882c209d764f046d_002
  eddf3774ba8040409197c17ba86a2ab0_006
)

for sample_id in "${sample_ids[@]}"
do
  /n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python make_eval_orbit.py \
    --evaluation-dir outputs/evaluation \
    --output-dir outputs/evaluation/orbits_camera \
    --sample-id "$sample_id" \
    --conditions \
    official \
    decoded_gt \
    stage1_image_full_cross_attention \
    stage1_full_surface_full_cross_attention \
    stage1_image_no_pointmap_full_cross_attention \
    stage1_full_surface_no_pointmap_full_cross_attention \
    --modes mesh voxel \
    --frames 120 \
    --fps 20 \
    --gif-fps 10
done
