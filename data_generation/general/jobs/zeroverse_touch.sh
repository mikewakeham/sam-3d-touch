#!/bin/bash
# Run inside an existing CPU allocation; rendering and GPU encoding are already done.

set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-data/bin/python -u data_generation/general/make_touch_data.py \
    --data-root /n/netscratch/qianqian_lab/Lab/mwakeham/zeroverse_5000_sam3d_8views \
    --workers 12 \
    "$@"
