#!/bin/bash
#SBATCH --job-name=sam3d-data
#SBATCH --account=kempner_qianqian_lab
#SBATCH --partition=kempner_rtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=016:00:00
#SBATCH --output=sam3d-data-%j.out
#SBATCH --error=sam3d-data-%j.err

set -euo pipefail
module purge
# PyTorch cu128 wheels and Blender carry their own CUDA runtimes.
# Leave the existing sam3d-objects environment untouched.
source activate "${DATA_ENV:-sam3d-data}"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Submit from sam-3d-touch; override input/output through environment variables.
: "${OBJECTS_ROOT:?Set OBJECTS_ROOT to the input object directory}"
: "${DATA_ROOT:?Set DATA_ROOT to a separate output dataset directory}"
: "${BLENDER_BIN:?Set BLENDER_BIN to a Blender 4.5 LTS executable}"
python -m pip check
"$BLENDER_BIN" --version
nvidia-smi --query-gpu=name,driver_version --format=csv
python - <<'PY'
import torch
print(f"PyTorch {torch.__version__}, bundled CUDA {torch.version.cuda}", flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA device available in the data environment")
for index in range(torch.cuda.device_count()):
    value = torch.ones(1, device=f"cuda:{index}") + 1
    assert value.item() == 2
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}; CUDA kernel passed", flush=True)
PY
python -u data_generation/general/make_data.py \
    --objects-root "$OBJECTS_ROOT" \
    --data-root "$DATA_ROOT" \
    --pattern "${OBJECT_PATTERN:-**/*.glb}" \
    --blender "$BLENDER_BIN" \
    --render-device OPTIX \
    --workers 6 \
    --render-workers-per-gpu 2 \
    "$@"
