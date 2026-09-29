#!/bin/bash
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_qianqian_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -euo pipefail

# Launch with bash; choose the Slurm name and logs before submitting this script.
# --render is passed internally to distinguish the submitted worker from the launcher.
if [[ "${1:-}" != --render ]]; then
  orbit_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  cd "$orbit_script_dir/.."
  evaluation_dir=""
  orbit_args=("$@")
  for ((i=0; i<${#orbit_args[@]}; i++)); do
    case "${orbit_args[i]}" in
      --evaluation-dir) evaluation_dir="${orbit_args[i+1]:-}" ;;
      --evaluation-dir=*) evaluation_dir="${orbit_args[i]#*=}" ;;
    esac
  done
  if [[ -z "$evaluation_dir" || ! -d "$evaluation_dir" ]]; then
    echo "Usage: bash jobs/make_eval_orbits.sh --evaluation-dir EXISTING_DIR [render options]" >&2
    exit 1
  fi
  evaluation_dir="$(cd -- "$evaluation_dir" && pwd)"
  orbit_job_name="$(basename -- "$(dirname -- "$evaluation_dir")")-$(basename -- "$evaluation_dir")"
  orbit_job_name="$(printf '%s' "$orbit_job_name" | tr -c 'A-Za-z0-9_.-' '_')"
  mkdir -p logs/orbits
  exec sbatch --job-name="$orbit_job_name" \
    --chdir="$PWD" \
    --output="$PWD/logs/orbits/%x-%j.out" \
    --error="$PWD/logs/orbits/%x-%j.err" \
    "$orbit_script_dir/make_eval_orbits.sh" --render "$@"
fi
shift

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python -m evaluation.make_orbit \
  --all-samples \
  --max-samples 10 \
  --with-inputs \
  --modes mesh voxel \
  --blender /n/holylabs/qianqian_lab/Lab/mwakeham/blender/blender-4.5.9-linux-x64/blender "$@"
