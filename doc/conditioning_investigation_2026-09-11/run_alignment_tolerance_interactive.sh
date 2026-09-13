(
set -euo pipefail
cd /n/holylabs/qianqian_lab/Lab/mwakeham/visuotactile-objects/sam-3d-touch
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LIDRA_SKIP_INIT=true
probe_python=/n/holylabs/qianqian_lab/Lab/mwakeham/.conda/envs/sam3d-objects/bin/python
probe_root=outputs/conditioning_investigation/alignment_tolerance/manual
test ! -e "$probe_root"
mkdir -p "$probe_root"
probe_gpu_count=$("$probe_python" -c 'import torch; print(min(4, torch.cuda.device_count()))')
test "$probe_gpu_count" -gt 0
for ((probe_base=0; probe_base<4; probe_base+=probe_gpu_count)); do
  probe_pids=()
  for ((probe_offset=0; probe_offset<probe_gpu_count && probe_base+probe_offset<4; probe_offset++)); do
    probe_shard=$((probe_base+probe_offset))
    printf 'Starting shard %s on visible GPU %s; log: %s/%s.log\n' "$probe_shard" "$probe_offset" "$probe_root" "$probe_shard"
    "$probe_python" doc/conditioning_investigation_2026-09-11/probe_alignment_tolerance_gpu.py \
      --shard "$probe_shard" --device-index "$probe_offset" \
      --output-dir "$probe_root/$probe_shard" \
      > "$probe_root/$probe_shard.log" 2>&1 &
    probe_pids+=("$!")
  done
  probe_failed=0
  for probe_pid in "${probe_pids[@]}"; do
    wait "$probe_pid" || probe_failed=1
  done
  if test "$probe_failed" -ne 0; then
    tail -n 30 "$probe_root"/*.log
    exit 1
  fi
done
"$probe_python" doc/conditioning_investigation_2026-09-11/analyze_alignment_tolerance.py \
  "$probe_root" --output "$probe_root/analysis.json"
)
