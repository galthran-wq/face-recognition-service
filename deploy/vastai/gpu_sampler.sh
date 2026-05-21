#!/usr/bin/env bash
# Sample nvidia-smi for a single GPU at 250 ms cadence; print avg/p50/p95/peak.
# Usage:  ./gpu_sampler.sh <gpu_index> <duration_seconds>
set -euo pipefail
gpu="${1:-0}"
duration="${2:-30}"

samples_file=$(mktemp)
trap 'rm -f "$samples_file"' EXIT

end=$(( $(date +%s) + duration ))
while [[ $(date +%s) -lt $end ]]; do
  nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw \
    --format=csv,noheader,nounits -i "$gpu" >>"$samples_file" 2>/dev/null
  sleep 0.25
done

python3 - "$samples_file" <<'PY'
import sys
util_gpu, util_mem, mem_used, power = [], [], [], []
with open(sys.argv[1]) as f:
    for line in f:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            util_gpu.append(float(parts[0]))
            util_mem.append(float(parts[1]))
            mem_used.append(float(parts[2]))
            power.append(float(parts[3]))
        except ValueError:
            continue

def stats(name, vals, unit):
    if not vals:
        return
    vs = sorted(vals)
    p50 = vs[len(vs)//2]
    p95 = vs[min(int(len(vs)*0.95), len(vs)-1)]
    peak = max(vals)
    avg = sum(vals)/len(vals)
    print(f"  {name:18s} avg={avg:7.1f}{unit}  p50={p50:7.1f}{unit}  p95={p95:7.1f}{unit}  peak={peak:7.1f}{unit}  n={len(vals)}")

print("GPU sampler results:")
stats("util.gpu",    util_gpu, "%")
stats("util.memory", util_mem, "%")
stats("memory.used", mem_used, "MiB")
stats("power.draw",  power,    "W")
PY
