#!/usr/bin/env bash
# Sample all GPUs at 250 ms cadence; per-GPU avg/p50/p95/peak.
# Usage:  ./gpu_sampler_all.sh <duration_seconds>
set -euo pipefail
duration="${1:-30}"

samples_file=$(mktemp)
trap 'rm -f "$samples_file"' EXIT

end=$(( $(date +%s) + duration ))
while [[ $(date +%s) -lt $end ]]; do
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw \
    --format=csv,noheader,nounits >>"$samples_file" 2>/dev/null
  sleep 0.25
done

python3 - "$samples_file" <<'PY'
import sys, collections

by_gpu = collections.defaultdict(lambda: {"util": [], "mem": [], "power": []})
with open(sys.argv[1]) as f:
    for line in f:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            by_gpu[idx]["util"].append(float(parts[1]))
            by_gpu[idx]["mem"].append(float(parts[2]))
            by_gpu[idx]["power"].append(float(parts[3]))
        except ValueError:
            continue

def s(vals):
    if not vals:
        return None
    return {
        "avg": sum(vals)/len(vals),
        "p50": sorted(vals)[len(vals)//2],
        "p95": sorted(vals)[min(int(len(vals)*0.95), len(vals)-1)],
        "peak": max(vals),
    }

n = len(next(iter(by_gpu.values()))["util"]) if by_gpu else 0
print(f"per-GPU summary (n={n} samples):")
print("  gpu   util(%) avg/p50/p95/peak       mem(MiB) peak    power(W) avg/peak")
for idx in sorted(by_gpu):
    g = by_gpu[idx]
    u, m, p = s(g["util"]), s(g["mem"]), s(g["power"])
    print(f"   {idx}   {u['avg']:5.1f}/{u['p50']:5.1f}/{u['p95']:5.1f}/{u['peak']:5.1f}        {m['peak']:7.0f}        {p['avg']:5.1f}/{p['peak']:5.1f}")
PY
