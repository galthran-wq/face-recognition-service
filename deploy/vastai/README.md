# Multi-instance deployment on vast.ai

This folder ships a launcher and tooling for running **N face-recognition-service instances per GPU** on a vast.ai box, fronted by a round-robin Caddy load balancer and (optionally) a Cloudflare Named Tunnel for HTTPS exposure under your own domain.

This is what gets baked in:

- One uvicorn process per "instance", pinned to one GPU via `CUDA_VISIBLE_DEVICES`.
- Per-instance TensorRT engine cache on disk so cold starts are cheap after the first build.
- A Caddy reverse proxy on loopback that load-balances across the pool (least-connections, active health checks).
- A Cloudflare Named Tunnel publishing each instance under a subdomain (`gpu0-a.example.com`, …) plus the LB (`fr-pool.example.com`).

## Why N processes instead of one bigger worker

A single uvicorn worker is GIL-bound: `onnxruntime` GPU calls are synchronous and block the event loop, so per-process throughput plateaus at roughly `1 / per_request_latency` regardless of `--workers` or `--concurrency`. With one instance per GPU and the bundled 1280×886 test image (6 faces):

| Endpoint | Cold (warm TRT) | Saturated RPS @ C=4 | GPU util | Power |
|---|---:|---:|---:|---:|
| `/faces/detect` | 41 ms | ~33 RPS | 6–8 % | ~100 W |
| `/faces/embed`  | 56 ms | ~23 RPS | 10–13 % | ~120 W |
| `/faces/analyze` | 78 ms | ~17 RPS | 10–12 % | ~120 W |

The GPU is at ~10 % utilization at full single-process saturation, so adding more processes per GPU is pure win until VRAM or compute actually saturates. With **3 instances per GPU** on a 3×RTX 5090 host (the default `INSTANCES` array) we measured:

| | C=72 over the LB | per-GPU util peak | power peak |
|---|---:|---:|---:|
| `/faces/detect`  | **436 RPS**, p50 153 ms | 47 % | 164 W |
| `/faces/embed`   | **303 RPS**, p50 223 ms | 61 % | 225 W |
| `/faces/analyze` | **237 RPS**, p50 296 ms | 76 % | 212 W |

Roughly linear scaling with the number of instances. Going past 3/GPU is possible but watch VRAM — under sustained load each ORT arena holds onto its workspace and the resident set grows; see "Memory and OOM" below.

## Prerequisites

The vast.ai instance must already have:

- The repo cloned and `uv sync` run from the repo root.
- `onnxruntime-gpu`, TensorRT pip wheels, and NVIDIA cu12 runtime libs installed into `.venv`:
  ```bash
  uv pip install --python .venv/bin/python --reinstall onnxruntime-gpu==1.24.1
  uv pip install --python .venv/bin/python \
      tensorrt-cu12-libs==10.7.0 tensorrt-cu12-bindings==10.7.0 \
      nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-curand-cu12 nvidia-cufft-cu12 \
      nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-nvjitlink-cu12 nvidia-cuda-nvrtc-cu12
  ```
  The launcher prepends the matching directories to `LD_LIBRARY_PATH` automatically.
- One-time model download:
  ```bash
  .venv/bin/python -c "from insightface.app import FaceAnalysis; \
    FaceAnalysis(name='buffalo_l', root='/root/.insightface', \
    providers=['CUDAExecutionProvider']).prepare(ctx_id=0, det_size=(640,640))"
  ```

`caddy` is preinstalled on vast.ai instance images at `/opt/instance-tools/bin/caddy`. `cloudflared` is at `/opt/portal-aio/tunnel_manager/cloudflared`. The launcher picks both up automatically; override with `CADDY_BIN` / `CLOUDFLARED_BIN` if needed.

## Quick start

```bash
cd deploy/vastai
./run_instances.sh start
./run_instances.sh status
./run_instances.sh urls
./run_instances.sh stop
```

After `start`:

- 9 uvicorn workers (default) bound to `127.0.0.1:11113..11127`.
- Caddy LB on `127.0.0.1:7999`.
- If `~/.cloudflared/cert.pem` exists (i.e. you ran `cloudflared tunnel login`), the Named Tunnel comes up too and `urls` lists the public HTTPS hostnames.

### Editing topology

Edit the `INSTANCES` array at the top of `run_instances.sh`:

```bash
INSTANCES=(
  "gpu0-a 0 11113"   # name  gpu_index  loopback_port
  "gpu0-b 0 11114"
  "gpu0-c 0 11115"
  "gpu1-a 1 11119"
  ...
)
```

The launcher regenerates `caddy_lb.generated.conf` and `cloudflared.generated.yml` from this array on every `start`. To dump either without starting anything: `./run_instances.sh gen-caddy` / `./run_instances.sh gen-cloudflared`.

## Exposing the pool to the internet

vast.ai NATs only a handful of ports from `185.x.x.x:<external> → container:<internal>` (see `env | grep VAST_TCP_PORT_`). On a typical instance there is no spare external TCP port to put a public LB behind, so we tunnel out instead.

### Option A — Cloudflare Named Tunnel (recommended for HTTPS)

1. Add your domain to Cloudflare (free plan is fine; you'll need to change NS at your registrar).
2. From the box, log in once — this writes `~/.cloudflared/cert.pem`:
   ```bash
   cloudflared tunnel login
   ```
   It prints a URL; open it in your browser, **pick the zone, click "Authorize"**.
3. Create the tunnel — writes `<tunnel-id>.json` (credentials) next to the cert:
   ```bash
   cloudflared tunnel create face-recognition-pool
   ```
4. Edit `CLOUDFLARED_DOMAIN` in `run_instances.sh` (or export it) to your domain, then:
   ```bash
   ./run_instances.sh provision-dns   # creates CNAMEs in CF for fr-pool + each instance
   ./run_instances.sh restart
   ```

You get `https://fr-pool.<your-domain>` (LB) and `https://gpu0-a.<your-domain>` …, all properly TLS-terminated by Cloudflare with arbitrary body sizes. The tunnel uses QUIC to four CF edge POPs.

**Avoid "quick tunnels" (`*.trycloudflare.com`).** They throttle uploads to ~1 Mbps; any `/faces/*/batch` request with more than a handful of images dies on `WriteTimeout`. We verified Named Tunnels handle 5 MiB bodies (`batch=32 × 640×640`) at ~2 s wall time end-to-end through CF.

### Option B — SSH port-forward from your prod box

If you don't want CF in the path, your prod host can reverse the vast SSH port (`VAST_TCP_PORT_22` → external `433XX`) and forward each loopback port:

```bash
ssh -N -p $VAST_SSH_PORT root@$VAST_IP \
  -L 11113:127.0.0.1:11113 -L 11114:127.0.0.1:11114 ... \
  -L 7999:127.0.0.1:7999 \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes
```

No throttle, no CF, plain HTTP on loopback — the LB is then reachable as `http://127.0.0.1:7999` on the prod host.

### Option C — repurpose a vast-allocated external port

vast.ai's external ports (typically 1 SSH + 5 service ports) are NATed 1:1 to specific container ports. If any service inside the container that holds a NATed port is unused (Syncthing on 8384, Tensorboard on 6006, the unused 43539→43539 1:1 mapping on a default instance), kill it and rebind your LB to its internal port. Cleanest for plain-IP exposure but limited to HTTP (no public TLS cert).

## Tools in this folder

| File | What it does |
|---|---|
| `run_instances.sh` | The launcher. Start/stop/restart/status/urls + config generation. |
| `warmup_all.py` | Fires `/faces/{detect,embed,analyze}` once per instance in parallel — primes TRT engines from the on-disk cache so the first real request is fast. |
| `loadtest.py` | httpx-async load tester. Configurable concurrency/duration, prints p50/p90/p95/p99 latencies and RPS. |
| `gpu_sampler.sh` | 250 ms-cadence sampling of `nvidia-smi` for one GPU; prints avg/p50/p95/peak util, memory, power over a window. |
| `gpu_sampler_all.sh` | Same but for every GPU at once — handy to run alongside a load test. |
| `probe_tunnels.py` | Verifies the external Cloudflare-tunnel hostnames handle three body sizes (small/medium/big batch). |

## Observations and gotchas

These are the things that surprised us; capture so the next person doesn't repeat them.

### Throughput is GIL-bound, not GPU-bound

At C=1 a single instance does `1000 / per_request_ms` RPS; at C=4 it plateaus. Adding more `--workers` to a single uvicorn doesn't help because each request blocks on a synchronous GPU call. The fix is N independent processes; the GPU has room to spare (under 50 % util at full saturation).

### TRT cold start

The first time any model runs through `TensorrtExecutionProvider` it compiles an engine, which can take 30–60 s per model. With `trt_engine_cache_enable=true` and a per-instance `FACE_TRT_CACHE_PATH` (set automatically by the launcher), subsequent restarts on the same GPU rebuild the engine context from disk in ~5–10 s.

Each instance must have its own cache directory or they race on first-run compilation. The launcher gives each one `trt_cache/<instance-name>/`.

### Memory grows under load and doesn't shrink

Idle, each instance holds ~600 MiB on its GPU (model weights + engine context). Under sustained load the ORT arena grows aggressively and stays — we observed a 9-instance pool reach 22–26 GiB per RTX 5090 (32 GiB cards) after a load test, with no decay when idle. Running 6/GPU got us OOM'd; 3/GPU is comfortable; 4/GPU is workable with a small headroom margin.

If you need to reclaim memory, `./run_instances.sh restart` returns each card to ~5–6 GiB (cold-state × instance count), and TRT engines reload from cache in seconds.

### CUDA 12 vs system CUDA 13

vast.ai images ship CUDA 13 toolkit on the host, but `onnxruntime-gpu` cu12 wheels expect CUDA 12 runtime libs. Installing the `nvidia-*-cu12` pip wheels and putting them on `LD_LIBRARY_PATH` (which `run_instances.sh` does for you) is sufficient — no system CUDA reinstall needed.

### Cloudflare Quick Tunnels are not suitable

`cloudflared tunnel --url http://...` (without a config or login) gives a `*.trycloudflare.com` URL that's bandwidth-throttled. Single small requests sneak through fine (~150 ms), but anything POSTing > ~100 KiB hits `WriteTimeout`. Use Named Tunnels (Option A above) or skip CF entirely.

### Health endpoint != model loaded

`/health` returns 200 as soon as uvicorn binds the port, before InsightFace loads the model. The launcher's `status` action reports `health=200` immediately; the model takes a few seconds longer. If a request races startup it'll wait for the lifespan to finish loading, so this is mostly cosmetic — but `warmup_all.py` waits for `/health` plus a real inference call.

### Per-instance bind is `127.0.0.1`, not `0.0.0.0`

The instances only listen on loopback. Public exposure is exclusively via the LB or the Cloudflare Tunnel. This is intentional: no accidental open ports if a container's `iptables` config changes.

### `failed to sufficiently increase receive buffer size` (cloudflared)

The vast container is unprivileged, so cloudflared can't tune the kernel's UDP receive buffer. cloudflared still works; throughput per QUIC stream is somewhat capped but with four parallel connections this hasn't been a real bottleneck for our payloads.

## Operational cheatsheet

```bash
# Bring everything up
./run_instances.sh start
.venv/bin/python deploy/vastai/warmup_all.py    # primes TRT — optional but recommended

# Watch what's happening
./run_instances.sh status
nvidia-smi -lms 500
tail -f logs/gpu0-a.log

# Load test one instance / the LB
.venv/bin/python deploy/vastai/loadtest.py --url http://127.0.0.1:11113 \
    --endpoint /faces/detect --concurrency 16 --duration 15
.venv/bin/python deploy/vastai/loadtest.py --url http://127.0.0.1:7999 \
    --endpoint /faces/analyze --concurrency 72 --duration 15

# Sample all GPUs alongside a test
./deploy/vastai/gpu_sampler_all.sh 15

# Verify the public Cloudflare hostnames end-to-end
.venv/bin/python deploy/vastai/probe_tunnels.py

# Resize the pool: edit INSTANCES in run_instances.sh, then
./run_instances.sh restart
```
