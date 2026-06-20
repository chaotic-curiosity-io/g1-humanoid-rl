# Setup notes — DGX Spark G1 RL demo

Append-as-we-go log of deviations from the manual, decisions, and gotchas encountered during actual setup. This is the canonical "what we did" record for future sessions.

Manual source of truth: `docs/dgx-spark-manual.md` (saved copy of Google Doc `16L1rR7fo_XdKKl7PdE4ynwMsbAOFfdeEuCVg3ItFXSY`).
Plan this session is following: `~/.claude/plans/i-want-to-begin-zesty-anchor.md`.

## 2026-04-17 — Session 1 (env setup + baseline training)

### Host survey findings (vs. manual assumptions)

| Manual step | Host state | Action taken |
|---|---|---|
| Ubuntu 24.04 | 24.04.4 LTS ✓ | none |
| CUDA toolkit | 13.0.88, `nvcc` on PATH ✓ | skipped install |
| Docker + NVIDIA runtime | Docker 29.1.3, `nvidia-container-toolkit` 1.18.2, CDI wired ✓ | skipped install |
| NGC PyTorch image | `nvcr.io/nvidia/pytorch:25.11-py3` already present (19.5 GB) | decision pending — check for newer tag |
| uv (host) | 0.10.10 installed ✓ | skipped (will install fresh inside container) |
| `~/robotic-simulation/` | existed, empty | created `docs/`, `wandb-cache/`, `logs/`, `warp-cache/` |
| Swap | `/swap.img` 16 GB ENABLED, 0 B in use | will disable just before training, re-enable after |
| Co-tenants running | `ollama-compose`, `open-webui`, `compose-arangodb-1` containers; comfyui systemd service (`comfyui.service`, running as root, python listening on 0.0.0.0:8188, 170 MiB GPU) | will stop before training, restart after |

GPU: NVIDIA GB10, driver 580.126.09, compute cap 12.1. `memory.total` reports N/A (normal — unified memory).
System memory: 121 GiB total, ~111 GiB available at survey time.

### Decisions

- **Scope this session:** env setup + smoke test + baseline training. Sprinter + playback are a follow-up session.
- **Swap:** disable only during training window, not permanently. Preserves safety net for ollama/comfyui the rest of the time.
- **Co-tenants:** stop via `docker stop`/process kill before training, restart after.
- **`TORCH_CUDA_ARCH_LIST`:** starting with `"12.1"` rather than manual's `"12.1a"`. The `a` suffix selects arch-specific features and isn't needed unless we hit "no kernel image" errors. Easy to revert.
- **NGC image tag:** `nvcr.io/nvidia/pytorch:26.03-py3` (newest available as of 2026-04-17; 26.04 not yet released). Multi-arch manifest includes arm64. Pulled fresh over the cached 25.11-py3 for better Blackwell/CUDA 13 support.

### Repo URL sanity check (2026-04-17)

GitHub API returned 200 for all three:
- `mujocolab/mjlab` — "Isaac Lab API, powered by MuJoCo-Warp, for RL and robotics research"
- `google-deepmind/mujoco_warp`
- `NVIDIA/warp`

### Execution log

(To be appended during Phases B–F.)

#### Phase A — workspace scaffolded
- Saved manual to `docs/dgx-spark-manual.md`
- Created `docs/`, `wandb-cache/`, `logs/`, `warp-cache/`
- Seeded this file

#### Phase B — container launch
- [x] NGC tag check: 26.03-py3 is newest (26.04 not out yet). Pulled.
- Deviation: **host port 8080 was already bound by open-webui** (uvicorn on host network). Remapped mjlab-dev → host 8081 (container 8080). Viser URL is http://localhost:8081. Open-webui will be stopped in Phase E anyway; after that, either port works.
- [x] `mjlab-dev` container launched with port 8081:8080
- [x] Container exec verified. PyTorch 2.11.0a0+nv26.03, CUDA 13.2, GB10 at compute cap (12, 1), no compute-cap warning (26.03 natively supports sm_121 — nice surprise vs. manual §8).

#### Phase C — Python stack
- [x] PyTorch + CUDA verified: PyTorch 2.11.0a0+nv26.03, CUDA 13.2, GB10 sm_121
- [-] uv install: **skipped** — mjlab installed fine with pip, uv wasn't needed
- [x] `warp-lang` 1.13.0.dev20260417 (manylinux_2_34_aarch64 wheel from pypi.nvidia.com). Warp reports "CUDA Toolkit 12.9, Driver 13.2" and finds GB10 at sm_121. Kernel cache lands in `/root/.cache/warp/1.13.0.dev20260417` (persisted via bind mount to host `~/robotic-simulation/warp-cache/`).
- [x] `mujoco-warp` 3.7.0.1 (built from git; also pulled `mujoco` 3.7.0)
- [x] `mjlab` 1.3.0 (editable, pip install -e .) — **no** `--no-deps` needed, deps resolved cleanly (`torch>=2.7.0` is satisfied by our 2.11.0)
- Deviation: **EGL libs were missing inside the container** — `import mujoco` failed with `'NoneType' object has no attribute 'eglQueryString'`. Fixed per manual §4 troubleshooting: `apt-get install -y libegl1 libgl1 libglib2.0-0` inside the container. Full stack imports cleanly after.
- [x] `wandb login` — credentials loaded from `~/robotic-simulation/.env` (W_AND_B_API_KEY) and written to `/root/.netrc` inside the container. Entity confirmed as `donb-chaotic-curiosity`. Note: `.netrc` is in the container's writable layer; if the container is removed, re-run the netrc write from the .env file. **Security:** `.env` holds a secret — add to `.gitignore` before any `git init` of this workspace.

#### Phase D — smoke test
- [x] `python -m mjlab.scripts.demo` running (task: Mjlab-Tracking-Flat-Unitree-G1, 8 envs, viser viewer). MuJoCo-Warp JIT compiled all kernels successfully on sm_121; first-run kernel compile took ~30s total.
- [x] Viser at http://localhost:8081 (host) → 8080 (container). Don visually confirmed 8 G1 humanoid cartwheels rendering correctly.

#### Phase E — baseline training
- [x] Co-tenants stopped (docker stop open-webui compose-arangodb-1 ollama-compose; sudo systemctl stop comfyui.service)
- [x] Swap disabled (sudo swapoff -a). Memory: 111 GiB available, 0 GPU compute processes at launch.
- [x] Training launched: `Mjlab-Velocity-Flat-Unitree-G1`, 2048 envs. Log: `logs/train-baseline-20260417-184620.log`.
- [x] W&B run path: **`donb-chaotic-curiosity/mjlab/qtk6gyny`** — https://wandb.ai/donb-chaotic-curiosity/mjlab/runs/qtk6gyny
- Iteration time steady at ~1.25–1.30 s/iter. At 2048 envs, 1000 iters ≈ 21 min, 2000 iters ≈ 42 min (faster than the manual's 45–90 min estimate).
- [x] Final: stopped cleanly at iter **2050** after ~46 min wall time. Mean reward climbed 2.95 (iter 500) → 21.3 (iter 900) → 31.5 (iter 1400) → **50.51** (iter 2050). Mean episode length reached **995/1000** (near-maximum). Reward was still climbing — not plateaued; a longer run (3–5k iters) would produce a smoother gait.
- Checkpoints: 42 `.pt` files (every 50 iters) at `~/robotic-simulation/mjlab/logs/rsl_rl/g1_velocity/2026-04-17_18-46-23/`. Use `model_2050.pt` (5.1 MB) for the best baseline replay.
- SIGINT note: the `bash -c … > logfile` wrapper didn't propagate SIGINT to the python child on the first try. Had to send `kill -INT <python-pid>` directly (PID found via `docker exec mjlab-dev pgrep -f mjlab.scripts.train`). Remember this for next session's clean stop.

#### Phase F — cleanup
- [x] Swap re-enabled (`sudo swapon -a`): 16 GB back
- [x] Co-tenants restarted: ollama-compose, open-webui, compose-arangodb-1 (docker start), comfyui.service (sudo systemctl start)
- `mjlab-dev` container intentionally **left running** — has the full mjlab stack installed. Next session can `docker exec -it mjlab-dev bash` and go straight to sprinter config edit + training.

### Session 1 summary

Environment setup + baseline training for the mjlab G1 velocity demo went cleanly end-to-end on the DGX Spark.

**What worked:**
- NGC `pytorch:26.03-py3` natively supports sm_121, no compute-cap warning. Full `TORCH_CUDA_ARCH_LIST=12.1` suffix-less works fine.
- `warp-lang` 1.13.0 nightly has aarch64+CUDA-13 wheels (no source build needed — manual had this as a fallback path).
- `mujoco-warp` 3.7.0.1 and mjlab 1.3.0 install cleanly; no `--no-deps` needed.
- mjlab's RL loop hit reward 50.51 and near-max episode length (995/1000) in ~46 min at 2048 envs — faster per-iter than manual's estimate.

**Deviations from manual:**
- Host port 8080 was bound by open-webui → Viser remapped to host 8081.
- Container was missing EGL libs → installed `libegl1 libgl1` inside.
- `.netrc` direct write was more reliable than `wandb login` stdin.
- SIGINT to the bash wrapper didn't reach python; had to kill the python PID directly.

**Artifacts for next session:**
- W&B baseline run: `donb-chaotic-curiosity/mjlab/qtk6gyny`
- Best checkpoint: `~/robotic-simulation/mjlab/logs/rsl_rl/g1_velocity/2026-04-17_18-46-23/model_2050.pt`
- Container `mjlab-dev` has everything installed and ready.

## 2026-06-18 — Session 2 (pre-run recon)

### Findings

- **Progression script CLI**: `python -m mjlab.scripts.record_learning_progression <TASK> --help` loads cleanly with tyro. Confirmed flags: `--run-dir`, `--num-checkpoints`, `--cameras`, `--command-lin-vel-x`. Requires the task name as a positional arg before `--help`.
- **Control smoke (64 envs, 3 iters)**: Completed successfully. Iteration time at 64 envs: ~0.4–0.7 s/iter (kernel JIT warms up after iter 0). At 2048 envs (Session 1 baseline) this is ~1.25 s/iter — consistent.
- **Velocity-range override — CURRICULUM CLOBBER (resolved)**:
  - `--env.commands.twist.ranges.lin-vel-x=(-0.5, 0.5)` alone does NOT stick. The `commands_vel` curriculum function fires at every episode boundary; its stage 0 (`step: 0`) always applies and resets `lin_vel_x = (-1.0, 1.0)`.
  - **Fix**: must override all three curriculum stages via CLI in addition to the command range. Working invocation:
    ```bash
    "--env.commands.twist.ranges.lin-vel-x=(-0.5, 0.5)" \
    "--env.curriculum.command-vel.params.velocity-stages.0.lin-vel-x=(-0.5, 0.5)" \
    "--env.curriculum.command-vel.params.velocity-stages.1.lin-vel-x=(-0.5, 0.5)" \
    "--env.curriculum.command-vel.params.velocity-stages.2.lin-vel-x=(-0.5, 0.5)"
    ```
  - Verified: `params/env.yaml` in the saved run shows `(-0.5, 0.5)` at both `commands.twist.ranges.lin_vel_x` and all three `curriculum.command_vel.velocity_stages[*].lin_vel_x`. Override confirmed good.
  - Note: tuple args must be passed with `=` syntax (e.g. `--flag=(-0.5, 0.5)`) — space-separated positional values are rejected by tyro as "Unrecognized options".
- **Tooling confirmed in container**: matplotlib 3.10.8, pandas 3.0.1, tensorboard present.
- **Tensorboard scalar tags**: 39 tags total. Reward-relevant: `Train/mean_reward`, `Episode_Reward/track_linear_velocity`, `Episode_Reward/track_angular_velocity`, and 12 others. Full tag list confirmed via `EventAccumulator` on the baseline run dir.
- **Baseline intact**: `logs/rsl_rl/g1_velocity/2026-04-17_18-46-23/` untouched — 42 `.pt` files present after cleanup.

### Session 2 summary

Smoke tests confirm the Spark stack is ready. The critical finding is that the velocity-range override requires patching all three curriculum stages, not just the command range. Tasks 3–5 must use the 4-flag form above for any `lin_vel_x` override.

## 2026-06-19 — Session 2 execution (learning-arc reports)

Built a 4-part beginner learning series (`docs/reports/00–03`) backed by real runs, on branch `g1-walking-learning-arc`. (This replaces the earlier "sprinter via flat_env_cfg.py edit" stub — we did a CLI-override **slow** tweak instead, no source edit.)

**Runs** (both fresh from scratch, 1500 iters @ 2048 envs, seed 42, ~26 min each):
- **Control** `arc-control` → `logs/rsl_rl/g1_velocity/2026-06-19_07-39-36_arc-control`, final reward **38.43**, ep len ~988, W&B `donb-chaotic-curiosity/mjlab/5zkg1dh7`. Reproduces the Session-1 baseline — curves overlap at matched iters (500→2.9 vs 3.3, 900→19.5 vs 21.3, 1400→36.0 vs 31.5).
- **Tweak** `arc-tweak-slow` → `logs/rsl_rl/g1_velocity/2026-06-20_00-36-34_arc-tweak-slow`, `lin_vel_x` pinned to (-0.5, 0.5) via the 4-flag override. Final reward **57.84** — *higher* than control because the slow-only task is easier (the A/B's central lesson: higher reward ≠ better robot). Commanded to 1.0 m/s (2× its training max) it drifts off-heading rather than striding.

**Host-prep deviation:** sudo is NOT passwordless on the Spark, so `swapoff -a` / `systemctl stop comfyui` could not run non-interactively. Instead: stopped the 3 co-tenant containers (docker, no sudo) and ran with live `free -h` monitoring — with ~110 GiB free for a ~13 GiB workload, swap never engaged. Restored co-tenants after with `docker start` (no swapon needed).

**Gotchas / learnings:**
- **Flat task curriculum is ACTIVE** (contradicts the earlier "curriculum off for flat" assumption): `command_vel` stage-0 fires at step 0 and resets `lin_vel_x` to (-1,1), clobbering a plain override. Must pin all 3 stages too (see the recon block's 4-flag form above).
- **Background-watcher self-match:** `pgrep -f mjlab.scripts.train` (or any `grep`/count) matches its OWN command line → loops forever / false counts. Use the bracket trick: `ps -eo cmd | grep "[p]ython -u -m mjlab.scripts.train"`.
- **ffmpeg** was missing in the container → installed once via `apt-get install -y ffmpeg`.

**New committed artifacts:** `scripts/plot_training_curves.py` (tensorboard → CSV + curve PNGs); `docs/reports/00–03` + `docs/reports/assets/*.png` (plots + frame stills — MP4s gitignored, pulled to `assets/*.mp4`, originals on the Spark at `~/robotic-simulation/out/arc/`); design spec + plan under `docs/superpowers/{specs,plans}/2026-06-18-g1-walking-learning-arc*`.
