# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This is a workspace built around **mjlab** (upstream: mujocolab/mjlab), an Isaac-Lab-style RL API built on MuJoCo-Warp. It is set up to train Unitree G1 humanoid policies on the DGX Spark (aarch64 / CUDA 13 / sm_121 Blackwell).

- `mjlab/` — editable clone of the upstream project. **Owned by root** because it is developed inside the `mjlab-dev` Docker container. Has its own `CLAUDE.md` for the upstream project's conventions.
- `scripts/` — host-side helper scripts that import mjlab (e.g. `watch_learning.py` — side-by-side multi-checkpoint viewer).
- `logs/` — host-side training run logs (not to be confused with `mjlab/logs/rsl_rl/...` which contains the actual RSL-RL checkpoints written from inside the container).
- `docs/` — `dgx-spark-manual.md` (local copy of the ops manual) and `session-*.md` run writeups.
- `setup-notes.md` — append-as-we-go log of actual deviations from the manual, per session. Update it after notable events; it is the canonical "what we did" record.
- `wandb-cache/`, `warp-cache/` — bind-mount targets for W&B and Warp kernel caches.

## Running code

Everything runs inside the `mjlab-dev` Docker container. The host directory `~/robotic-simulation/` is mounted at `/workspace`, so edits on the host show up at `/workspace/mjlab/...` inside the container.

```sh
# Inspect the container (it is normally left running between sessions)
docker ps --format 'table {{.Names}}\t{{.Status}}'

# Open a shell
docker exec -it mjlab-dev bash

# Run mjlab entrypoints (use `python -m`, NOT `uv run` — the container installs mjlab via pip -e)
docker exec mjlab-dev bash -lc "cd /workspace/mjlab && python -m mjlab.scripts.train Mjlab-Velocity-Flat-Unitree-G1 ..."
docker exec mjlab-dev bash -lc "cd /workspace/mjlab && python -m mjlab.scripts.play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path ..."
docker exec mjlab-dev bash -lc "cd /workspace && MUJOCO_GL=egl python scripts/watch_learning.py"
```

The Makefile / `uv run` targets documented inside `mjlab/CLAUDE.md` assume a uv-managed venv. **Do not use them in this workspace** — `uv` was skipped in favor of pip editable. `python -m mjlab.scripts.<name>` is the canonical invocation here.

## Editing mjlab source

`mjlab/` on the host is owned by root because the container writes into the bind mount as root. Do not `sudo chown` the tree (that also desynchronises the container's file ownership). Two ways to edit:

1. **`docker cp` from `/tmp`** — write the file to `/tmp/foo.py` on the host, then `docker cp /tmp/foo.py mjlab-dev:/workspace/mjlab/src/mjlab/...`. This is the path the existing `record_learning_progression.py` was added through.
2. **Edit inside the container** via `docker exec -it mjlab-dev bash` + your editor of choice. Slower round-trip but fine for small tweaks.

mjlab is installed editable (`pip install -e .`), so new `.py` files under `src/mjlab/...` are picked up immediately — no reinstall needed.

## Training / playing G1 velocity policy

Task ids (registry):

- `Mjlab-Velocity-Flat-Unitree-G1` — plane ground, curriculum disabled. This is the baseline used in Session 1.
- `Mjlab-Velocity-Rough-Unitree-G1` — procedural terrain with curriculum.

Both share `experiment_name="g1_velocity"`, which means `logs/rsl_rl/g1_velocity/<timestamp>/` does not tell you which variant produced the run — check `params/env.yaml` (`terrain_type: plane` ⇒ Flat). Any helper that auto-infers task id from the log dir must handle this ambiguity (see `record_learning_progression.py`'s `_infer_task_id`).

Baseline Session 1 checkpoints live at `mjlab/logs/rsl_rl/g1_velocity/2026-04-17_18-46-23/` — 42 `.pt` files every 50 iters, 0 → 2050. Reward 50.5 at `model_2050.pt`, ep length 995/1000 (still climbing; was not plateaued).

Stopping a training run cleanly: **send SIGINT directly to the python child**, not to the wrapping shell.

```sh
docker exec mjlab-dev bash -lc "kill -INT \$(pgrep -f mjlab.scripts.train)"
```

The `bash -c '... > log'` wrapper does not propagate SIGINT on the first try.

## Spark host prep for training

Co-tenants on this host (ollama, open-webui on `:8080`, arangodb, comfyui systemd service) must be quiesced before a long run — the Spark's unified memory means swap-death-spirals under contention cause hard reboots, not clean CUDA OOMs. See `project_spark_host_layout.md` memory for full quiesce/restore commands. In short:

```sh
# Before training
docker stop open-webui compose-arangodb-1 ollama-compose
sudo systemctl stop comfyui.service
sudo swapoff -a

# After
sudo swapon -a
docker start open-webui compose-arangodb-1 ollama-compose
sudo systemctl start comfyui.service
```

Viser runs at **host :8081** (not :8080 — open-webui owns that port). Train launches print a `http://localhost:8080` URL from inside the container; translate to `:8081` when opening in the host browser.

## Rendering / recording

The offscreen renderer (`mjlab/src/mjlab/viewer/offscreen_renderer.py`) has a few sharp edges worth knowing:

- It mutates `model.stat.extent`, `model.vis.global_.offheight/offwidth`, and `model.light_castshadow[:]` based on `ViewerConfig`. To re-use an env for multiple camera angles, mutate `renderer._cam.azimuth/elevation/distance` between `env.render()` calls rather than rebuilding the renderer.
- `ViewerConfig.enable_shadows=True` + tight tracking-camera extent produces **shadow-map acne that flickers frame-to-frame** (looks like jittery horizontal lines on the ground). Disable shadows for clean recording, or set `model.vis.quality.shadowsize = 4096+`.
- Manager/sensor debug visualizers (command arrow, foot height scan rings, contact markers) are drawn on top of the scene every render — these are the usual source of "lines on the ground" in recorded video. Disable by monkey-patching `env.update_visualizers = lambda *a, **k: None` before the render loop. There is no CLI flag for this in upstream mjlab.
- Checkpoints drive `model_<iter>.pt`. Load into an existing runner via `runner.load(path, load_cfg={"actor": True}, strict=True, map_location=device)` then `runner.get_inference_policy(device)`. This is what `play.py`'s CheckpointManager and `record_learning_progression.py` both do — no need to rebuild the env per checkpoint.
- The `UniformVelocityCommand` ("twist") randomises the commanded velocity on every reset. To hold it fixed while recording, after each reset: set `term.vel_command_b[:]` and zero the `is_heading_env/is_standing_env/is_world_env/is_forward_env` flags, then re-assert every step (the command is recomputed on each `env.step()`).

Local additions on top of upstream:

- `mjlab/src/mjlab/scripts/record_learning_progression.py` — multi-checkpoint, multi-camera progression recorder (stitches one MP4 per angle and optional 2×2 grid; labels iteration number). Pass `--no-shadows --no-reflections --no-debug-viz` for clean frames.
- `scripts/watch_learning.py` — host-level helper that loads 8 checkpoints into one 1024-env scene and routes each 128-env block to a different policy, with an orbit-cam toggle in the Viser GUI.

## Pre-commit expectations inside mjlab

If touching files under `mjlab/`, the upstream project's checks still apply:

```sh
docker exec mjlab-dev bash -lc "cd /workspace/mjlab && make check"   # format + lint + type
docker exec mjlab-dev bash -lc "cd /workspace/mjlab && make test-fast"
```

However, `make` shells out to `uv run`, which the container doesn't have set up. In practice, use the raw tools directly:

```sh
docker exec mjlab-dev bash -lc "cd /workspace/mjlab && ruff format && ruff check --fix && pyright"
docker exec mjlab-dev bash -lc "cd /workspace/mjlab && pytest tests/<test>.py"
```
