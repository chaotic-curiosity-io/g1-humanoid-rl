# Session 1 — Baseline G1 Walking Policy on DGX Spark

**Date:** 2026-04-17
**Duration:** ~1 hour wall time (environment setup + smoke test + training + restore)
**Outcome:** A trained Unitree G1 velocity-tracking walking policy, demonstrably walking in a 3D browser viewer.

---

## TL;DR

We stood up a GPU-accelerated robotics RL stack on the DGX Spark (Arm64, CUDA 13, Blackwell sm_121), ran a smoke-test demo to confirm it worked, trained a Unitree G1 humanoid policy from randomly-initialized weights for ~46 minutes using 2048 parallel physics simulations, and replayed the result in a browser. The policy learned to stay upright and track commanded velocities — the walking behavior you see in the viewer is entirely emergent from ~2000 iterations of PPO training on this specific machine.

**No pretrained weights were used for the velocity policy.** The only pretrained asset touched all session was a cartwheel-tracking checkpoint downloaded during the smoke test, which is unrelated to the policy we trained.

---

## What We Actually Ran, Step by Step

### Phase A — Workspace scaffold

- Created `~/robotic-simulation/` subdirectories (`docs/`, `logs/`, `wandb-cache/`, `warp-cache/`).
- Saved the April 2026 Chaotic Curiosity manual locally at `docs/dgx-spark-manual.md`.
- Seeded `setup-notes.md` as a running log of deviations from the manual.

### Phase B — NGC container

- Checked the NGC PyTorch catalog; `26.03-py3` was the newest multi-arch tag (aarch64 supported). Pulled it (22.5 GB).
- Launched the container `mjlab-dev` with GPU passthrough, 16 GB shared memory, workspace bind-mount, and the Warp kernel cache bind-mount.
- Container is still running and contains the full Python environment for future sessions.

**Deviation from the manual:** Host port `8080` was already bound by `open-webui` (uvicorn on host network). Remapped the container's Viser port to host `:8081`. All viewer URLs in this session used `:8081`.

### Phase C — Python stack inside the container

- Verified PyTorch 2.11.0a0+nv26.03, CUDA 13.2, GB10 at compute capability (12, 1). No compute-cap warning — NGC 26.03 natively supports sm_121.
- `pip install` chain completed cleanly, all aarch64+CUDA13 wheels available:
  - `warp-lang==1.13.0.dev20260417` (NVIDIA nightly)
  - `mujoco-warp==3.7.0.1` (from Google DeepMind git)
  - `mujoco==3.7.0`
  - `mjlab==1.3.0` (editable install; deps `rsl-rl-lib==5.0.1`, `viser==1.0.26`, `wandb==0.26.0`, `tyro`, `torchrunx`, `tensordict`, `trimesh`, `mjviser`, etc., all resolved without conflicts)

**Deviation from the manual:** Container was missing EGL libs (`libegl1`, `libgl1`), causing `import mujoco` to crash. Fixed with `apt install -y libegl1 libgl1 libglib2.0-0` inside the container.

### Phase C.5 — W&B authentication

- Loaded the W&B API key from `.env` (variable `W_AND_B_API_KEY`).
- Wrote it to `/root/.netrc` inside the container so it persists across `docker exec` calls.
- Entity confirmed as `donb-chaotic-curiosity`.

**Deviation:** `wandb login` via stdin pipe didn't persist auth to disk in this version; direct `.netrc` write was the reliable path.

### Phase D — Smoke test

- Ran `python -m mjlab.scripts.demo` inside the container. This script downloads a pretrained tracking checkpoint (`demo_ckpt.pt`) and motion clip (`demo_motion.npz`) from Google Cloud Storage and replays 8 Unitree G1s doing cartwheels.
- Browser confirmation at `http://localhost:8081` — 8 G1s cartwheeling on a ground plane.

This smoke test was the **only** use of a pretrained model in the session. It confirmed the entire stack (Warp JIT, MuJoCo physics, mjlab, Viser viewer, browser rendering) was functional end-to-end before we invested an hour in training.

### Phase E — Baseline training

**Host quiescence first** (all reversible, restored in Phase F):
- `docker stop open-webui compose-arangodb-1 ollama-compose` — freed the 3 always-on containers.
- `sudo systemctl stop comfyui.service` — freed the 170 MiB GPU slot held by ComfyUI.
- `sudo swapoff -a` — disabled the 16 GB swap file. Critical on unified-memory systems: an OOM with swap active can swap-death-spiral the whole machine.

**Training launch:**

```
docker exec -d mjlab-dev bash -c \
  'cd /workspace/mjlab && \
   MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
   python -u -m mjlab.scripts.train \
   Mjlab-Velocity-Flat-Unitree-G1 \
   --env.scene.num-envs 2048 \
   > /workspace/logs/train-baseline-20260417-184620.log 2>&1'
```

**Training run:**
- Task: `Mjlab-Velocity-Flat-Unitree-G1` (flat ground, 3-DOF commanded velocity: forward/lateral/yaw-rate)
- Parallel envs: 2048
- Rate: ~1.25 s/iter (faster than the manual's 1–2 s/iter estimate)
- Started 18:46 local time, ran to iteration 2050, stopped via `SIGINT`
- Peak GPU utilization: 87%, GPU memory 1.4 GiB, system memory 13 GiB (of 121 GiB available)

**Learning curve** (from the W&B run at https://wandb.ai/donb-chaotic-curiosity/mjlab/runs/qtk6gyny):

| Iteration | Mean reward | Mean episode length |
|---|---|---|
| 500 | 3.30 | 173 steps |
| 900 | 21.30 | 962 steps |
| 1400 | 31.47 | 925 steps |
| 2050 | **50.51** | **995 / 1000 steps** |

The classic S-curve: random flailing for the first ~500 iterations, a sharp "aha" climb between iterations 500–1000 when the policy found the basic pattern, steady refinement through 2050. Reward was still rising when we stopped — a longer run (5000+ iters) would push the curve higher before plateauing.

**Deviation from the manual:** `SIGINT` sent to the `bash -c …` wrapper did not propagate to the inner Python process. Had to `kill -INT <python-pid>` directly using `docker exec mjlab-dev pgrep -f mjlab.scripts.train`. Worth remembering for next session's clean stop.

### Phase F — Host restore

- `sudo swapon -a` — re-enabled the 16 GB swap.
- `sudo systemctl start comfyui.service` — restored ComfyUI.
- `docker start ollama-compose open-webui compose-arangodb-1` — restored the three always-on Docker services.
- `mjlab-dev` container intentionally left running for next session (preserves the installed Python stack and the MuJoCo-Warp kernel cache).

### Phase G — Playback (on your request, after the initial completion)

- `python -m mjlab.scripts.play Mjlab-Velocity-Flat-Unitree-G1 --checkpoint-file /workspace/mjlab/logs/rsl_rl/g1_velocity/2026-04-17_18-46-23/model_2050.pt --num-envs 8 --viewer viser`
- Viser live at http://localhost:8081 rendering 8 independent G1s walking/turning under random commanded velocities.
- This is the trained policy doing inference only — no further learning.

---

## The Software Stack, Layer by Layer

Bottom-up. Every layer listed here was used in this session.

### DGX Spark hardware

NVIDIA's Arm-based desktop AI box. **GB10 Grace Blackwell Superchip** — 20-core Arm CPU and a Blackwell GPU on the same die, sharing **121 GiB of unified LPDDR5x memory** (no separate VRAM). Compute capability **sm_121**. All our GPU libraries had to be compiled targeting this specific compute capability, which was the main compatibility concern of the session.

### DGX OS + NVIDIA driver 580

Ubuntu 24.04.4 LTS base with NVIDIA's tooling pre-installed. Driver version 580.126.09, supports CUDA 13.x. Not modified.

### CUDA 13.0.88

NVIDIA's GPU programming API. Consumed indirectly by everything above it — we never wrote CUDA kernels directly.

### Docker 29.1.3 + NVIDIA Container Toolkit 1.18.2

Standard containerization + GPU passthrough. Provides isolation from the host's other Python environments (ComfyUI, ollama).

### NGC PyTorch container (`nvcr.io/nvidia/pytorch:26.03-py3`)

NVIDIA's prebuilt "known-good" PyTorch container, published monthly. Saves us from cross-compiling PyTorch and cuDNN for aarch64+Blackwell ourselves. 22.5 GB, Python 3.12 inside, aarch64-native.

### PyTorch 2.11.0a0 (NVIDIA build)

Meta's deep-learning framework. Holds the **policy neural network** — ~5 MB of weights that map `observations → actions`. Standard architecture for RL locomotion: an MLP (multi-layer perceptron) with a few hundred units per hidden layer. The learning happens through PyTorch's autograd and Adam optimizer.

### NVIDIA Warp 1.13.0 nightly

NVIDIA's Python-to-CUDA JIT compiler, aimed at physics/simulation/geometry. Lets researchers write GPU kernels in a Python subset; Warp compiles them on demand. When you saw `Module mujoco_warp._src.solver ... (compiled)` at startup, that was Warp JITing MuJoCo's physics kernels for your specific sm_121 GPU. Compiled kernels are cached to `~/robotic-simulation/warp-cache/` so subsequent runs skip recompilation.

### MuJoCo 3.7.0 + MuJoCo Warp 3.7.0.1

**MuJoCo** is the physics engine: rigid-body dynamics, joint actuators, contacts, friction. Originally written by Emo Todorov (2012), acquired and open-sourced by DeepMind (2021). Gold-standard for contact-rich robotics sim.

**MuJoCo Warp** is the Google DeepMind + NVIDIA port of MuJoCo onto NVIDIA Warp. Traditional MuJoCo runs one sim per CPU thread; MuJoCo Warp runs 2048 sims as batched GPU kernels. This is the 100x–1000x speedup that made modern parallel-env RL practical.

**MJCF** (`.xml`) files describe robot bodies: kinematic tree, masses, joint types and limits, actuator torque curves, contact geometry. The Unitree G1 MJCF ships inside mjlab's repo and defines 29 actuated joints.

### RL algorithm layer: rsl_rl 5.0.1 + PPO

**rsl_rl** is a minimalist PPO implementation built by ETH Zürich's Robotic Systems Lab — the same group behind ANYmal and the ANYbotics commercial work. De-facto default PPO for modern humanoid/quadruped RL.

**PPO** (Proximal Policy Optimization) is the learning algorithm itself, invented by John Schulman et al. at OpenAI in 2017. Policy-gradient method with a clipped surrogate objective that prevents each weight update from changing the policy too much at once — makes training stable across thousands of iterations without diverging.

Each of our 2050 iterations ran this inner loop: collect ~24 physics steps × 2048 envs = ~49K transitions → compute advantages → 5 epochs of minibatch SGD on the policy+value networks → log to W&B → repeat. ~1.25 s per full iteration.

### mjlab 1.3.0

The **"glue" layer**. Self-described as "Isaac Lab API, powered by MuJoCo-Warp." Provides:
- Task definitions (`Mjlab-Velocity-Flat-Unitree-G1` and 11 others)
- Reward functions (14 weighted terms for the velocity task: `track_linear_velocity`, `track_angular_velocity`, `upright`, `pose`, `soft_landing`, `self_collisions`, `action_rate`, `joint_acc`, `base_height`, `foot_clearance`, `alive`, `energy`, `symmetry`, plus termination penalties)
- The Unitree G1 / Go1 / H2 MJCF robot files
- `train.py` / `play.py` / `demo.py` entry-point scripts
- Glue between MuJoCo Warp's tensor API and rsl_rl's expected Python interface
- W&B integration

Created late 2024 to give researchers the Isaac Lab workflow with MuJoCo physics instead of NVIDIA PhysX.

### Weights & Biases (wandb 0.26.0)

SaaS experiment tracker. `wandb.log({...})` is called once per PPO iteration; every metric you saw in the dashboard (reward curves, termination rates, curriculum ranges) came through that call. Data stored in the cloud at `api.wandb.ai` and also locally cached at `~/robotic-simulation/wandb-cache/wandb/run-20260417_184630-qtk6gyny/`.

### Viser 1.0.26

Headless browser-based 3D viewer from the Nerfstudio team at UC Berkeley. Uses WebSockets + WebGL — no X display required on the server side, which is why it works over SSH and on headless boxes. mjlab creates a Viser scene during `play.py`, pushes MuJoCo state to the browser every rendered frame.

### The Unitree G1

Hangzhou-based Unitree Robotics' consumer/research-tier humanoid. ~1.3 m tall, ~35 kg, released 2024. Commercial price ~$16–20k. We used only its MJCF model — no real hardware touched. The MJCF defines 29 degrees of freedom that the policy learned to coordinate.

---

## Baselines / Pretrained Models — Exhaustive List

| Asset | Where it came from | Used for |
|---|---|---|
| `demo_ckpt.pt` (tracking policy) | mjlab maintainers, hosted on Google Cloud Storage | The cartwheel smoke test only (Phase D). NOT connected to our velocity training. |
| `demo_motion.npz` (cartwheel motion clip) | Same | Smoke test only. |
| `model_2050.pt` (our velocity policy) | **Trained from scratch on your Spark this session** | The walking demo you're watching now. |
| Unitree G1 MJCF | Bundled in mjlab's repo | Physics/rendering only — not a model in the ML sense. |

Our velocity policy started from a randomly-initialized neural network at iteration 0. Every bit of walking behavior is learned.

---

## Artifacts on Disk

- **Checkpoints:** `~/robotic-simulation/mjlab/logs/rsl_rl/g1_velocity/2026-04-17_18-46-23/model_*.pt` — 42 files, one every 50 iterations. `model_2050.pt` (5.1 MB) is the freshest.
- **W&B local cache:** `~/robotic-simulation/wandb-cache/wandb/run-20260417_184630-qtk6gyny/`
- **Training log:** `~/robotic-simulation/logs/train-baseline-20260417-184620.log`
- **Setup notes:** `~/robotic-simulation/setup-notes.md` — running log of every deviation and decision, useful for future sessions.
- **Container:** `mjlab-dev` is running with the full installed stack; `docker exec -it mjlab-dev bash` to enter it anytime.

---

## What's Ready for the Next Session

The container is live, the stack is installed, the Warp kernel cache is warm, W&B is authenticated, and we have a baseline checkpoint. For any new training run, the short list is:

1. Quiesce host: `docker stop open-webui compose-arangodb-1 ollama-compose && sudo systemctl stop comfyui.service && sudo swapoff -a`
2. (Optional) `git checkout -b <experiment-name>` in `/workspace/mjlab` if we're tweaking configs.
3. Launch the training command.
4. Monitor W&B.
5. Stop cleanly via `docker exec mjlab-dev bash -c 'kill -INT $(pgrep -f mjlab.scripts.train)'`.
6. Restore host: `sudo swapon -a && sudo systemctl start comfyui.service && docker start ollama-compose open-webui compose-arangodb-1`.

No reinstall needed.

---
