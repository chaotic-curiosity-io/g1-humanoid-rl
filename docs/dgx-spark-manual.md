# Robot Simulation & RL Training on NVIDIA DGX Spark

## A Practical Manual for Chaotic Curiosity

**Author:** Generated for Don Balanzat
**Date:** April 2026
**Hardware Target:** NVIDIA DGX Spark (GB10 Grace Blackwell Superchip)
**Goal:** Train a Unitree G1 velocity-tracking policy from scratch and showcase the result

**Source:** Google Doc `16L1rR7fo_XdKKl7PdE4ynwMsbAOFfdeEuCVg3ItFXSY` (saved locally 2026-04-17 so we don't need to re-fetch).

---

## 1. The Honest Compatibility Picture

Before touching the keyboard, you need to understand why the DGX Spark is a different animal from a standard GPU workstation for this specific workload.

**The DGX Spark is:**

- ARM64 (aarch64) — not x86_64
- CUDA 13.0 — not CUDA 12.x
- Blackwell GPU, compute capability sm_121 — not Ampere/Ada
- Unified memory (128 GB shared CPU/GPU pool) — not discrete VRAM
- Running DGX OS (Ubuntu 24.04 base)

**Why this matters for robot simulation:**

The mjlab → MuJoCo Warp → NVIDIA Warp → PyTorch → CUDA dependency chain needs every link to work on aarch64 + CUDA 13 + sm_121. As of April 2026, this ecosystem is still catching up:

- **PyTorch:** aarch64 + CUDA 13 wheels exist (cu130 index). This works.
- **NVIDIA Warp:** Officially supports ARMv8 on Linux. Nightly builds support CUDA 13. Prebuilt pip wheels are CUDA 12 and will fail. You need the nightly or a source build.
- **MuJoCo Warp:** Depends on NVIDIA Warp. No prebuilt aarch64 + CUDA 13 wheel. Must install from source (git).
- **mjlab:** Pure Python on top of the above. Once dependencies work, mjlab itself is fine.

**Bottom line:** This is a "build from source inside an NGC container" situation, not a "pip install and go" situation. Budget 1–2 hours for environment setup vs. 15 minutes on x86. The training itself runs fine once the stack is up.

**If you want the fastest possible path to a demo:** Use the RTX 3090 Ti workstation (x86, CUDA 12, everything pip-installable). Then replay/present the results on the Spark if needed. The manual below covers both paths, Spark-native first.

## 2. DGX Spark Environment Setup

### 2.1 Verify Your Hardware

SSH into the Spark (or open a local terminal if you have a monitor connected):

```bash
# Confirm you're on ARM
uname -m
# Expected: aarch64

# Confirm GPU is visible
nvidia-smi
# Expected: NVIDIA GB10, Driver 580.x or newer

# Confirm CUDA version
nvcc --version
# Expected: CUDA 13.0.x
# If nvcc is not found, see Section 2.2
```

If nvidia-smi works but nvcc doesn't, the CUDA toolkit may not be installed or not on your PATH. The DGX OS ships with the driver but the toolkit may need activation:

```bash
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Add to ~/.bashrc so it persists
echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
```

### 2.2 Disable Swap (Critical for Unified Memory)

On the Spark, GPU memory IS system memory. If a training run exceeds available RAM, the system doesn't throw a clean CUDA OOM error — it enters a swap death spiral that can brick the machine until you hard-reboot.

```bash
# Disable swap immediately
sudo swapoff -a

# Verify it's off
swapon --show
# Should print nothing

# Make permanent: comment out swap entries in fstab
sudo nano /etc/fstab
# Comment out any line containing "swap"
```

This converts "machine freezes, SSH dies, hard reboot" into "job dies, machine lives." Much better.

### 2.3 Install Docker + NVIDIA Container Toolkit

Docker should be preinstalled on DGX OS. Verify:

```bash
docker --version
# If not installed: sudo apt update && sudo apt install -y docker.io

# Verify GPU passthrough works
docker run --rm --gpus=all nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi
# Should show your GB10
```

If the GPU test fails, install the NVIDIA Container Toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 3. The NGC Container Approach (Recommended for Spark)

The NVIDIA NGC PyTorch container is the safest base — it ships with aarch64 + CUDA 13 + Blackwell-optimized PyTorch pre-built. We install everything else on top.

### 3.1 Pull the NGC PyTorch Container

Use the latest available tag. As of April 2026, try the most recent monthly release:

```bash
# Pull the latest NGC PyTorch image
# Check https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch for current tags
docker pull nvcr.io/nvidia/pytorch:25.12-py3
```

If that specific tag doesn't exist, list available tags or try 25.11-py3, 26.01-py3, 26.03-py3, etc. The key requirement is that the image supports CUDA 13 + aarch64.

### 3.2 Create a Persistent Workspace

```bash
# Create a directory on the Spark for your work
mkdir -p ~/robot-sim
mkdir -p ~/robot-sim/wandb-cache
mkdir -p ~/robot-sim/logs
```

### 3.3 Launch the Container

```bash
docker run -it --gpus=all \
  --name mjlab-dev \
  --shm-size=16g \
  -v ~/robot-sim:/workspace \
  -p 8080:8080 \
  -p 8888:8888 \
  -e WANDB_ENTITY=your-wandb-username \
  -e WANDB_DIR=/workspace/wandb-cache \
  -e MUJOCO_GL=egl \
  -e TORCH_CUDA_ARCH_LIST="12.1a" \
  nvcr.io/nvidia/pytorch:25.12-py3
```

Flags explained:

- `--shm-size=16g` — shared memory for parallel dataloader workers
- `-v ~/robot-sim:/workspace` — your work persists outside the container
- `-p 8080:8080` — for the Viser web viewer
- `-p 8888:8888` — spare port for Jupyter if needed
- `TORCH_CUDA_ARCH_LIST="12.1a"` — tells any from-source builds to target Blackwell

You're now inside the container with a working PyTorch + CUDA 13 + aarch64 environment.

### 3.4 Verify PyTorch + CUDA Inside the Container

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
"
```

Expected output includes `CUDA available: True` and `GPU: NVIDIA GB10`. You may see a warning about compute capability 12.1 not being in the supported list — this is safe to ignore (sm_120 and sm_121 are binary compatible).

If CUDA is not available, the container image doesn't support your hardware. Try a newer NGC tag.

### 3.5 Install uv (Package Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

### 3.6 Install NVIDIA Warp (Nightly, for CUDA 13)

The stable pip release of Warp is compiled against CUDA 12. You need the nightly:

```bash
pip install -U --pre warp-lang --extra-index-url=https://pypi.nvidia.com/
```

Verify:

```bash
python -c "
import warp as wp
wp.init()
print(f'Warp version: {wp.__version__}')
print(f'CUDA Toolkit: {wp.get_cuda_toolkit_version()}')
"
```

If this prints without errors and shows CUDA 13, you're golden. If it fails with a `libcudart.so.12` error, the nightly doesn't yet have aarch64 + CUDA 13 wheels. In that case, build from source:

```bash
git clone https://github.com/NVIDIA/warp.git /workspace/warp
cd /workspace/warp
pip install -e .
python build_lib.py --cuda_path /usr/local/cuda
pip install -e .
```

### 3.7 Install MuJoCo Warp

```bash
pip install "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp"
```

Verify:

```bash
python -c "import mujoco_warp; print('MuJoCo Warp OK')"
```

### 3.8 Install mjlab

```bash
cd /workspace
git clone https://github.com/mujocolab/mjlab.git
cd mjlab
pip install -e .
```

If there are dependency conflicts (mjlab pins a specific PyTorch version that conflicts with the NGC image), install with relaxed constraints:

```bash
pip install -e . --no-deps
# Then manually install any missing deps mjlab needs (rsl-rl, viser, etc.)
pip install rsl-rl viser tyro wandb
```

### 3.9 Install Weights & Biases

```bash
pip install wandb
wandb login
# Paste your API key from wandb.ai/authorize
```

## 4. Smoke Test

Still inside the container:

```bash
cd /workspace/mjlab
python -m mjlab.scripts.demo
```

This should open a Viser viewer. From your local machine (or another machine on the network), open a browser to `http://<spark-ip>:8080`. You should see G1 humanoids dancing.

**If the demo fails**, read the error carefully:

- `ModuleNotFoundError` → a dependency is missing, install it
- CUDA error → Warp/MuJoCo Warp didn't compile for sm_121. Rebuild Warp from source with `TORCH_CUDA_ARCH_LIST="12.1a"`
- EGL errors → install `sudo apt install -y libegl1 libgl1` inside the container
- Viewer doesn't load → check that port 8080 is forwarded. If SSH, use: `ssh -L 8080:localhost:8080 nvidia@<spark-ip>`

**If the demo works, everything else is straightforward.**

## 5. Baseline Training Run

```bash
cd /workspace/mjlab

MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
  python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Unitree-G1 \
  --env.scene.num-envs 2048
```

Note: start with num-envs 2048 (not 4096) on the Spark. Unified memory means the training job competes with the OS for RAM. If it runs cleanly, you can bump to 4096 on the next run.

### What You'll See

**First 60–90 seconds:** MuJoCo Warp compiling CUDA kernels (JIT). Terminal looks frozen. This is normal. Do not Ctrl+C.

**Then, iteration logs:**

```
Iteration   10/15000 | Mean Reward: -3.21 | Mean Episode Len: 42 | FPS: 18432
Iteration   50/15000 | Mean Reward: -0.87 | Mean Episode Len: 120 | FPS: 22100
Iteration  200/15000 | Mean Reward:  1.45 | Mean Episode Len: 380 | FPS: 23500
Iteration  500/15000 | Mean Reward:  4.12 | Mean Episode Len: 900 | FPS: 24000
```

**Reading the numbers:**

- **Mean Reward** starts negative (robot falls immediately, gets penalized). It should climb steadily and plateau positive. This is your "learning curve."
- **Mean Episode Length** starts short (robot falls fast) and grows as it learns to balance.
- **FPS** (frames per second across all envs) measures sim throughput. Higher is faster training.

### When to Stop

For a demo-quality walking policy, 1000–2000 iterations is usually sufficient. On the Spark at 2048 envs, expect this to take 45–90 minutes. Watch the W&B dashboard — when the reward curve flattens, you're done.

Ctrl+C to stop. Checkpoints auto-save to the `logs/` directory and to W&B.

**Note the W&B run path** — visible in terminal output or the W&B URL bar. Format: `your-username/mjlab/abc12345`. You need this to replay the policy.

## 6. The Tweak: Make It Sprint

### 6.1 Find the Config File

```bash
cd /workspace/mjlab
grep -rn "lin_vel_x" src/mjlab/tasks/velocity/
```

This will print something like:

```
src/mjlab/tasks/velocity/config/unitree_g1/flat_env_cfg.py:47:    lin_vel_x=(-1.0, 1.0),
```

Open that file in your editor.

### 6.2 Edit the Velocity Range

Find the block that looks like:

```python
ranges=UniformVelocityCommandCfg.Ranges(
    lin_vel_x=(-1.0, 1.0),
    lin_vel_y=(-1.0, 1.0),
    ang_vel_z=(-0.5, 0.5),
),
```

Change `lin_vel_x` to:

```python
    lin_vel_x=(-2.5, 2.5),
```

Save. That's it. You just told the RL algorithm to command the robot at speeds up to 2.5 m/s (a brisk jog) instead of 1.0 m/s (a casual walk).

### 6.3 Train the Tweaked Version

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
  python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Unitree-G1 \
  --env.scene.num-envs 2048
```

Same command. mjlab picks up the modified config. Note this second W&B run path.

**What to expect:** The reward curve will rise more slowly and plateau lower than the baseline. The higher velocity range is a harder task — the robot needs to learn a fundamentally different gait (sprinting is biomechanically distinct from walking). This is the interesting part of your demo narrative.

## 7. Showcase the Results

### 7.1 Play Back the Baseline

```bash
cd /workspace/mjlab

# Revert the config change first
git checkout -- src/mjlab/tasks/velocity/

# Play the baseline policy
python -m mjlab.scripts.play \
  Mjlab-Velocity-Flat-Unitree-G1 \
  --wandb-run-path your-username/mjlab/<baseline-run-id> \
  --num-envs 8
```

Open `http://<spark-ip>:8080` in a browser. You'll see 8 G1s walking.

### 7.2 Play Back the Sprinter

In a second terminal (or after stopping the first):

```bash
# Re-apply the tweak
# (edit the file again, or use git stash/branch)

python -m mjlab.scripts.play \
  Mjlab-Velocity-Flat-Unitree-G1 \
  --wandb-run-path your-username/mjlab/<sprinter-run-id> \
  --num-envs 8
```

### 7.3 Capture the Demo

**Screen recording:** Use OBS on whatever machine has the browser open, or the built-in screen recorder on macOS/Windows. Record 30 seconds of each.

**W&B comparison chart:** On wandb.ai, select both runs, click "Create Report," drag the "Mean Reward" chart in. Export as PNG for slides.

### 7.4 The Four-Beat Narrative

1. "This is GPU-accelerated reinforcement learning — 2048 Unitree G1 humanoids learning in parallel on the DGX Spark's Blackwell GPU"
2. [W&B reward curve comparison] "Both policies used identical PPO, identical reward functions, identical neural network architecture. The only difference is the commanded velocity range"
3. [Baseline video] "After about an hour of training, the baseline policy walks confidently"
4. [Sprinter video] "The sprinter policy learned a completely different gait — wider stance, more aggressive arm swing — emergent from the same system, same algorithm, one config change"

## 8. Troubleshooting

### Container Won't Start with GPU

```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

Fix: Install NVIDIA Container Toolkit (Section 2.3).

### libcudart.so.12: cannot open shared object file

A dependency was compiled against CUDA 12. You're on CUDA 13. Fix: Install the offending package from the cu130 index or build from source:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

### CUDA error: no kernel image is available for execution on the device

A CUDA kernel wasn't compiled for sm_121. Fix: Set `TORCH_CUDA_ARCH_LIST="12.1a"` and rebuild:

```bash
export TORCH_CUDA_ARCH_LIST="12.1a"
pip install --no-build-isolation --force-reinstall <package>
```

### Machine Freezes During Training (No SSH, No Response)

OOM on unified memory caused a swap death spiral. Fix: Hard reboot (hold power button), then disable swap (Section 2.2), then reduce `--env.scene.num-envs` to 1024 or 512.

### Viser Viewer Won't Load in Browser

Port 8080 isn't reachable from your machine. Fix: If using SSH, add port forwarding:

```bash
ssh -L 8080:localhost:8080 nvidia@<spark-ip>
```

Then open `http://localhost:8080` on your local machine.

### nvidia-smi Shows "Memory-Usage: N/A"

Normal on unified memory systems. The GPU shares DRAM with the CPU. Use `free -h` to monitor total system memory consumption instead.

### MuJoCo Warp Compilation Takes Forever (>5 minutes)

First-run JIT compilation. Subsequent runs are cached in `~/.cache/warp/`. If you destroy/recreate the container, the cache is lost. Mount it as a volume:

```
-v ~/robot-sim/warp-cache:/root/.cache/warp
```

### PyTorch Warning About Compute Capability 12.1

```
Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
```

This is safe to ignore. sm_120 and sm_121 are binary compatible.

## 9. Fallback: RTX 3090 Ti Workstation

If the Spark environment setup hits a wall you can't resolve in 30 minutes, pivot to the 3090 Ti. Everything works out of the box on x86 + CUDA 12:

```bash
# On the 3090 Ti workstation (Ubuntu)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Smoke test (zero install)
uvx --from mjlab --refresh demo

# Full install
git clone https://github.com/mujocolab/mjlab.git && cd mjlab
uv sync

# Train
MUJOCO_GL=egl uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096

# Play
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-username/mjlab/<run-id>
```

That's it. No containers, no source builds, no CUDA version juggling. The 3090 Ti will also train faster per-iteration thanks to higher memory bandwidth (936 GB/s GDDR6X vs. 273 GB/s LPDDR5x on the Spark).

You can still present the demo on the Spark by copying the trained checkpoint and running play there — the showcase doesn't need to run where the training happened.

## 10. What Each Piece Does (The Systems View)

For when someone asks "but how does it actually work":

**Physics Engine (MuJoCo Warp):** Simulates rigid-body dynamics — gravity, joint torques, ground contact, friction. Each of the 2048+ parallel environments is an independent physics world. MuJoCo is the gold standard for contact-rich robotics simulation. "Warp" means GPU-accelerated via NVIDIA's Warp framework.

**Robot Model (MJCF):** An XML file describing the G1's kinematic tree — 29 joints, their limits, masses, inertias, actuator properties. This is the digital twin. MuJoCo reads this to know "what" to simulate.

**RL Algorithm (PPO via RSL-RL):** Proximal Policy Optimization. A neural network (the "policy") receives observations (joint angles, velocities, gravity direction, commanded velocity) and outputs joint torques. PPO collects rollouts across all parallel envs, computes advantages, and updates the network weights. This is the "learning" part.

**Reward Function:** A weighted sum of terms that define "good behavior" — track the commanded velocity (positive reward), don't use excessive torque (penalty), don't fall (termination + penalty), keep joints near default pose (regularization). The specific weights and terms are in the task config file — the same file where you changed the velocity range.

**Sim-to-Real Gap:** The policy trains in simulation where physics is approximate. Real motors have latency, real floors have uneven friction, real IMUs have noise. Domain randomization (randomizing friction, mass, delays during training) helps the policy generalize. The sim2sim step (replaying in a different simulator like standalone MuJoCo) is a cheap sanity check before touching real hardware.

## 11. Next Steps After the Demo

Once you've shown you can train a walking policy, natural follow-ups:

- **Terrain curriculum:** Train on stairs, slopes, and rough ground (mjlab has a terrain generator). Much more visually impressive and relevant to real deployment.
- **Motion imitation:** Use the tracking task to teach the G1 to reproduce human motion capture data. Requires BeyondMimic preprocessing.
- **Sim2sim validation:** Run the trained ONNX policy in unitree_mujoco to verify it transfers across simulators.
- **Real deployment:** Flash the ONNX to a physical G1 or H2 via unitree_sdk2. The unitree_rl_mjlab repo has the full deployment pipeline.
- **Multi-robot comparison:** Train the same task on Go2 (quadruped) and G1 (humanoid) side by side. Different morphologies, same algorithm — great for a workshop demo.

## Appendix: Key Links

| Resource | URL |
|---|---|
| mjlab repo | https://github.com/mujocolab/mjlab |
| mjlab docs | https://mujocolab.github.io/mjlab |
| unitree_rl_mjlab | https://github.com/unitreerobotics/unitree_rl_mjlab |
| unitree_mujoco | https://github.com/unitreerobotics/unitree_mujoco |
| NVIDIA Warp | https://github.com/NVIDIA/warp |
| MuJoCo Warp | https://github.com/google-deepmind/mujoco_warp |
| DGX Spark setup guide (natolambert) | https://github.com/natolambert/dgx-spark-setup |
| NGC PyTorch containers | https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch |
| W&B | https://wandb.ai |
| DGX Spark User Guide | https://docs.nvidia.com/dgx/dgx-spark/ |
