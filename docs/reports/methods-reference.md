# Methods & Techniques Reference

*A lookup handbook for the G1 humanoid RL series. Companion to the [reports index](README.md).*

---

**Who this is for:** anyone using this repo's tools and wanting a single place to look up reward terms, pipeline steps, known gotchas, and operations procedures. No robotics or ML background assumed — every term is defined on first use.

**How to use it:** jump to the section you need. Nothing here is a narrative; it is designed to be scanned, not read top-to-bottom.

---

## Contents

1. [Reward-term catalog](#1-reward-term-catalog)
2. [Terminations & thresholds](#2-terminations--thresholds)
3. [The retargeting pipeline](#3-the-retargeting-pipeline)
4. [Curricula](#4-curricula)
5. [Domain randomization](#5-domain-randomization)
6. [Recording gotchas](#6-recording-gotchas)
7. [Helper-script index](#7-helper-script-index)
8. [Ops & safety](#8-ops--safety)

---

## 1. Reward-term catalog

A **reward term** is a single number added to the score the robot receives each simulation step. The policy (the neural network controlling the robot) learns to make that score as large as possible. These are the terms used in this repo's two task families.

### Velocity-tracking terms (`Mjlab-Velocity-Flat-Unitree-G1` / `Mjlab-Velocity-Rough-Unitree-G1`)

In velocity tracking, the robot is given a commanded walking speed and rewarded for matching it. The following terms make up the reward (verified against the baseline run's `params/env.yaml` on the Spark).

| Term | What it rewards (plain language) |
|---|---|
| `track_linear_velocity` | Walking at the commanded forward/sideways speed. The closer the actual speed to the commanded speed, the higher the score. This is the primary driving signal. |
| `track_angular_velocity` | Turning at the commanded yaw rate. Same idea: match the commanded spin. |
| `air_time` | Spending time in the air between steps. Rewards the policy for lifting each foot cleanly off the ground rather than shuffling or dragging. A higher weight on this term encourages a true flight phase (both feet briefly off the ground). CLI flag: `--env.rewards.air_time.weight`. |
| `upright` | Keeping the torso vertical. Discourages the robot from leaning far forward or sideways. |
| `pose` | Keeping joint angles close to a nominal reference pose. Discourages extreme or unnatural limb positions. |
| `body_ang_vel` | Penalizes excessive angular velocity of the base body. Encourages stable, controlled motion rather than spinning or tumbling. (Negative term.) |
| `angular_momentum` | Penalizes whole-body angular momentum. Encourages the robot to move without spinning up undesirable rotational motion. (Negative term.) |
| `dof_pos_limits` | Penalizes joint angles that approach or exceed their hardware limits. Protects actuators and encourages physically realistic motion. (Negative term.) |
| `action_rate_l2` | Penalizes large, jerky changes in joint commands between steps. Encourages smooth, consistent motion. (Negative term.) |
| `foot_clearance` | Rewards lifting each foot sufficiently during the swing phase to clear the ground. Discourages dragging or shuffling steps. |
| `foot_swing_height` | Rewards appropriate foot height during the swing arc. Encourages a natural step trajectory. |
| `foot_slip` | Penalizes lateral foot movement while the foot is in contact with the ground. Encourages clean ground contact without sliding. (Negative term.) |
| `soft_landing` | Rewards gentle foot-to-ground contact velocity. Discourages heavy, stomping landings. |
| `self_collisions` | Penalizes the robot's links colliding with each other. Discourages physically impossible self-intersecting poses. (Negative term.) |

**Note on base height:** there is **no standalone `base_height` reward term** in the velocity task. Height is shaped indirectly through the `upright`, `pose`, and termination logic rather than a dedicated height penalty.

**Note on exact key paths:** the terms and CLI paths above are confirmed by reading `params/env.yaml` inside the baseline run directory on the Spark (`logs/rsl_rl/g1_velocity/2026-04-17_18-46-23/`). Example override: `--env.rewards.air_time.weight 2.0`.

### Motion-tracking terms (`Mjlab-Tracking-Flat-Unitree-G1`)

In motion tracking, the robot is given a reference motion (a recording of a motion mapped onto the G1's body) and rewarded for matching it frame-by-frame. The two termination-related terms below are also what drive the **termination** logic (see Section 2).

| Term | What it rewards (plain language) |
|---|---|
| `anchor_pos` | Keeping the pelvis close to the reference pelvis position at each frame. The policy loses the episode — the simulation resets — if the pelvis drifts further than the threshold (default 0.25 m; raised to 0.5 m for aerial motions). |
| `ee_body_pos` | Keeping the end-effectors (feet and wrists) close to their reference positions at each frame. Same threshold applies. A cartwheel involves feet and wrists sweeping through wide arcs, so this term is the usual culprit in early-termination failures. |

---

## 2. Terminations & thresholds

A **termination** ends the current episode early and resets the robot to its starting position. Terminations are not failures in the training sense — the policy still learns from terminated episodes. But if terminations fire *too often*, the policy never sees the end of the motion it is trying to learn, and learning stalls.

### The 0.25 → 0.5 m aerial lesson

The stock threshold for both `anchor_pos` and `ee_body_pos` is **0.25 m** — if the robot's pelvis or any end-effector drifts more than 25 cm from the reference, the episode resets.

For motions that involve the arms and legs sweeping through wide arcs — cartwheels, flips, any aerial phase — 0.25 m is too tight. During a cartwheel the hands swing through a full arc; the wrists are almost never within 25 cm of the reference during the airborne phase. The episode kept resetting mid-flip, so the policy never received the reward for completing the motion. Training plateaued at a low score and no full flips were learned (Iter A failure, cartwheel training).

**The fix:** raise both thresholds to **0.5 m** to give the policy enough room to complete the aerial phase and learn from the full trajectory. With 0.5 m thresholds, the cartwheel policy converged within 2000 extra training iterations.

### Critical rule: render at the training threshold (or with terminations off)

**If you train with 0.5 m thresholds, you must also evaluate/render with 0.5 m thresholds — or disable terminations entirely.**

Using the default 0.25 m thresholds at render time when the policy was trained at 0.5 m will cut every episode short mid-flip, making a successful policy *look* like it fails. This is exactly what happened in the Iter B retraction: the render environment still used 0.25 m, so every cartwheel attempt was cut off before the landing, and the video appeared to show failures.

`record_policy.py` has two flags to handle this:

```sh
--termination-threshold 0.5   # match training threshold exactly
--disable-terminations        # remove all non-timeout terminations (best for continuous eval)
```

Use `--disable-terminations` for a clean "what does the policy actually do" evaluation; use `--termination-threshold` when you want to replicate the exact training condition.

### Score-spoofer warning

Raising thresholds to allow aerial motion also opens a loophole: a policy that simply *collapses* to the ground and rolls through 180° of body rotation can accumulate roll-angle "inversion" signal without actually completing a flip. The automated scorer (`score_cartwheel.py`) is roll-angle-only and was fooled by this exact crash-roll in Iter B. **Always pair any numeric score with frame-by-frame visual review.** A good number that looks wrong on video means the scorer is being spoofed.

---

## 3. The retargeting pipeline

**Retargeting** is the process of taking a human motion recording and mathematically mapping it onto the robot's body — scaling the proportions and converting the joint angles to match the G1's skeleton. The output is a file the tracking trainer can use as a reference.

### The full pipeline

```
Source motion (.pkl)
        ↓
  pkl_to_csv.py          (in g1_spinkick_example repo, or smpl_backflip_to_g1.py for SMPL sources)
        ↓
  Intermediate CSV
        ↓
  mjlab.scripts.csv_to_npz
        ↓
  motion.npz             (the reference motion file for the tracking trainer)
        ↓
  mjlab.scripts.train Mjlab-Tracking-Flat-Unitree-G1
        --env.commands.motion.motion-file <path/to/motion.npz>
```

**Step-by-step:**

1. **Source motion (.pkl):** A MimicKit `.pkl` file containing the retargeted motion. MimicKit ships a set of pre-retargeted G1 motions (including a cartwheel); these are hand-curated and work better for acrobatics than automatic retargeters.

2. **pkl_to_csv.py** (from the `g1_spinkick_example` repo): converts the `.pkl` into a row-per-frame CSV. **Watch the `--duration` flag** — a bug here once produced a double-length reference (two back-to-back cartwheels), and the policy could not finish even one. Verify the output CSV frame count against the expected motion length.

3. **smpl_backflip_to_g1.py** (in `scripts/`): an alternative for SMPL-format source motions (e.g. a backflip captured from a video). Bypasses SMPL-X body-model weights by using MimicKit's `smpl.xml` MJCF for forward kinematics, then calls GMR's IK retargeter to produce a G1-compatible `.pkl`. That `.pkl` then follows the same `pkl_to_csv` → `csv_to_npz` path above.

4. **csv_to_npz** (`python -m mjlab.scripts.csv_to_npz`): converts the CSV into the binary `.npz` format the tracking trainer expects.

5. **Training:** pass `--env.commands.motion.motion-file <path/to/motion.npz>` to the trainer.

### Why hand-curated retargets beat automatic ones for acrobatics

Automatic SMPL-X → humanoid retargeters degrade during aerial phases — the body model loses reliable contact constraints, and the IK solution can drift into physically implausible poses. MimicKit's shipped G1 set was hand-curated with those phases in mind. For acrobatics, always try a MimicKit retarget first; fall back to `smpl_backflip_to_g1.py` only when the motion is not in the MimicKit set.

---

## 4. Curricula

A **curriculum** automatically adjusts the training conditions as learning progresses — starting easy and gradually making the task harder. Mjlab's velocity task has two curriculum mechanisms.

### The velocity-range curriculum — and the "clobber" gotcha

The Flat velocity task has an active `command_vel` curriculum even though "flat = curriculum off" sounds intuitive. The curriculum runs in **three stages** (stage 0, 1, 2). Stage 0 fires at every episode boundary and resets the commanded `lin_vel_x` range back to `(-1.0, 1.0)` m/s — overwriting any command-range override you set on the command line.

**The fix:** override both the command range *and* all three curriculum stages, using `=` syntax for tuples:

```sh
"--env.commands.twist.ranges.lin-vel-x=(-0.5, 0.5)" \
"--env.curriculum.command-vel.params.velocity-stages.0.lin-vel-x=(-0.5, 0.5)" \
"--env.curriculum.command-vel.params.velocity-stages.1.lin-vel-x=(-0.5, 0.5)" \
"--env.curriculum.command-vel.params.velocity-stages.2.lin-vel-x=(-0.5, 0.5)"
```

Replace `(-0.5, 0.5)` with your target range. The `=` syntax (not a space) is required for tuple-type arguments; without it the argument parser rejects the value silently.

**Verify** the override landed by reading `params/env.yaml` in the saved run directory:

```sh
ssh spark "docker exec mjlab-dev bash -lc 'cat /workspace/mjlab/logs/rsl_rl/g1_velocity/<timestamp>/params/env.yaml'"
```

If the `lin_vel_x` range still shows `(-1.0, 1.0)`, the curriculum clobber is still active and the override did not take effect.

**Important:** the exact YAML paths for the three curriculum-stage overrides (`velocity-stages.0`, `velocity-stages.1`, `velocity-stages.2`) should be confirmed by reading the live env config on the Spark — verify against `params/env.yaml` before a long run.

### The terrain curriculum (`Mjlab-Velocity-Rough-Unitree-G1`)

The Rough variant adds a **terrain curriculum**: the robot begins on flat ground and, as it learns to walk, the terrain procedurally becomes more challenging (slopes, steps, rubble). The terrain type is stored in `params/env.yaml` under `terrain_type`; a value of `plane` means the Flat variant ran (no terrain curriculum).

Both Flat and Rough share `experiment_name="g1_velocity"`, so the log directory name alone does not tell you which variant produced a run. Always check `params/env.yaml`.

---

## 5. Domain randomization

**Domain randomization** deliberately varies physical parameters (friction, mass, motor gains, etc.) during training so the policy learns to handle a range of conditions rather than a single fixed world. This makes policies more robust.

The mjlab G1 tasks use domain randomization by default on parameters such as ground friction, link masses, and motor properties. The exact set of randomized parameters for each task is stored in `params/env.yaml` under the randomization section of the saved run config.

**What is known from this repo's training logs:**

- The Flat velocity task (Session 1 baseline, 2026-04-17) ran with default randomization settings; the policy achieved reward 50.5 and episode length 995/1000, indicating the policy was robust enough to walk stably under whatever randomization was active.
- The tracking task ran with default randomization; the cartwheel policy succeeded at Iter C.

**Uncertainty note:** the precise list of which parameters are randomized, and by how much, is not documented in this checkout. To see the exact settings for any run, read the `params/env.yaml` in that run's directory on the Spark. This section will be updated with concrete values once the Spark probes in Task 12 of the implementation plan are completed.

---

## 6. Recording gotchas

All recording happens inside `mjlab-dev` on the Spark via the offscreen renderer (`mjlab/src/mjlab/viewer/offscreen_renderer.py`). Several sharp edges are worth knowing before you start a recording session.

### Renderer state mutations (multi-angle recording)

The offscreen renderer mutates model properties when it initializes:

- `model.stat.extent` — bounding box used for camera distance calculations
- `model.vis.global_.offheight` / `model.vis.global_.offwidth` — render resolution
- `model.light_castshadow[:]` — shadow casting on/off

Because these are mutations (not copies), rebuilding the renderer between camera angles is unnecessary and slow. **The right pattern:** build the renderer once, then change the camera angle between renders by mutating `renderer._cam.azimuth`, `renderer._cam.elevation`, and `renderer._cam.distance` directly. Do not rebuild the renderer per angle.

To load a new checkpoint into an existing runner without rebuilding the environment:

```python
runner.load(path, load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device)
```

### Shadow-map acne (jittery lines on the ground)

`enable_shadows=True` combined with a tight tracking-camera extent produces **shadow-map acne**: jittery horizontal lines that flicker frame-to-frame on the ground plane. This looks terrible in video.

**Fix options (in order of preference):**
1. Pass `--no-shadows` to `record_learning_progression.py` (or set `enable_shadows=False` in the renderer config).
2. Increase shadow quality: `model.vis.quality.shadowsize = 4096` or higher.

### Debug-visualizer lines on the ground

The environment's manager and sensor debug visualizers (the command-velocity arrow, foot-height scan rings, contact markers) are redrawn every render call. These appear as colored lines on the ground and are the most common source of "there are lines in my video" complaints.

There is no upstream CLI flag to disable them. The workaround is to monkey-patch the update method before the render loop:

```python
env.update_visualizers = lambda *a, **k: None
```

`record_learning_progression.py` exposes a `--no-debug-viz` flag that does this for you. For `record_policy.py`, patch manually or check the script's options.

### Holding a fixed velocity command while recording

The `UniformVelocityCommand` manager (called "twist") re-randomises the commanded velocity on every episode reset. If you want a consistent, fixed speed in a recording (e.g. always 1.5 m/s forward), you must override the command every step:

```python
# After each reset:
term.vel_command_b[:] = your_fixed_command
# Zero the mode flags so the re-randomiser doesn't fire:
is_heading[:] = 0
is_standing[:] = 0
is_world[:] = 0
is_forward_env[:] = 0

# After each env.step():
# Re-assert the command — it is recomputed from the mode flags every step.
term.vel_command_b[:] = your_fixed_command
```

This is already handled in `record_learning_progression.py` (which takes a fixed velocity seed). For `record_policy.py`, check the script's telemetry-dump path for the relevant implementation.

---

## 7. Helper-script index

All scripts live in `scripts/` in this checkout. They are mirrors of what runs on the Spark; they require `mjlab` (editable install) and/or `tensorboard` / `matplotlib`, so **they only execute inside `mjlab-dev`**, not on the Windows host.

To run an edited script on the Spark:

```sh
scp scripts/foo.py spark:robotic-simulation/scripts/
# or
docker cp /tmp/foo.py mjlab-dev:/workspace/scripts/
```

---

### `record_policy.py`

**What it does:** Headless multi-camera renderer for a single trained **tracking** policy. Drives a 1-environment play-mode session, runs the policy for N steps, and records offscreen frames from preset camera angles (chase, side, front, top). Writes one MP4 per angle plus an optional 2×2 grid composite. Also dumps a telemetry `.npz` (pelvis position/quaternion, reset signal, fps) for downstream scoring.

**Important: tracking-only.** This script requires a `--motion-file` reference motion and is hardcoded to `Mjlab-Tracking-Flat-Unitree-G1`. It does **not** support the velocity task. It is also not currently present inside `mjlab-dev` and must be `docker cp`'d in before use. For velocity-policy recording, use `record_learning_progression.py` (see below) — the dedicated velocity recorder is an open item in the S1 training plan.

**Key flags:**
- `--motion-file <path>` — required; path to the `motion.npz` reference file the tracking task was trained on.
- `--disable-terminations` — runs the policy without any non-timeout terminations (best for honest evaluation of what the policy actually does).
- `--termination-threshold <value>` — sets the `anchor_pos` / `ee_body_pos` thresholds at render time. Must match the training threshold; otherwise the render cuts episodes short.

**When to use it:** after training a tracking policy (e.g. cartwheel), for the A/B comparison clips in tracking reports, and for producing the telemetry file that feeds `score_cartwheel.py`. CPU-runnable while a training job owns the GPU.

---

### `record_learning_progression.py`

**What it does:** Multi-checkpoint, multi-camera progression recorder. Walks a training run directory, samples N evenly-spaced checkpoints, and for each one records a short deterministic rollout (same seed, same fixed velocity command). Stitches the clips into one MP4 per camera angle showing a single robot evolving from stumbling to walking. Also emits an optional 2×2 grid composite and labels each clip with the iteration number.

**Key flags:**
- `--no-shadows` — disables shadow rendering (avoids shadow-map acne).
- `--no-reflections` — disables reflections.
- `--no-debug-viz` — suppresses manager/sensor debug visualizers (removes lines from the ground).
- `--num-checkpoints N` — how many evenly-spaced checkpoints to sample (default 20).

**When to use it:** to create the "learning progression" videos embedded in the reports (e.g. the walking-arc reports). Used to produce the assets in `docs/reports/assets/`.

---

### `watch_learning.py`

**What it does:** Loads 8 checkpoints from a training run into a single Viser 3D viewer scene with 1024 parallel environments (128 per checkpoint). Each group of 128 robots runs a different checkpoint's policy simultaneously, so you can see the full range — random-weight chaos on the left, fully trained walking on the right — all in the same scene. Includes an orbit camera that rotates around the scene for cinematic demos.

**Access:** Viser runs at **host port `:8081`** (not `:8080` — open-webui owns that port). The trainer prints `http://localhost:8080`; translate to `:8081` when viewing from outside the container.

**When to use it:** during or after training, to visually compare multiple stages of learning side by side without loading separate environments. The TASK_ID and CKPT_DIR are hardcoded at the top of the file — edit them before use.

---

### `play_motion_npz.py`

**What it does:** Headless replay of a `motion.npz` reference file — drives the G1 environment's joint positions frame-by-frame from the npz data (root pose + joint angles) and renders multiple camera angles. No policy, no training — just visualizes the reference motion itself.

**When to use it:** to verify that a newly generated `motion.npz` looks correct *before* starting a tracking training run. If the reference motion looks wrong here, there is no point training on it.

---

### `plot_training_curves.py`

**What it does:** Reads RSL-RL TensorBoard event files from one or more training run directories, extracts scalar timeseries (mean reward, mean episode length, etc.), writes them to CSV, and renders PNG curve plots. Multiple runs can be overlaid on the same plot by passing `--run name=dir` multiple times.

**Example:**
```sh
python scripts/plot_training_curves.py \
    --run baseline=logs/rsl_rl/g1_velocity/2026-04-17_18-46-23 \
    --run fast_runner=logs/rsl_rl/g1_velocity/<new-run> \
    --tags Train/mean_reward Train/mean_episode_length \
    --out out/comparison
```

**When to use it:** to produce the training-curve PNGs embedded in the reports, and to compare runs (e.g. the A/B walking experiment in report 03).

---

### `score_cartwheel.py`

**What it does:** Scores cartwheel attempts from a `record_policy.py` telemetry dump. Reads the `.npz` telemetry file (pelvis position, pelvis quaternion, reset signal), segments it into episodes, and for each episode checks: (1) did the pelvis roll exceed 150° (full inversion?), and (2) did the roll recover to under 45° (landed upright?). Reports a per-episode summary and an overall completion count.

**Critical limitation: roll-angle-only, known spoofable by crashes.** A face-plant that rolls the body through 180° on the ground will be counted as an "inversion", and the first post-reset frame (robot standing at the start of the next episode) will be counted as "recovery". This is exactly what happened in the Iter B false-positive. **Never trust this scorer's output without frame-by-frame visual review of the rendered video.**

**When to use it:** as a fast first pass after recording, to get a rough episode count. Always follow up with video review.

---

### `smpl_backflip_to_g1.py`

**What it does:** Custom SMPL → G1 retargeter for SMPL-format source motions (e.g. a backflip captured from video). Bypasses SMPL-X body-model weights (which require licensed model files) by using MimicKit's shipped `smpl.xml` MJCF as the authoritative skeleton. Computes forward kinematics from the SMPL axis-angle pose stream, then calls GMR (General Motion Retargeting)'s IK retargeter to produce a G1-compatible `.pkl`. That `.pkl` then feeds the standard pipeline: `pkl_to_csv.py` → `csv_to_npz` → training.

**When to use it:** when the target motion is available as a SMPL `.pkl` but is not in MimicKit's pre-retargeted G1 set. For motions *in* the MimicKit G1 set (like the cartwheel), use MimicKit's retarget directly — hand-curated retargets beat automatic ones for acrobatics.

---

## 8. Ops & safety

### Host quiesce — freeing memory before a long run

The Spark's memory is shared (unified CPU+GPU memory). Co-tenant containers — `open-webui` (on port 8080), `compose-arangodb-1`, `ollama-compose`, and `comfyui` (as a systemd service) — all compete for that pool. Under memory contention, the system does not crash cleanly with a CUDA out-of-memory error; it enters a swap-death spiral that can require a hard reboot.

**Procedure before starting a long training run:**

```sh
# Stop co-tenant containers (no sudo needed):
docker stop open-webui compose-arangodb-1 ollama-compose

# Check available memory — aim for ~110 GiB free for a ~13 GiB training workload:
free -h

# After the run, restart them:
docker start open-webui compose-arangodb-1 ollama-compose
```

**Note on `sudo`:** `sudo` is **not** passwordless on the Spark. Commands like `swapoff -a` and `systemctl stop comfyui.service` require an interactive sudo session and cannot run non-interactively over SSH. Stop co-tenant *containers* (which need no sudo) and verify `free -h` headroom; that is usually sufficient for a 13 GiB workload with 110 GiB free.

---

### Stopping a training run cleanly

The training command is wrapped in a `bash -c '... > log'` shell by the container. Sending SIGINT to the wrapping shell does not reliably reach the Python child process on the first try.

**The correct approach — send SIGINT directly to the Python process:**

```sh
docker exec mjlab-dev bash -lc "kill -INT \$(pgrep -f mjlab.scripts.train)"
```

This finds the PID of the `mjlab.scripts.train` process and sends it a keyboard-interrupt signal, which triggers RSL-RL's clean shutdown (saves a final checkpoint, closes the TensorBoard writer).

---

### The bracket trick — safe watcher and counting loops

A bare `pgrep -f mjlab.scripts.train` or `grep mjlab.scripts.train` inside a shell loop **matches its own command line** — the grep process itself contains the search string. This causes watcher loops to loop forever and process-count checks to return one count too many.

**The fix:** use the bracket trick in grep patterns to prevent self-match:

```sh
# Wrong — matches the grep process itself:
ps -eo cmd | grep "python -u -m mjlab.scripts.train"

# Correct — the bracket breaks the literal match on the grep process:
ps -eo cmd | grep "[p]ython -u -m mjlab.scripts.train"
```

The `[p]` makes the pattern match `python` but does not match the literal string `[p]ython`, so the grep process is not included in the results. Use this pattern in any loop that checks whether a training job is running.

---

*This reference is maintained alongside the training campaigns. If you find an error or a missing detail, append a note to `setup-notes.md` on the Spark and update this file in the next commit.*
