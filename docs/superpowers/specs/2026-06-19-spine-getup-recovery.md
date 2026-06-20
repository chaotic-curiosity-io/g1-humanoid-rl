# Spec: S3 — Get-up / Fall Recovery (From-Scratch Task)

**Date:** 2026-06-19
**Branch:** `g1-skills-curriculum`
**Status:** Ready for training campaign (pending new-code sub-step)
**Track:** From-scratch tasks (spine task 3 of 4)
**Feeds into:** `docs/reports/getting-up.md`; linked from the syllabus `docs/reports/README.md`

---

## Goal

Train a Unitree G1 humanoid to stand up from a randomized fallen pose — entirely from scratch, with no reference motion and no movement command. The robot starts each episode lying on the ground (face-up, face-down, or crumpled at a random angle) and must discover, through trial and error, how to get itself upright and hold that position.

**What does "from scratch" mean here?** In the S1 running task, we shaped an existing locomotion reward — we already knew the robot could walk and we pushed the parameters. In the S2 backflip task, we gave the robot a frame-by-frame reference to imitate. Here, there is neither. The robot has no built-in concept of "upright," no motion to copy, and no forward-speed target. The only signal it receives is the reward we design: a number that goes up when it is closer to standing and goes down when it is flailing. Getting that reward design right is the entire engineering challenge — and the reason this task is the most likely of all four spine tasks to produce unexpected hacks.

**Why a get-up skill?** A robot that can fall and recover is far more useful (and safe) in the real world than one that only works if it starts already standing. This task is also the canonical entry point to the "from-scratch tasks" paradigm: behavior is 100% the reward terms you write.

**A natural connection to reward hacking:** Because the behavior is entirely determined by the reward, this task is the program's richest source of unintended shortcuts. A robot that learns to jiggle in place and occasionally trigger the "height rising" threshold, or that props itself up on one elbow to satisfy the "uprightness" term without ever reaching full standing — these are all plausible failure modes. The shaping journey is documented as an honest iteration log and cross-linked to the [S4 reward-hacking gallery spec](./2026-06-19-spine-reward-hacking-gallery.md).

This is the spine task for the **From-scratch tasks track** and requires the only genuinely new mjlab code in the entire four-spine program.

---

## Audience & voice

This report is written for someone with no background in robotics or machine learning. Every term is defined on first use.

- **Policy:** the neural network that controls the robot — its "brain." Every 20 ms it reads sensor data (joint angles, body velocity, orientation) and outputs motor commands.
- **Reward:** a number the training algorithm uses as feedback. Higher reward means the robot did something we wanted. The reward is computed at every timestep; the policy learns to maximize its total reward over the course of an episode.
- **Episode:** one uninterrupted simulation run. The robot starts in a fallen pose, the policy runs for up to a fixed time limit, and then the simulation resets.
- **Termination:** a rule that ends an episode early — for example, when the robot has been standing stably for long enough (success) or when time runs out (timeout). In this task, termination on *falling* is deliberately disabled: falling is where every episode begins, so it cannot be a failure condition.
- **Base height:** the height of the robot's pelvis (its center of mass, roughly) above the ground. A standing G1 has a base height around 0.8–0.9 m. Lying flat it is around 0.3–0.4 m. Rising base height is the primary signal that the robot is trying to stand up.
- **Torso uprightness:** how closely the robot's torso points straight up. Measured as how close the robot's "up" direction is to the world's "up" direction — 1.0 means perfectly upright, 0 means horizontal.
- **Joint velocity / torque:** how fast the motors are spinning and how hard they are pushing. Wild, uncontrolled thrashing produces high joint velocity and torque — which we penalize to encourage controlled, efficient movement.
- **Reward hacking:** when a policy finds a way to maximize its numerical score without doing the thing we actually wanted. See [S4 reward-hacking gallery](./2026-06-19-spine-reward-hacking-gallery.md) for concrete examples.
- **Task / reward manager:** in mjlab's code architecture, a "task" is a Python class that defines the environment (observation space, reward terms, termination conditions, initial state). A "reward manager" is the component inside the task that computes the numerical reward each timestep. Writing a new task means writing new Python code — this is what makes S3 the most engineering-intensive spine run.

Concrete analogies are preferred over mathematical notation. Each report is standalone: a reader who picks it up without reading the rest of the series can follow the reasoning.

---

## Staged-arc table

| Stage | What runs on the Spark | Estimated cost | Output |
|---|---|---|---|
| **Write the new task** | Inside `mjlab-dev`: write `Mjlab-Recovery-Flat-Unitree-G1` — a new task class that initializes the robot in a randomized fallen pose, defines the reward terms (height-rising + uprightness + stability; penalizes thrashing), sets terminations to success-hold or timeout (no falling termination). Run `ruff format && ruff check --fix && pyright` for lint/type-checking, then `pytest` for unit tests. | Minutes to hours (code writing, not GPU) | New task class committed in-container; clean lint + type checks; passing unit tests |
| **Smoke-train probe** | A short `mjlab.scripts.train Mjlab-Recovery-Flat-Unitree-G1` run (~100–200 iterations) to confirm the task registers, initializes, steps, and produces non-NaN rewards. Kill immediately once the reward curve shows any non-zero signal. | ~5–10 minutes GPU | Train log showing non-NaN rewards; no import errors; task initializes cleanly |
| **Full train (iterA — initial shaping)** | Full `mjlab.scripts.train Mjlab-Recovery-Flat-Unitree-G1` run with the designed reward terms, likely 3000–6000 iterations of iterative shaping. | ~3–6 h GPU per iteration; multiple iterations expected | Checkpoint `model_<N>.pt`; W&B run; `params/env.yaml` confirming reward weights |
| **Record + visual review** | `record_policy.py` on the best checkpoint; frame-by-frame review looking for genuine stand-up behavior vs a propped-up hack | ~10 minutes (CPU-runnable) | Multi-camera MP4; honest assessment of what behavior emerged |
| **Iterate if needed (iterB, iterC…)** | Close each reward hack found; adjust reward weights; retrain. Document each iteration and what it revealed. | Additional hours per iter | Improved checkpoint + iteration log (cross-linked to S4) |

---

## Concrete runs

All commands are issued from the Windows host and execute inside `mjlab-dev` on the Spark.

### Step 1 — Write the new task (in-container)

This is the engineering heart of S3. The new task `Mjlab-Recovery-Flat-Unitree-G1` must be written as a Python class inside the mjlab source tree on the Spark. Because `mjlab/` on the Spark is owned by root (the container writes into the bind-mount as root), **do not `sudo chown` the tree** — this desyncs container file ownership. Use one of the two safe paths from CLAUDE.md:

**Option A — `docker cp` from `/tmp` (recommended for new files):**

```bash
# Write the task file locally on the Windows box, then:
scp scripts/recovery_task.py spark:/tmp/recovery_task.py
ssh spark "docker cp /tmp/recovery_task.py mjlab-dev:/workspace/mjlab/src/mjlab/tasks/recovery_task.py"
```

**Option B — edit inside the container directly:**

```bash
ssh spark "docker exec -it mjlab-dev bash"
# Then inside the container:
# nano /workspace/mjlab/src/mjlab/tasks/recovery_task.py
```

mjlab is installed editable (`pip install -e .`), so new `.py` files under `src/mjlab/...` are picked up immediately — no reinstall needed.

**Reward design for the new task:**

The task must implement these reward terms and termination rules:

- **`base_height_rising`**: reward proportional to the robot's pelvis height above the ground; scaled so a standing pose yields maximum contribution.
- **`torso_upright`**: reward for the dot product between the robot's "up" vector and the world "up" vector — high when the torso is vertical, zero when horizontal.
- **`stability_bonus`**: a bonus term (activated only once base height and uprightness both exceed a threshold) that rewards the robot for holding still after standing up.
- **`joint_velocity_penalty`**: negative reward proportional to the mean absolute joint velocity — penalizes thrashing.
- **`joint_torque_penalty`**: negative reward proportional to the mean absolute joint torque — penalizes forceful, inefficient movement.
- **No twist/velocity command**: there is no `UniformVelocityCommand` in this task. The robot has no direction to walk; the only goal is to stand.

**Termination rules — critical:**

- **Terminate on success-hold**: end the episode (success) when the robot has maintained base height above a standing threshold (e.g. 0.75 m) and torso uprightness above a uprightness threshold for a continuous hold period (e.g. 2 seconds / 100 timesteps).
- **Terminate on timeout**: end the episode when the maximum episode length is reached (e.g. 500 timesteps).
- **Do NOT terminate on falling**: the robot starts every episode in a fallen pose. Falling is the initial state, not a failure condition. Any termination rule that triggers when the robot is low to the ground will kill every episode immediately at the start and prevent any learning.

**Initial state randomization:**

Spawn the robot at the start of each episode in a randomized fallen pose. This typically means:
- Randomizing the root (pelvis) orientation across a range of supine, prone, and lateral angles.
- Optionally randomizing joint angles to a "crumpled" configuration rather than a canonical lying pose.
- Keeping the root position close to the ground (low base height) so the robot is clearly on the floor.

### Step 2 — Pre-commit checks (inside `mjlab-dev`)

After writing the task file but before running training, run mjlab's pre-commit suite inside the container. **Do not use `make` or `uv run`** — the container uses pip-editable and does not have `uv`. Use the raw tools directly:

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && ruff format && ruff check --fix && pyright'"
```

Then run the unit tests for the new task:

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && pytest tests/'"
```

All checks must pass before proceeding to training. If `pyright` reports type errors in the new task class, fix them before the smoke-train — type errors in reward managers often signal wrong tensor shapes that produce NaN rewards at runtime.

### Step 3 — Smoke-train probe

Before committing to a multi-hour training run, verify the task registers and steps cleanly:

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Recovery-Flat-Unitree-G1 \
  --agent.max-iterations 200 \
  --agent.num-envs 1024 \
  --agent.seed 0'"
```

Kill the run as soon as it has printed ~50 iterations of reward values. Verify:
- No import errors or `AttributeError` on startup.
- The reward values are finite (not `nan` or `inf`).
- The episode lengths are greater than zero (episodes are not terminating immediately — which would mean the fallen-pose termination is mis-triggered on start).
- The mean episode reward climbs at all from zero (even a tiny increase proves the policy is receiving a gradient signal).

If episodes terminate immediately (episode length = 1), the most likely cause is a termination rule that fires on the initial fallen pose — review the termination conditions and confirm falling is not listed as a termination trigger.

### Step 4 — Full training run (iterA)

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Recovery-Flat-Unitree-G1 \
  --agent.max-iterations 5000 \
  --agent.num-envs 4096 \
  --agent.seed 42'"
```

After the run completes, verify the reward weights landed as designed:

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cat logs/rsl_rl/g1_recovery/<timestamp>/params/env.yaml'"
```

Replace `<timestamp>` with the actual run directory printed at startup. The YAML must confirm the reward terms and weights match the design; if terms are missing, they were not registered correctly and the run must be repeated after fixing the task definition.

### Step 5 — Record and review

```bash
# Side view — primary review angle:
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Recovery-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_recovery/<timestamp>/model_<best_iter>.pt \
  --no-shadows --no-reflections --no-debug-viz \
  --output /workspace/clips/s3_iterA_side.mp4'"

# Multi-camera for the final confirmed policy:
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Recovery-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_recovery/<timestamp>/model_<best_iter>.pt \
  --no-shadows --no-reflections --no-debug-viz \
  --cameras chase side front top grid \
  --output /workspace/clips/s3_final_{camera}.mp4'"
```

Pull clips to the Windows box:

```bash
scp spark:/workspace/clips/s3_*.mp4 docs/reports/assets/
```

Watch the clips frame by frame. Look for:
1. The robot starting flat on the ground (fallen pose).
2. Progressive movement toward upright.
3. The robot reaching and *holding* a stable stand — not just briefly touching upright before falling again.

If the robot appears to cheat (e.g. rocks in place without getting up, props on one elbow, spins to exploit a height artifact), document the hack and adjust the reward design for the next iteration.

---

## The experiment

**Method:** from-scratch task reward. There is no reference motion, no commanded velocity, no imitation target. The robot is placed in a random fallen pose and must discover standing through trial-and-error guided by the reward signal. This is the purest example of the "behavior is 100% the reward terms you wrote" paradigm.

**The reward design:**

| Term | What it measures | Effect |
|---|---|---|
| `base_height_rising` | Pelvis height above ground | Encourages the robot to lift itself off the floor |
| `torso_upright` | Alignment of robot "up" with world "up" | Penalizes horizontal body orientation |
| `stability_bonus` | Time held in a standing pose (height + uprightness both above threshold) | Rewards *maintaining* the stand, not just touching it once |
| `joint_velocity_penalty` | Mean absolute joint velocity | Discourages wild thrashing |
| `joint_torque_penalty` | Mean absolute joint torque | Discourages forceful, inefficient movement |

**Termination rules:** Success-hold (episode ends when the robot has stood stably for a continuous hold period) or timeout (episode ends at the maximum time limit). **Falling is explicitly not a termination condition** — every episode begins with the robot on the ground, so a falling-termination would kill every episode at step 1.

**No command:** the `UniformVelocityCommand` ("twist") present in the velocity task is removed entirely. There is nothing to track, follow, or approach — only the reward.

**Reward-hacking risk:** This task is the most prone to reward hacking of all four spine tasks. Common hacks to watch for and document:
- **Height without uprightness:** the robot finds a way to raise its pelvis (e.g. kicking legs up) without actually going upright, exploiting a high `base_height_rising` weight relative to `torso_upright`.
- **Elbow-prop:** the robot gets partially up by propping on one elbow and "convinces" itself this satisfies uprightness if the threshold is loose.
- **Jitter-stand:** the robot briefly satisfies the success-hold threshold by flailing to a near-standing pose for the minimum hold duration, then falling — collecting the stability bonus repeatedly.

Each iteration's hacks are documented and cross-linked to [S4 reward-hacking gallery](./2026-06-19-spine-reward-hacking-gallery.md), which is the canonical record of what hacks this program found and how they were closed.

---

## Artifacts & retrieval

| Artifact | Location on Spark | Committed to repo |
|---|---|---|
| New task source file | `/workspace/mjlab/src/mjlab/tasks/recovery_task.py` (or equivalent path in mjlab's task package) | NOT committed to this repo — lives only in the container's `mjlab/` tree on the Spark |
| Reference preview / smoke-train log | `/workspace/logs/rsl_rl/g1_recovery/<timestamp>/` | NOT committed — stays on Spark |
| Final multi-camera clips | `/workspace/clips/s3_final_*.mp4` | `docs/reports/assets/s3_final_*.mp4` |
| Iteration-log clips (hacks documented) | `/workspace/clips/s3_iterA_*.mp4`, `s3_iterB_*.mp4`, etc. | `docs/reports/assets/s3_iterA_side.mp4` (representative frames/clip only) |
| Final policy checkpoint | `logs/rsl_rl/g1_recovery/<timestamp>/model_<best_iter>.pt` | NOT committed — stays on Spark |
| W&B training run | `donb-chaotic-curiosity` entity | Link embedded in `getting-up.md` |

MP4 files are committed under `docs/reports/assets/` only — `.gitignore` contains the deliberate exception `!docs/reports/assets/*.mp4`. Do not commit MP4s anywhere else. Do not commit checkpoints, source files from mjlab's tree, or NPZ files.

The iteration log (which hacks were found, which reward adjustments were made, which iteration converged) is written as prose in `docs/reports/getting-up.md` after the final successful run, while context is fresh.

---

## Ops & safety

The Spark's unified memory is shared by co-tenant containers (`open-webui`, `compose-arangodb-1`, `ollama-compose`, `comfyui`). Under memory contention the system enters a swap-death-spiral and hard-reboots — not a clean CUDA out-of-memory error.

**Before each training run (on the Spark):**

```sh
docker stop open-webui compose-arangodb-1 ollama-compose
# Confirm headroom: the workload needs ~13 GiB; aim for ~110 GiB free
free -h
# swapoff -a and systemctl stop comfyui.service need an interactive sudo session
# if memory is tight, open an interactive session first: ssh spark then sudo swapoff -a
```

**During the run:**

Send SIGINT directly to the Python child — the `bash -c '... > log'` wrapper does not propagate SIGINT on the first try:

```sh
docker exec mjlab-dev bash -lc "kill -INT \$(pgrep -f mjlab.scripts.train)"
```

Never use a bare `pgrep -f` or `grep` of the train command in a counting or watcher loop — it matches its own command line and loops forever or miscounts. Use the bracket trick instead: `ps -eo cmd | grep "[p]ython -u -m mjlab.scripts.train"`.

**After the run (always, even on failure):**

```sh
docker start open-webui compose-arangodb-1 ollama-compose
```

**mjlab source ownership:** `mjlab/` on the Spark is owned by root (the container writes into the bind mount as root). **Do not `sudo chown` the tree** — this desyncs container file ownership. Use `docker cp` from `/tmp` or edit directly inside the container via `docker exec -it mjlab-dev bash`.

**Code iteration loop:** Expect multiple code-write → smoke-train → reward-tweak → full-train cycles before the behavior converges. Each iteration should be bracketed by the host-quiesce procedure above. Keep iteration logs updated after each run; do not wait until the final iteration to write them.

**Viser:** training prints a `http://localhost:8080` URL from inside the container; translate to `:8081` on the host (port 8080 is owned by open-webui when running, but open-webui is stopped during training).

---

## Success criteria

The following criteria must all be met before the S3 run is declared successful and `getting-up.md` is published.

1. **Reliable stand from randomized fallen poses.** In the recorded video, a clear majority of episodes show the robot starting flat on the ground and reaching a stable upright stand — not just briefly touching upright before falling back. "Stable" means the robot holds the standing pose for at least the success-hold duration defined in the task termination rule.

2. **Strategy is visible and legible.** The clip must show a recognizable get-up sequence — not just a jump-cut from lying to standing, and not a jitter that accidentally satisfies thresholds. A viewer with no ML background should be able to watch the clip and say "yes, it figured out how to stand up."

3. **Shaping journey documented honestly.** The iteration log in `getting-up.md` records every distinct hack the policy found, what reward adjustment closed it, and how many iterations were needed. If the final policy is iteration C, iterA and iterB are documented — not just the winner.

4. **Pre-commit checks passed.** The new task code passed `ruff format && ruff check --fix && pyright` and `pytest` inside `mjlab-dev` before training began.

5. **Terminations verified.** `params/env.yaml` from the final training run confirms that no termination condition triggers on low base height or falling (only on success-hold or timeout).

6. **Report written and published.** `docs/reports/getting-up.md` is populated with the final clips, the iteration log, the honest reward-hacking account (with links to S4), and a plain-language explanation, committed to `main` and visible on GitHub Pages.

---

## Open questions

The following items were targeted for resolution by the read-only Spark probe in Task 12 of the implementation plan.

**Probe run: 2026-06-19 — `ls /workspace/mjlab/src/mjlab/tasks && grep -ril "recover|getup|get_up|stand_up" /workspace/mjlab/src/mjlab || true`**

1. **mjlab's task package layout and registration API — RESOLVED.** The tasks package contains these top-level sub-packages: `velocity`, `tracking`, `cartpole`, `manipulation` (plus `registry.py` and `__init__.py`). The `__init__.py` uses `import_packages(__name__, blacklist=["utils", ".mdp"])` for auto-discovery — any new sub-package dropped into `src/mjlab/tasks/` is automatically imported.

   Task registration uses an **explicit `register_mjlab_task()` call** in a robot-specific config `__init__.py`. The pattern (from `tasks/velocity/config/g1/__init__.py`) is:
   ```python
   from mjlab.tasks.registry import register_mjlab_task
   from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
   from .env_cfgs import unitree_g1_flat_env_cfg
   from .rl_cfg import unitree_g1_ppo_runner_cfg

   register_mjlab_task(
     task_id="Mjlab-Recovery-Flat-Unitree-G1",
     env_cfg=unitree_g1_flat_env_cfg(),       # replace with recovery cfg
     play_env_cfg=unitree_g1_flat_env_cfg(play=True),
     rl_cfg=unitree_g1_ppo_runner_cfg(),
     runner_cls=VelocityOnPolicyRunner,        # or None for default
   )
   ```
   The new recovery task must be placed under `src/mjlab/tasks/recovery/` (or similar), with a `config/g1/__init__.py` that calls `register_mjlab_task`. Auto-discovery via `import_packages` will pick it up — no changes to the top-level `tasks/__init__.py` needed.

2. **Whether an upstream recovery task already exists to fork — RESOLVED.** The grep for `recover`, `getup`, `get_up`, `stand_up` across `/workspace/mjlab/src/mjlab/` returned **only one match**: `/workspace/mjlab/src/mjlab/entity/data.py` (which contains neither a task class nor reward terms — it matched on an unrelated field name). **No recovery or get-up task exists in the current mjlab codebase.** The S3 task must be written from scratch without an upstream template to fork. The velocity task's `config/g1/__init__.py` and `velocity_env_cfg.py` are the closest structural reference.

3. **Cleanest way to spawn randomized fallen poses — partially resolved.** The velocity task uses a config-driven `init_state` in the `entities.robot` block (see the `env.yaml` baseline): root `pos`, `rot` (quaternion), `lin_vel`, `ang_vel`, and per-joint `joint_pos` patterns are all set declaratively. The `joint_pos` block uses regex patterns (e.g. `".*_hip_pitch_joint": -0.312`). For fallen-pose randomization, the most likely approach is overriding the root quaternion at episode reset via mjlab's event/randomization system (analogous to the terrain init randomization pattern). The exact API (`EventManager`, `reset_robot_state`, or direct `data.qpos` mutation) requires reading the velocity task's event/randomization config — not done in this probe. **⚠ still unresolved:** inspect `tasks/velocity/velocity_env_cfg.py` or an event config to find the reset/randomization hook used for initial state.

4. **Whether the smoke-train probe requires a registry edit — RESOLVED.** mjlab uses `import_packages` auto-discovery, not a hand-maintained list. A new task placed in a new sub-package under `src/mjlab/tasks/` with a `register_mjlab_task()` call in its `__init__.py` is discovered automatically at import time — no changes to any existing file are needed beyond the new task files themselves.
