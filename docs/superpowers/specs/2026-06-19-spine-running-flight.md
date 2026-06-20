# Spec: S1 — Running / Flight Phase

**Date:** 2026-06-19
**Branch:** `g1-skills-curriculum`
**Status:** Ready for training campaign
**Track:** Locomotion (spine task 1 of 4)
**Feeds into:** `docs/reports/running-and-flight.md`; linked from the syllabus `docs/reports/README.md`

---

## Goal

Train a Unitree G1 humanoid to run — not just walk — fast enough that both feet leave the ground at the same time. That brief moment when neither foot touches the floor is called a **flight phase**, and it is the physical signature that separates running from walking. The existing baseline walker (`model_2050.pt`, trained in Session 1) never achieves a flight phase: its step pattern at any commanded speed stays planted enough that at least one foot is always on the ground.

This experiment asks a simple question: if we tell the robot to practice fast speeds and reward it for getting its feet off the ground, does a true running gait emerge? The answer is visible — you can see it in a clip — and requires no special measurement beyond watching the feet.

The experiment is an extension of report [03 — Turning the Knobs](../../reports/03-turning-the-knobs.md), which showed that one changed knob (commanded speed range) produces a robot that looks better on paper but performs worse outside its training range. Here we go in the opposite direction: widen the commanded speed far beyond the default, add a reward signal that positively values airtime, and see whether the gait qualitatively changes.

---

## Audience & voice

This report is written for someone with no background in robotics or machine learning. Every term is defined on first use. "Policy" means the neural network that maps sensor readings to motor commands — the robot's brain. "Reward" is a number the training algorithm uses as feedback: higher reward means the robot is doing something we wanted. "Episode" is one uninterrupted simulation run — the robot starts standing, does things, and eventually the simulation resets (either because it fell, hit a time limit, or finished a trajectory). "Flight phase" is explained in plain language: both feet off the ground simultaneously. Concrete analogies are preferred over mathematical notation. Each report is standalone: a reader who picks it up without reading 00–03 can follow the reasoning.

---

## Staged-arc table

| Stage | What runs on the Spark | Estimated cost | Output |
|---|---|---|---|
| **Replay control** | Load existing `model_2050.pt` via `mjlab.scripts.play`; record a short side-view clip at 1.0 m/s commanded speed | Seconds | Baseline clip showing the walking gait — no flight phase |
| **Train fast runner** | Fresh `Mjlab-Velocity-Flat-Unitree-G1` from scratch with high `lin_vel_x` range + three curriculum-stage overrides + bumped `feet_air_time` weight | ~1–1.5 h GPU | New checkpoint (`model_<N>.pt`); W&B run; `params/env.yaml` confirming the overrides landed |
| **Record A/B clips** | `record_policy.py` on both policies (control `model_2050.pt` and the fast runner's best checkpoint), same commanded speed (e.g. 1.5 m/s), side and chase cameras | Minutes; CPU-runnable | Two short MP4s for the A/B comparison in the report |
| **Cadence plot** | `plot_training_curves.py` or `record_policy.py` telemetry → cadence-vs-commanded-speed PNG | Minutes | One PNG embedded in `running-and-flight.md` |

The two recording steps can run while the GPU is idle or even while a subsequent training job occupies the GPU (the offscreen renderer is CPU-runnable).

---

## Concrete runs

All commands are issued from the Windows host and execute inside `mjlab-dev` on the Spark.

### Step 1 — Replay the control walker

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_velocity/2026-04-17_18-46-23/model_2050.pt \
  --no-shadows --no-reflections --no-debug-viz \
  --output /workspace/clips/s1_control_walk.mp4'"
```

### Step 2 — Train the fast runner (B)

The key insight documented in CLAUDE.md and demonstrated in report 03: the Flat task's `command_vel` curriculum is active even though "flat = no curriculum" sounds intuitive. Its stage-0 fires at every episode boundary and silently resets the `lin_vel_x` range back to `(-1.0, 1.0)`, clobbering any plain range override. To lock the range you must override the command **and all three curriculum stages**. Tuples require the `=` syntax (no space between flag name and value).

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Unitree-G1 \
  \"--env.commands.twist.ranges.lin-vel-x=(0.0, 3.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.0.lin-vel-x=(0.0, 3.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.1.lin-vel-x=(0.0, 3.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.2.lin-vel-x=(0.0, 3.0)\" \
  --env.rewards.feet_air_time.weight 2.0 \
  --agent.max-iterations 2000 \
  --agent.seed 42'"
```

After the run completes, verify the overrides actually landed:

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cat logs/rsl_rl/g1_velocity/<timestamp>/params/env.yaml | grep -A5 lin.vel.x'"
```

Replace `<timestamp>` with the actual run directory (printed by the trainer at startup). The YAML must show `(0.0, 3.0)` for the command range and all three curriculum stages; if it still shows `(-1.0, 1.0)`, the override did not land and the run must be repeated with the corrected flag syntax.

### Step 3 — Record A/B clips

Record both policies at the same commanded forward speed (e.g. 1.5 m/s) so the comparison is fair:

```bash
# Control walker (A)
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_velocity/2026-04-17_18-46-23/model_2050.pt \
  --no-shadows --no-reflections --no-debug-viz \
  --output /workspace/clips/s1_A_walker_1p5ms.mp4'"

# Fast runner (B) — replace <timestamp> and <best_iter> with the actual values
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_velocity/<timestamp>/model_<best_iter>.pt \
  --no-shadows --no-reflections --no-debug-viz \
  --output /workspace/clips/s1_B_runner_1p5ms.mp4'"
```

### Step 4 — Cadence plot

Use `plot_training_curves.py` with the telemetry dump from `record_policy.py`. The plot should show step cadence (steps per second) or air-time fraction versus commanded forward speed, overlaid for the two policies:

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && python scripts/plot_training_curves.py \
  --run control=logs/rsl_rl/g1_velocity/2026-04-17_18-46-23 \
  --run fast=logs/rsl_rl/g1_velocity/<timestamp> \
  --output /workspace/plots/s1_cadence_vs_speed.png'"
```

Pull clips and plots to the Windows box for inspection and commit:

```bash
scp spark:/workspace/clips/s1_*.mp4 docs/reports/assets/
scp spark:/workspace/plots/s1_cadence_vs_speed.png docs/reports/assets/
```

---

## The experiment

**Control (A):** The existing `model_2050.pt` walker from Session 1 (`logs/rsl_rl/g1_velocity/2026-04-17_18-46-23/`). Trained with the default commanded speed range `(-1.0, 1.0)` m/s. Reward at convergence: 50.5. Episode length: 995/1000 steps. This policy is already on disk — no training needed.

**Treatment (B):** A fresh policy trained with:
- Commanded `lin_vel_x` range raised to `(0.0, 3.0)` m/s, pinned via all three curriculum-stage overrides.
- `feet_air_time` reward weight bumped (the weight value to use is an open question — see below; start with `2.0` and tune if the flight phase does not emerge).
- All other hyperparameters identical: same task (`Mjlab-Velocity-Flat-Unitree-G1`), same PPO, same network, same seed 42.

**One changed family of knobs:** the commanded speed distribution and the air-time incentive. Everything else is held constant. This is the same "change exactly one thing" discipline that made report 03's result legible — here we extend it in the other direction (faster, not slower).

**What to look for in the clips:** At a commanded speed where the control clearly walks (say, 1.0 m/s), both should look similar. At a higher commanded speed (1.5–2.0 m/s), the fast runner should show visible flight phases — moments where the camera catches both feet off the ground. The control will either refuse to go that fast or maintain a planted, shuffling gait.

**Visual-verification gate:** The flight phase is determined by watching the clips, not by a score. Look for frames where the robot's entire body is visibly airborne — neither foot touching the ground plane. This is the primary success signal.

---

## Artifacts & retrieval

| Artifact | Location on Spark | Committed to repo |
|---|---|---|
| Control clip (A) | `/workspace/clips/s1_A_walker_1p5ms.mp4` | `docs/reports/assets/s1_A_walker_1p5ms.mp4` |
| Fast-runner clip (B) | `/workspace/clips/s1_B_runner_1p5ms.mp4` | `docs/reports/assets/s1_B_runner_1p5ms.mp4` |
| Cadence-vs-speed PNG | `/workspace/plots/s1_cadence_vs_speed.png` | `docs/reports/assets/s1_cadence_vs_speed.png` |
| Fast-runner checkpoint (best) | `logs/rsl_rl/g1_velocity/<timestamp>/model_<N>.pt` | NOT committed — stays on Spark |
| W&B run | `donb-chaotic-curiosity` entity | Link embedded in `running-and-flight.md` |

MP4 files are committed under `docs/reports/assets/` because `.gitignore` contains the deliberate exception `!docs/reports/assets/*.mp4`. Do not commit MP4s anywhere else.

The cadence plot is rendered as a PNG via `plot_training_curves.py` (which reads RSL-RL tensorboard scalars and writes curve PNGs). The approach was established in the walking-arc reports and reused here without change.

---

## Ops & safety

The Spark's unified memory is shared by co-tenant containers. Under memory contention the system enters a swap-death-spiral and hard-reboots — not a clean CUDA out-of-memory error. The following bracket must be run before and after every training run.

**Before the training run (on the Spark):**

```sh
docker stop open-webui compose-arangodb-1 ollama-compose
# Confirm headroom: the workload needs ~13 GiB; aim for ~110 GiB free
free -h
# swapoff -a and systemctl stop comfyui.service need an interactive sudo session
# if memory is tight, open an interactive session first: ssh spark then sudo swapoff -a
```

**During the run:**

Training is launched as a single foreground command (or backgrounded explicitly). To stop it cleanly, send SIGINT directly to the Python child — the `bash -c '... > log'` wrapper does not propagate SIGINT on the first try:

```sh
docker exec mjlab-dev bash -lc "kill -INT \$(pgrep -f mjlab.scripts.train)"
```

Never use a bare `pgrep -f` or `grep` of the train command in a counting or watcher loop — it matches its own command line and loops forever or miscounts. Use the bracket trick instead: `ps -eo cmd | grep "[p]ython -u -m mjlab.scripts.train"`.

**After the run (always, even on failure):**

```sh
docker start open-webui compose-arangodb-1 ollama-compose
```

**Duration:** ~1–1.5 h GPU for 2,000 iterations at 2,048 envs. Recording steps are CPU-runnable and can overlap with a subsequent job if needed.

---

## Success criteria

1. **Visible flight phase:** at least one clip clearly shows both feet off the ground simultaneously in the fast-runner policy (B). This is verified by watching the clip frame by frame — not by a score.
2. **Measurably different cadence or step length:** the cadence-vs-speed plot shows a quantitatively different relationship between the two policies at the same commanded speed. Step length or air-time fraction is a suitable metric.
3. **Attributable to the changed knobs:** the training config (`params/env.yaml`) confirms the high-speed range and elevated air-time weight landed correctly. No other reward weights or hyperparameters differ between A and B.
4. **Report is written and published:** `docs/reports/running-and-flight.md` is populated with the clips, plot, and plain-language explanation, committed to `main`, and visible on GitHub Pages.

---

## Open questions

The following items were targeted for resolution by the read-only Spark probe in Task 12 of the implementation plan.

**Probe run: 2026-06-19 — `sed -n 1071,1530p /workspace/mjlab/logs/rsl_rl/g1_velocity/2026-04-17_18-46-23/params/env.yaml`**

1. **Exact air-time reward term name — RESOLVED.** The YAML key in the Flat task's reward config is **`air_time`** (not `feet_air_time`). The underlying function is `mjlab.tasks.velocity.mdp.rewards.feet_air_time`, but the CLI flag to override its weight is `--env.rewards.air_time.weight`. In the baseline run this term has `weight: 0.0` — it is registered but inactive. The training command in the Concrete runs section must use `--env.rewards.air_time.weight 2.0` (not `feet_air_time`).

   Full reward term names confirmed in the baseline `env.yaml` (for reference across S1 and S4):
   `track_linear_velocity`, `track_angular_velocity`, `upright`, `pose`, `body_ang_vel`, `angular_momentum`, `dof_pos_limits`, `action_rate_l2`, `air_time`, `foot_clearance`, `foot_swing_height`, `foot_slip`, `soft_landing`, `self_collisions`.
   There is **no `base_height` term** and **no `termination` term** in the velocity task reward config.

2. **Three curriculum-stage override keys — RESOLVED.** The YAML structure confirmed:
   ```yaml
   curriculum:
     command_vel:
       func: mjlab.tasks.velocity.mdp.curriculums.commands_vel
       params:
         command_name: twist
         velocity_stages:
         - step: 0
           lin_vel_x: (-1.0, 1.0)
         - step: 120000
           lin_vel_x: (-1.5, 2.0)
         - step: 240000
           lin_vel_x: (-2.0, 3.0)
   ```
   The three CLI override keys are (using hyphen-to-underscore translation that the mjlab CLI applies):
   `--env.curriculum.command-vel.params.velocity-stages.0.lin-vel-x`
   `--env.curriculum.command-vel.params.velocity-stages.1.lin-vel-x`
   `--env.curriculum.command-vel.params.velocity-stages.2.lin-vel-x`
   These match the CLAUDE.md snippet exactly.

3. **Whether high speed needs episode-length or terrain adjustment — RESOLVED.** The baseline run's `episode_length_s: 20.0` at a decimation of 4 (50 Hz control) equals 1000 steps. At 3.0 m/s for 20 seconds the robot travels 60 m; the flat arena uses `env_spacing: 2.0` m per env with no explicit wall bounds visible in the YAML. This is a standard periodic/infinite flat plane, not a bounded arena — the robot can run indefinitely without hitting a wall. No episode-length or terrain adjustment is required for S1.
