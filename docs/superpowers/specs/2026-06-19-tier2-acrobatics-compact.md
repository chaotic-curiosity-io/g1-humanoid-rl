# Compact Specs: Tier-2 Deferred Acrobatics

**Date:** 2026-06-19
**Branch:** `g1-skills-curriculum`
**Status:** Deferred — ready to run when prioritised
**Track:** Imitation (Tier-2 long tail)
**Linked from:** `docs/reports/README.md` (syllabus "Ready-to-run" section)

---

## What "compact spec" means

A compact spec gives you everything needed to promote a task to a full training run
later — without redesigning from scratch. Each section is deliberately terse: goal,
what to change in the environment, whether new code is required, the exact commands
to train and record, a cost estimate, and the criteria for calling it done.

All three tasks below use `Mjlab-Tracking-Flat-Unitree-G1` — the same tracking
environment as the cartwheel and S2 backflip spine task. The retargeting + NPZ
pipeline (`.pkl` → CSV → `motion.npz` → training flag) already exists on the Spark;
no new infrastructure is needed. The central lesson from the cartwheel carries
across all three: **visual frame-by-frame review is the success gate**, not a
numerical score.

None of these tasks has been trained yet. They are ready-to-run, not yet run.

---

## 1. Spinkick / Martial-Arts Kick

**Goal:** Train the G1 to perform a single-leg spinning kick — a fast rotational
strike where the robot pivots on one foot and swings the other leg in a wide arc.
This is a natural follow-on to the cartwheel: same aerial-phase difficulty, shorter
duration, and the reference is already in the pipeline.

**Env + reward terms to touch:** `Mjlab-Tracking-Flat-Unitree-G1` with
`anchor_pos` and `ee_body_pos` termination thresholds at 0.5 m (same reasoning as
the cartwheel — the kicking leg sweeps a large arc; the default 0.25 m threshold
will terminate mid-kick). The reference motion ships with the `g1_spinkick_example`
asset already present in the pipeline (confirm the exact file path at run time:
look for a spinkick `.pkl` or `.npz` in `pose-pipeline/` or the MimicKit asset
directory). No reward weight changes needed beyond the thresholds; the tracking
reward already incentivises following the reference each frame.

**New-code flag:** None — the retargeter and pipeline exist. If the spinkick
reference is already in `.npz` form (check `pose-pipeline/motions/`), skip the
retarget step. If only a `.pkl` exists, run `csv_to_npz` as below.

**Train/record commands:**

```bash
# Step 1 — retarget if needed (skip if .npz already present)
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && \
  python -m mjlab.scripts.csv_to_npz \
  /workspace/pose-pipeline/g1_spinkick_example/spinkick.pkl \
  /workspace/pose-pipeline/motions/spinkick.npz'"

# Step 2 — inspect the reference before training
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/play_motion_npz.py \
  --motion /workspace/pose-pipeline/motions/spinkick.npz \
  --output /workspace/clips/tier2_spinkick_reference.mp4'"

# Step 3 — train
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file /workspace/pose-pipeline/motions/spinkick.npz \
  --env.terminations.anchor-pos.threshold 0.5 \
  --env.terminations.ee-body-pos.threshold 0.5 \
  --agent.num-envs 4096 \
  --agent.max-iterations 10000 \
  --agent.seed 42'"

# Step 4 — record (threshold must match or relax training; never tighter)
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Tracking-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_tracking/<timestamp>/model_<best_iter>.pt \
  --termination-threshold 0.5 \
  --no-shadows --no-reflections --no-debug-viz \
  --cameras chase side front \
  --output /workspace/clips/tier2_spinkick_{camera}.mp4'"
```

Exact termination flag paths (`--env.terminations.anchor-pos.threshold` etc.) to
confirm at run time against the cartwheel run's `params/env.yaml`. The `g1_spinkick_example`
asset path and file name to confirm at run time.

**Cost estimate:** ~5–15 minutes to retarget (CPU) + ~4–8 hours GPU training.

**Success criteria:** Side-view clip confirmed frame by frame shows: the robot
plants on one foot, the kicking leg sweeps through a full arc (at least 90° of
rotation visible in the air), and the robot recovers upright. Visual review is the
gate — do not rely on a roll-angle scorer alone.

---

## 2. Jump

**Goal:** Train the G1 to perform a two-footed vertical jump — both feet leaving
the ground simultaneously, the body rising to a peak height meaningfully above
standing, then landing upright. Unlike the spinkick and dance, jump *can be trained
as motion tracking OR as a from-scratch task reward*; the two approaches are
documented below and the choice can be made at promotion time.

**Env + reward terms to touch:**

*Option A — Motion tracking (recommended first try):* `Mjlab-Tracking-Flat-Unitree-G1`
with a retargeted jump reference (`jump.npz`). Source the reference from SMPL mocap
or a MimicKit jump asset (confirm availability at run time). Set `anchor_pos` and
`ee_body_pos` thresholds to 0.5 m — both feet leave the ground simultaneously, so
the default 0.25 m will terminate at the apex. Follow the same retarget → inspect →
train → record loop as the spinkick.

*Option B — Task reward (no reference needed):* `Mjlab-Velocity-Flat-Unitree-G1`
(or a lightweight clone). Reward terms to add: a large bonus on `base_height` when
both feet have zero contact simultaneously (the air phase), an uprightness term so
the robot doesn't tilt mid-air, and a landing-stability term (low velocity and
upright on first foot re-contact). No velocity command needed — zero the `lin_vel_x`
range and all curriculum stages. This is a from-scratch approach with light new code
(one new reward term for the simultaneous-air-phase detection).

**New-code flag:** Option A — none. Option B — light new code: one reward term for
simultaneous foot-off-ground detection; implement via `docker cp` / in-container
edit, then `ruff format && ruff check --fix && pyright` before training.

**Train/record commands (Option A — tracking):**

```bash
# Retarget jump reference
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && \
  python -m mjlab.scripts.csv_to_npz \
  /workspace/pose-pipeline/<jump_source>.pkl \
  /workspace/pose-pipeline/motions/jump.npz'"

# Inspect before training
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/play_motion_npz.py \
  --motion /workspace/pose-pipeline/motions/jump.npz \
  --output /workspace/clips/tier2_jump_reference.mp4'"

# Train (tracking)
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file /workspace/pose-pipeline/motions/jump.npz \
  --env.terminations.anchor-pos.threshold 0.5 \
  --env.terminations.ee-body-pos.threshold 0.5 \
  --agent.num-envs 4096 \
  --agent.max-iterations 10000 \
  --agent.seed 42'"

# Record
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Tracking-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_tracking/<timestamp>/model_<best_iter>.pt \
  --disable-terminations \
  --no-shadows --no-reflections --no-debug-viz \
  --cameras chase side \
  --output /workspace/clips/tier2_jump_{camera}.mp4'"
```

**Train/record commands (Option B — task reward):**

```bash
# Train (task reward, after new term is in container)
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Unitree-G1 \
  \"--env.commands.twist.ranges.lin-vel-x=(0.0, 0.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.0.lin-vel-x=(0.0, 0.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.1.lin-vel-x=(0.0, 0.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.2.lin-vel-x=(0.0, 0.0)\" \
  --env.rewards.air_phase_bonus.weight 5.0 \
  --env.rewards.base_height.weight 2.0 \
  --agent.num-envs 4096 \
  --agent.max-iterations 5000'"

# Record
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_velocity/<timestamp>/model_<best_iter>.pt \
  --no-shadows --no-reflections --no-debug-viz \
  --cameras chase side \
  --output /workspace/clips/tier2_jump_reward_{camera}.mp4'"
```

Jump reference source (for Option A) to confirm at run time — check MimicKit's
shipped G1 asset set for a jump or vertical-leap pkl; if absent, source from SMPL
mocap. Option B reward term flag paths to confirm at run time.

**Cost estimate:** Option A — ~5–15 minutes retarget + ~4–8 hours GPU. Option B —
~1–2 hours new-code step + ~2–4 hours GPU.

**Success criteria:** Side-view clip confirmed frame by frame shows: both feet leave
the ground at the same time, the robot's center of mass rises visibly above standing
height, and the robot lands upright without falling. The clip is the gate — if the
robot hops with one foot or stumbles on landing, it does not pass.

---

## 3. Dance

**Goal:** Train the G1 to perform a short repeating dance sequence — a rhythmic
series of arm and leg movements taken from human mocap. This is the most
expressive use of the tracking pipeline and the hardest reference to source well:
a natural-looking human dance involves fast limb movements that stress the
retargeter and tight tracking tolerances that may need loosening further.

**Env + reward terms to touch:** `Mjlab-Tracking-Flat-Unitree-G1`. The tracking
reward (`anchor_pos`, `ee_body_pos`) drives the policy toward the reference as
usual. Dance motions involve wide arm sweeps and weight shifts that may need
thresholds raised beyond 0.5 m — start at 0.5 m and increase to 0.75 m if the
policy is repeatedly terminated before the sequence completes a full cycle. The
reference must come from an SMPL mocap source (e.g. AMASS or a Creative Commons
dance capture); it does not ship with MimicKit's G1 set. Retarget via
`smpl_backflip_to_g1.py` (the same SMPL→G1 retargeter used for the backflip);
confirm the retargeter handles the full SMPL mocap format of the chosen source.

**New-code flag:** None in the training pipeline. Reference sourcing is the work:
find a short (~3–6 s), Creative Commons SMPL mocap dance clip, download it, and run
it through the retarget → `csv_to_npz` pipeline. The retargeter already handles
SMPL body-model data; test its output with `play_motion_npz.py` before training.

**Train/record commands:**

```bash
# Step 1 — retarget the dance reference (source path to fill in at run time)
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && \
  python scripts/smpl_backflip_to_g1.py \
  --input /workspace/pose-pipeline/dance_source.pkl \
  --output /workspace/pose-pipeline/outputs/gmr_pkl/dance_g1.pkl'"

ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && \
  python -m mjlab.scripts.csv_to_npz \
  /workspace/pose-pipeline/outputs/gmr_pkl/dance_g1.pkl \
  /workspace/pose-pipeline/motions/dance.npz'"

# Step 2 — inspect reference
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/play_motion_npz.py \
  --motion /workspace/pose-pipeline/motions/dance.npz \
  --output /workspace/clips/tier2_dance_reference.mp4'"

# Step 3 — train
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file /workspace/pose-pipeline/motions/dance.npz \
  --env.terminations.anchor-pos.threshold 0.5 \
  --env.terminations.ee-body-pos.threshold 0.5 \
  --agent.num-envs 4096 \
  --agent.max-iterations 15000 \
  --agent.seed 42'"

# Step 4 — record (disable terminations to see full sequence loop)
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Tracking-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_tracking/<timestamp>/model_<best_iter>.pt \
  --disable-terminations \
  --no-shadows --no-reflections --no-debug-viz \
  --cameras chase side front \
  --output /workspace/clips/tier2_dance_{camera}.mp4'"
```

`smpl_backflip_to_g1.py` CLI flags (`--input`, `--output`) to confirm at run time
against `python scripts/smpl_backflip_to_g1.py --help`. Threshold may need raising
to 0.75 m if dance limb sweeps are wider than a backflip's; start at 0.5 m and
iterate. Dance reference must be sourced before this task can be promoted; AMASS
and similar SMPL mocap databases are the recommended starting point.

**Cost estimate:** ~30–60 minutes reference sourcing and retargeting (CPU) + ~6–12
hours GPU training (dance sequences are longer and more varied than a single-skill
motion).

**Success criteria:** A 20–40 second clip confirmed frame by frame shows the robot
performing at least two complete cycles of the reference dance sequence — arm and
leg movements visibly matching the intended choreography, the robot remaining upright
throughout. "It did something rhythmic" is not sufficient; the movements must
correspond to the reference. If the retarget degrades badly (robot is rigid or
collapses), the reference must be re-sourced or the retargeter tuned before training.
