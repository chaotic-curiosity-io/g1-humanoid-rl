# Compact Specs: Tier-1 Deferred Gait Tweaks

**Date:** 2026-06-19
**Branch:** `g1-skills-curriculum`
**Status:** Deferred — ready to run when prioritised
**Track:** Locomotion (Tier-1 long tail)
**Linked from:** `docs/reports/README.md` (syllabus "Ready-to-run" section)

---

## What "compact spec" means

A compact spec gives you everything needed to promote a task to a full training run
later — without redesigning from scratch. Each section below is deliberately terse:
goal, what to change in the environment, whether new code is required, the exact
commands to train and record, a cost estimate, and the criteria for calling it done.

These six tasks all live on the existing `Mjlab-Velocity-Flat-Unitree-G1` environment
and share the same gotcha documented in CLAUDE.md: the Flat task's `command_vel`
curriculum is active and silently resets the `lin_vel_x` range at every episode
boundary. Any range override must also override all three curriculum stages using the
`=` syntax. Commands below follow the pattern established in the S1 spine spec.

None of these tasks has been trained yet. They are ready-to-run, not yet run.

---

## 1. Crouched "Groucho" walk

**Goal:** Train the G1 to walk at a significantly reduced torso height — a
continuously crouched gait, nicknamed "Groucho" after the exaggerated low-walk
popularised by the comedian. The point is to show that reward terms alone, without
any special controller, can lock the robot into an unusual but stable posture.

**Env + reward terms to touch:** `Mjlab-Velocity-Flat-Unitree-G1` (no task change).
Add a strong negative penalty on `base_height` when it exceeds a low target
(e.g. 0.55 m instead of the default ~0.8 m) — or equivalently, reward `base_height`
with a tight Gaussian centred on 0.55 m. Keep all velocity-tracking rewards. The
exact reward term name to confirm at run time: look for `base_height` in
`params/env.yaml` from the baseline run.

**New-code flag:** None. The `base_height` term already exists; changing its target
and weight is a CLI override.

**Train/record commands:**

```bash
# Train — crouched walk
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Unitree-G1 \
  \"--env.commands.twist.ranges.lin-vel-x=(0.0, 1.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.0.lin-vel-x=(0.0, 1.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.1.lin-vel-x=(0.0, 1.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.2.lin-vel-x=(0.0, 1.0)\" \
  --env.rewards.base_height.weight 5.0 \
  --env.rewards.base_height.params.target_height 0.55 \
  --agent.max-iterations 2000'"

# Record — side view, fixed forward command
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_velocity/<timestamp>/model_<best_iter>.pt \
  --no-shadows --no-reflections --no-debug-viz \
  --output /workspace/clips/tier1_groucho.mp4'"
```

Confirm the target height landed: `cat logs/rsl_rl/g1_velocity/<timestamp>/params/env.yaml | grep -A3 base_height`.
Exact reward-term flag path (`--env.rewards.base_height.*`) to confirm at run time.

**Cost estimate:** ~1 h GPU (2,000 iterations, same env as the baseline).

**Success criteria:** Side-view clip shows the robot walking at a torso height
visibly lower than the baseline (~0.55 m vs ~0.8 m); the posture is sustained, not
intermittent; the policy still tracks the commanded forward velocity.

---

## 2. Tiptoe walk

**Goal:** Train the G1 to walk with its feet in a plantarflexed (toes-down,
heel-up) posture — a tiptoe gait. This demonstrates reward-driven control of
foot contact geometry, a detail that standard velocity-tracking rewards ignore.

**Env + reward terms to touch:** `Mjlab-Velocity-Flat-Unitree-G1`. Add a reward
(or remove a penalty) that incentivises high ankle flexion or rewards contact only
at the forefoot. The exact mechanism depends on which foot-contact or joint-angle
terms are available in the reward manager — check `params/env.yaml` for terms like
`foot_contact`, `ankle_dof`, or `joint_pos`; the approach to confirm at run time.
Alternatively, penalise heel-contact body IDs if the simulator exposes them via a
contact reward term.

**New-code flag:** Light — may require a small new reward term (a joint-angle target
on ankle joints) if no existing term covers ankle posture. Implement as a `joint_pos`
target override if that term exists; otherwise add a one-function reward term inside
the container following the `docker cp` or in-container-edit path from CLAUDE.md.

**Train/record commands:**

```bash
# Train — tiptoe (ankle joint target override; flag path to confirm at run time)
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Unitree-G1 \
  \"--env.commands.twist.ranges.lin-vel-x=(0.0, 1.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.0.lin-vel-x=(0.0, 1.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.1.lin-vel-x=(0.0, 1.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.2.lin-vel-x=(0.0, 1.0)\" \
  --env.rewards.joint_pos.weight 3.0 \
  --agent.max-iterations 2000'"

# Record — close side/front view to see ankle posture
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_velocity/<timestamp>/model_<best_iter>.pt \
  --no-shadows --no-reflections --no-debug-viz \
  --output /workspace/clips/tier1_tiptoe.mp4'"
```

**Cost estimate:** ~1 h GPU, plus up to 30 min for the light new-code step if a
joint-angle reward term must be added.

**Success criteria:** Close-up clip shows a clearly elevated heel with weight on the
forefoot during the stance phase; the gait remains stable and forward-progressing;
visually distinguishable from the flat-footed baseline.

---

## 3. Energy-efficiency / cost of transport

**Goal:** Train a version of the G1 walker that prioritises minimising energy use
(measured as electrical power or joint torques times joint velocities) for a given
forward speed — the robotics metric called "cost of transport." The resulting policy
should look smoother and less jerky than the default walker, which optimises reward
without a strong efficiency incentive.

**Env + reward terms to touch:** `Mjlab-Velocity-Flat-Unitree-G1`. The existing
reward set already contains terms like `joint_torques` and `action_rate` that act as
efficiency proxies. Increase their weights significantly relative to the velocity-
tracking reward, or add a dedicated power-consumption penalty
(`joint_torques * joint_velocities`). Exact term names to confirm at run time from
`params/env.yaml`: look for `torques`, `action_rate`, `power`, or `energy`.

**New-code flag:** None if `joint_torques` and `action_rate` terms already exist
with tunable weights (very likely — they are standard RSL-RL humanoid terms). If a
true power term (torques times velocities) is absent, it is a one-function addition.

**Train/record commands:**

```bash
# Train — high efficiency penalty, moderate speed range
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Unitree-G1 \
  \"--env.commands.twist.ranges.lin-vel-x=(0.0, 1.5)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.0.lin-vel-x=(0.0, 1.5)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.1.lin-vel-x=(0.0, 1.5)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.2.lin-vel-x=(0.0, 1.5)\" \
  --env.rewards.joint_torques.weight -0.5 \
  --env.rewards.action_rate.weight -0.5 \
  --agent.max-iterations 2000'"

# Record — side view; also dump telemetry to compare torque profiles
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_velocity/<timestamp>/model_<best_iter>.pt \
  --no-shadows --no-reflections --no-debug-viz \
  --output /workspace/clips/tier1_efficiency.mp4'"
```

The telemetry dump from `record_policy.py` can be used with `plot_training_curves.py`
to produce a torque-vs-speed or power-vs-speed comparison against the baseline.

**Cost estimate:** ~1 h GPU.

**Success criteria:** Clip shows a visibly smoother gait with less arm flailing and
jerky motion than the baseline; telemetry shows lower mean joint torques or action
rate at the same commanded speed; policy still tracks the commanded velocity.

---

## 4. Spin-in-place

**Goal:** Train the G1 to spin on its vertical axis at a commanded angular rate
while keeping its base position nearly fixed — a pure yaw rotation, not a turning
walk. Distinct from the existing walker's turn behaviour, which combines lateral
movement with heading change.

**Env + reward terms to touch:** `Mjlab-Velocity-Flat-Unitree-G1`. Fix `lin_vel_x`
and `lin_vel_y` commands at zero and command a high `ang_vel_z` (yaw rate), e.g.
2–4 rad/s. To prevent the robot drifting while spinning, add a base-position
penalty or reward low lateral velocity. Override the curriculum stages for `ang_vel_z`
and zero the linear-velocity curriculum stages. Exact curriculum key for `ang_vel_z`
to confirm at run time (likely `--env.curriculum.command-vel.params.velocity-stages.N.ang-vel-z`).

**New-code flag:** None. The angular velocity command and reward already exist in the
velocity task.

**Train/record commands:**

```bash
# Train — spin in place: zero linear, high angular
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Unitree-G1 \
  \"--env.commands.twist.ranges.lin-vel-x=(0.0, 0.0)\" \
  \"--env.commands.twist.ranges.lin-vel-y=(0.0, 0.0)\" \
  \"--env.commands.twist.ranges.ang-vel-z=(-4.0, 4.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.0.lin-vel-x=(0.0, 0.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.1.lin-vel-x=(0.0, 0.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.2.lin-vel-x=(0.0, 0.0)\" \
  --agent.max-iterations 2000'"

# Record — top-down or angled view to show rotation clearly
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_velocity/<timestamp>/model_<best_iter>.pt \
  --no-shadows --no-reflections --no-debug-viz \
  --output /workspace/clips/tier1_spin.mp4'"
```

Curriculum key for `ang_vel_z` and `lin_vel_y` stages to confirm at run time; linear
stages for x and y may share one key or be separate.

**Cost estimate:** ~1 h GPU.

**Success criteria:** Clip shows the robot spinning continuously at the commanded yaw
rate without translating significantly; base position remains within ~0.5 m of start
over a 10-second clip; the spin direction follows the commanded sign.

---

## 5. Prescribed-gait (hop / march)

**Goal:** Train the G1 to follow a specific footfall rhythm — either a two-beat hop
(both feet leaving and landing together) or a four-beat march (strict left-right-
left-right with a prescribed timing). Unlike the baseline, where gait pattern is
entirely emergent, this task uses a **contact-schedule reward** that gives bonus
reward only when the correct foot is on the ground at the correct phase of a repeating
cycle. This is the lightest form of gait templating.

**Env + reward terms to touch:** `Mjlab-Velocity-Flat-Unitree-G1`. The contact-
schedule reward is not in the existing velocity task — it must be added. The reward
checks, at each timestep, which phase of a fixed-period cycle the episode is in and
gives a bonus if the expected foot (left or right, or both for hop) has non-zero
ground contact. The reward weight should be large enough to break the emergent gait
but not so large that velocity tracking collapses.

**New-code flag:** Light new code — a contact-schedule reward term. Implementation:
add a one-function reward class inside `mjlab-dev` (the container's editable install
picks it up immediately). Follow the `docker cp` or in-container-edit path from
CLAUDE.md; run `ruff format && ruff check --fix && pyright` inside the container
before training. The contact detection API to use: check an existing foot-contact or
`feet_air_time` term in the codebase for how contact bodies are queried.

**Train/record commands:**

```bash
# Train — after adding the contact-schedule reward term to the codebase
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Unitree-G1 \
  \"--env.commands.twist.ranges.lin-vel-x=(0.0, 1.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.0.lin-vel-x=(0.0, 1.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.1.lin-vel-x=(0.0, 1.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.2.lin-vel-x=(0.0, 1.0)\" \
  --env.rewards.contact_schedule.weight 2.0 \
  --env.rewards.contact_schedule.params.period 0.5 \
  --agent.max-iterations 3000'"

# Record — side view at normal walking speed
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_velocity/<timestamp>/model_<best_iter>.pt \
  --no-shadows --no-reflections --no-debug-viz \
  --output /workspace/clips/tier1_march.mp4'"
```

The reward term name (`contact_schedule`) and its parameter keys (`period`, gait
type, foot body IDs) are placeholders to be set when the term is written. Pre-commit
check: `docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && ruff format && ruff check --fix && pyright'`.

**Cost estimate:** ~1 h GPU for the run, plus 1–2 h for the new-code step (write,
copy into container, pyright-clean, smoke-test).

**Success criteria:** Clip clearly shows the prescribed footfall pattern — either
both feet leaving the ground together (hop) or strict left-right alternation at the
target period; the pattern is consistent across at least 10 steps; the robot still
makes forward progress at the commanded speed.

---

## 6. Backward / sideways walking

**Goal:** Train the G1 to walk backward (negative `lin_vel_x`) and/or sideways
(positive or negative `lin_vel_y`) on command. The baseline policy was trained with
`lin_vel_x` in `(-1.0, 1.0)` so backward walking may partially exist, but it was
not verified or highlighted. This task pins specific velocity commands and checks
whether the resulting policy is stable and legible in both directions.

**Env + reward terms to touch:** `Mjlab-Velocity-Flat-Unitree-G1`. No reward changes
needed — the existing velocity-tracking reward (`track_lin_vel`, `track_ang_vel`)
already handles any commanded direction. The intervention is purely in the commanded
range: include negative `lin_vel_x` and non-zero `lin_vel_y`, override all three
curriculum stages for both axes. Exact curriculum key for `lin_vel_y` to confirm at
run time (parallel to the `lin_vel_x` key pattern).

**New-code flag:** None.

**Train/record commands:**

```bash
# Train — backward + sideways range
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Unitree-G1 \
  \"--env.commands.twist.ranges.lin-vel-x=(-1.5, 1.5)\" \
  \"--env.commands.twist.ranges.lin-vel-y=(-1.0, 1.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.0.lin-vel-x=(-1.5, 1.5)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.1.lin-vel-x=(-1.5, 1.5)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.2.lin-vel-x=(-1.5, 1.5)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.0.lin-vel-y=(-1.0, 1.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.1.lin-vel-y=(-1.0, 1.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.2.lin-vel-y=(-1.0, 1.0)\" \
  --agent.max-iterations 2000'"

# Record — side view (backward) and front view (sideways); fixed commands for each clip
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_velocity/<timestamp>/model_<best_iter>.pt \
  --no-shadows --no-reflections --no-debug-viz \
  --output /workspace/clips/tier1_backward.mp4'"
```

To capture a sideways clip: record a second pass with the lateral-command equivalent
(pin `lin_vel_y` to +1.0 or -1.0 via the `vel_command_b` override pattern documented
in CLAUDE.md's "Rendering / recording gotchas" section).
Curriculum key for `lin_vel_y` stages to confirm at run time.

**Cost estimate:** ~1 h GPU.

**Success criteria:** Two clips — one clearly showing the robot walking backward
(torso facing forward, body moving rearward), one showing lateral drift to the left
or right at a commanded sideways velocity; both gaits are stable across at least
5 seconds; reward in `params/env.yaml` confirms both axes were trained.
