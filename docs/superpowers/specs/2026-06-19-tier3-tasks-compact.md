# Compact Specs: Tier-3 Deferred From-Scratch Tasks

**Date:** 2026-06-19
**Branch:** `g1-skills-curriculum`
**Status:** Deferred — ready to run when prioritised
**Track:** From-scratch tasks (Tier-3 long tail)
**Linked from:** `docs/reports/README.md` (syllabus "Ready-to-run" section)

---

## What "compact spec" means

A compact spec gives you everything needed to promote a task to a full training run
later — without redesigning from scratch. Each section is deliberately terse: goal,
what to change in the environment, whether new code is required, the exact commands
to train and record, a cost estimate, and the criteria for calling it done.

Both tasks below are **from-scratch task rewards** — no reference motion, no
commanded velocity target. The robot is rewarded for a behavior we define entirely
through the reward terms we write. This is the same paradigm as the S3 get-up spine
task; both tasks here mirror that approach: a new mjlab task class, a new reward
manager, and custom initial-state handling. Before promoting either task, read the
[S3 get-up spec](./2026-06-19-spine-getup-recovery.md) — it documents the full
new-task pattern in detail, including the ownership rules for editing `mjlab/`, the
pre-commit checks, and the smoke-train protocol. This file only records the
task-specific differences.

Neither task has been trained yet. They are ready-to-run, not yet run.

---

## 1. Push-Recovery / Disturbance Rejection

**Goal:** Train the G1 to stay upright when external forces are applied to its
body during an episode — a deliberate push, shove, or sustained lateral pressure.
The result is a policy that can recover from disturbances rather than simply
maintaining an undisturbed stance.

**Env + reward terms to touch:** Fork `Mjlab-Velocity-Flat-Unitree-G1` into a new
task (e.g. `Mjlab-PushRecovery-Flat-Unitree-G1`). Key additions:

- **External-force application:** apply random impulses or sustained lateral forces
  to the robot's base (pelvis) during training. In MuJoCo this is typically done by
  writing to `data.xfrc_applied` for the base body at randomised intervals and
  magnitudes; the mjlab API for this (e.g. a domain-randomization callback or a
  per-step force injection) must be confirmed at run time — see the get-up spec's
  "Open questions" pattern and check existing domain-randomization terms in the
  velocity task's `params/env.yaml`. This is the principal new-code element and the
  distinguishing feature of push-recovery training.
- **Stay-upright reward:** reward the torso uprightness vector remaining close to
  the world "up" direction each timestep (a `torso_upright` term, the same concept
  as in the get-up spec). Penalise large deviations from upright that persist beyond
  a recovery window.
- **Base-position penalty (optional):** reward the robot for not drifting far from
  its original position after a push — keeps the episode in frame and prevents the
  policy from recovering by skating away.
- **Retain velocity-tracking reward (optional):** if combined with walking, keep a
  moderate `track_lin_vel` weight so the policy must recover *and* continue moving.
  A stationary-stance-only variant (zero velocity command) is simpler to train first.
- **No termination on falling during pushes:** if a push causes a fall, allow the
  episode to continue unless the robot cannot recover within a timeout; terminating
  immediately on fall prevents the policy from learning to get back up after a hard
  shove.

**New-code flag:** Yes — new task class + reward manager. The external-force
injection in particular requires new code: either a per-step `xfrc_applied` write
in the task's `step()` method, or a new domain-randomization term if mjlab exposes
that hook. Follow the `docker cp` / in-container-edit path from the get-up spec;
see that spec's "Concrete runs — Step 1" for the ownership and copy pattern. Exact
mjlab API for `xfrc_applied` or equivalent force injection to confirm at run time.

**Train/record commands:**

```bash
# Step 1 — write the new task inside mjlab-dev
# Follow the docker cp or in-container-edit path from the get-up spec.
# Write to /workspace/mjlab/src/mjlab/tasks/push_recovery_task.py (path to confirm
# at run time against the existing task package layout).
scp scripts/push_recovery_task.py spark:/tmp/push_recovery_task.py
ssh spark "docker cp /tmp/push_recovery_task.py \
  mjlab-dev:/workspace/mjlab/src/mjlab/tasks/push_recovery_task.py"

# Step 2 — pre-commit checks (no uv, no make — use raw tools as in the get-up spec)
ssh spark "docker exec mjlab-dev bash -lc \
  'cd /workspace/mjlab && ruff format && ruff check --fix && pyright'"
ssh spark "docker exec mjlab-dev bash -lc \
  'cd /workspace/mjlab && pytest tests/'"

# Step 3 — smoke-train probe (kill after ~50 iterations; verify non-NaN rewards)
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-PushRecovery-Flat-Unitree-G1 \
  --agent.max-iterations 200 \
  --agent.num-envs 1024 \
  --agent.seed 0'"

# Step 4 — full training run
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-PushRecovery-Flat-Unitree-G1 \
  --agent.max-iterations 5000 \
  --agent.num-envs 4096 \
  --agent.seed 42'"

# Step 5 — record (side view catches the push and recovery arc most clearly)
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-PushRecovery-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_push_recovery/<timestamp>/model_<best_iter>.pt \
  --no-shadows --no-reflections --no-debug-viz \
  --cameras side chase front \
  --output /workspace/clips/tier3_push_recovery_{camera}.mp4'"

# Pull clips to Windows
scp spark:/workspace/clips/tier3_push_recovery_*.mp4 docs/reports/assets/
```

Host-quiesce before training: `docker stop open-webui compose-arangodb-1 ollama-compose`
then `free -h` (target ~110 GiB free). Restart afterward. SIGINT to the training
child: `docker exec mjlab-dev bash -lc "kill -INT \$(pgrep -f mjlab.scripts.train)"`.
Task name (`Mjlab-PushRecovery-Flat-Unitree-G1`) and checkpoint directory
(`g1_push_recovery/`) are placeholders — the actual `experiment_name` set in the new
task class determines the log directory; confirm from the startup printout.

**Cost estimate:** ~2–4 hours new-code step (task class + force injection + pre-commit
cycle) + ~3–6 hours GPU training (iterative reward shaping likely; push-recovery is
prone to hacks such as crouching low to reduce topple angle without actually recovering).

**Success criteria (visually verified):** Side-view clip shows the robot receiving a
visible push (lateral displacement of the torso), then actively correcting its posture
and returning to upright — not just swaying passively or falling and restarting. The
recovery must be visible as a deliberate sequence of joint adjustments, not a random
flail that happens to land upright. A range of push magnitudes should be tested: the
policy should handle small nudges cleanly and show a more pronounced recovery effort
on hard shoves.

---

## 2. Single-Leg / Flamingo Balance

**Goal:** Train the G1 to stand on one leg while holding the other foot off the
ground — the pose sometimes called "flamingo balance." The robot must find and
maintain a stable single-leg stance without falling, using only arm and torso
adjustments for balance. This is a pure balance task: no walking, no motion target,
no forward velocity.

**Env + reward terms to touch:** Fork `Mjlab-Velocity-Flat-Unitree-G1` into a new
task (e.g. `Mjlab-FlamingoBalance-Flat-Unitree-G1`). Key reward terms:

- **One-foot-off-ground reward:** give a bonus each timestep that the commanded
  "raised" foot has zero ground contact and its height is above a minimum threshold
  (e.g. 0.15 m above ground). Which foot is raised can be fixed (always left or
  always right) or randomised per episode — randomisation per episode is harder to
  train but more general.
- **Torso uprightness:** same term as the get-up spec — reward the torso "up" vector
  aligning with the world "up" direction. Essential: without it the robot will tip
  sideways to balance on the raised leg's side.
- **Stay-in-place penalty:** penalise lateral drift of the base position — the robot
  must balance in a fixed spot, not hop around.
- **No velocity command:** remove `UniformVelocityCommand` entirely. The sole goal
  is holding the pose.
- **Termination:** end the episode on the standing foot losing contact (fall
  termination is appropriate here, unlike the get-up task, because a single-leg fall
  is genuinely the end of the behavior). Also terminate on timeout.

**New-code flag:** Yes — new task class + reward manager. The one-foot-off-ground
detection requires querying contact state per body (checking which foot body IDs
have ground contact each timestep). The mjlab API for this — likely the same contact
querying used by `feet_air_time` in the velocity task — must be confirmed at run
time: look at the existing `feet_air_time` reward term in the codebase for how it
identifies foot contact bodies and reads contact force or sensor data. Follow the
same `docker cp` / in-container-edit path as the get-up spec; run
`ruff format && ruff check --fix && pyright` then `pytest` before training.

**Train/record commands:**

```bash
# Step 1 — write the new task inside mjlab-dev
scp scripts/flamingo_task.py spark:/tmp/flamingo_task.py
ssh spark "docker cp /tmp/flamingo_task.py \
  mjlab-dev:/workspace/mjlab/src/mjlab/tasks/flamingo_task.py"

# Step 2 — pre-commit checks
ssh spark "docker exec mjlab-dev bash -lc \
  'cd /workspace/mjlab && ruff format && ruff check --fix && pyright'"
ssh spark "docker exec mjlab-dev bash -lc \
  'cd /workspace/mjlab && pytest tests/'"

# Step 3 — smoke-train probe (confirm non-NaN rewards; episodes should not terminate
# immediately — check that the "raised foot off ground" term does not fire at episode
# start if the robot initialises standing on both feet)
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-FlamingoBalance-Flat-Unitree-G1 \
  --agent.max-iterations 200 \
  --agent.num-envs 1024 \
  --agent.seed 0'"

# Step 4 — full training run
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-FlamingoBalance-Flat-Unitree-G1 \
  --agent.max-iterations 5000 \
  --agent.num-envs 4096 \
  --agent.seed 42'"

# Step 5 — record (front and side views are both useful: front shows lateral sway;
# side shows foot height and torso posture)
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-FlamingoBalance-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_flamingo/<timestamp>/model_<best_iter>.pt \
  --no-shadows --no-reflections --no-debug-viz \
  --cameras side front chase \
  --output /workspace/clips/tier3_flamingo_{camera}.mp4'"

# Pull clips to Windows
scp spark:/workspace/clips/tier3_flamingo_*.mp4 docs/reports/assets/
```

Host-quiesce and SIGINT procedures identical to push-recovery (and to the get-up
spec). Task name (`Mjlab-FlamingoBalance-Flat-Unitree-G1`) and checkpoint directory
(`g1_flamingo/`) are placeholders — confirm from the startup printout. One-foot
contact API (exact body IDs, contact force query method) to confirm at run time by
reading the `feet_air_time` reward term source in `mjlab-dev`.

**Cost estimate:** ~2–3 hours new-code step (task class + contact query logic +
pre-commit cycle) + ~2–4 hours GPU training (balance tasks often converge faster
than locomotion but may need reward-weight iteration if the robot discovers cheats
such as leaning against an invisible wall or oscillating to avoid the fall termination).

**Success criteria (visually verified):** Front-view clip shows the robot standing
on one leg with the other foot clearly off the ground (visibly elevated, not just
barely lifted) for at least 5 consecutive seconds; the torso remains upright and
centred over the standing foot without extreme arm flailing; the balance holds across
at least three episodes in the recording. A policy that lifts the foot for one second
and then falls does not pass.
