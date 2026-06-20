# Compact Specs: Tier-4 Object / Whole-Body Interaction (Spec-Only)

**Date:** 2026-06-19
**Branch:** `g1-skills-curriculum`
**Status:** Spec-only — NOT trained in this program
**Track:** Object / whole-body (Tier-4 long tail)
**Linked from:** `docs/reports/README.md` (syllabus "Ready-to-run" section)

---

## Important: Tier 4 is spec-only in this program

These three tasks are **the highest-risk, highest-engineering-effort work in the
entire curriculum.** They are not trained here. They are documented so the ideas
are not lost and so a future session can promote any one of them without redesigning
from scratch — but that promotion is a significant project, not a weekend run.

Why the caution? Every other tier starts from an existing mjlab environment and
touches only reward weights or adds a reward term. Tier-4 tasks require **scene
surgery**: a free body (a ball, a box, a target marker) must be added to the MuJoCo
XML model that defines the simulation world, the environment code must be extended to
track that body's state, and entirely new reward terms must be written against it.
This is the heaviest engineering of any tier — closer to "write a new research
environment" than "tune a knob." The relevant APIs (how to add a free body to a
mjlab env's MJCF, how to read its runtime state, how to reset it per episode) are
not fully visible from this checkout and must be confirmed in-container before any
training can begin.

There is also genuine research risk: reach, kick, and carry are active areas of
humanoid whole-body manipulation research; the behaviors may require curriculum
stages, initial-condition engineering, or reward-shaping iteration well beyond what
a velocity or tracking task needs. Treat every field below as a **best-effort
sketch**, with API specifics to confirm at run time.

---

## What "compact spec" means

A compact spec gives you everything needed to promote a task to a full training run
later — without redesigning from scratch. Each section is terse: goal, what to
change in the environment, whether new code is required, the exact commands to train
and record, a cost estimate, and the criteria for calling it done.

Read this file alongside the [Tier-3 compact specs](./2026-06-19-tier3-tasks-compact.md)
and the [S3 get-up full spec](./2026-06-19-spine-getup-recovery.md) — the new-code
patterns (docker cp path, pre-commit checks, ownership rules) are identical; this
file does not repeat them. Tier 4 adds one more step that Tier 3 does not have:
**MJCF scene editing** to introduce the object into the world.

None of these tasks has been trained. They are speculative, not yet run.

---

## 1. Reach-to-Target

**Goal:** Train the G1 to extend one arm and touch (or hold near) a stationary
3-D target point — a sphere or marker placed in the scene. The robot must move its
end-effector (wrist or hand body) close to the target while staying upright and
balanced. No locomotion is required; the robot stands in place.

**Env + reward terms to touch:** A new task forked from
`Mjlab-Velocity-Flat-Unitree-G1` — call it `Mjlab-Reach-Flat-Unitree-G1` for
planning purposes; the actual name is set in the task class. Key additions:

- **MJCF scene edit — target body:** add a small visual-only sphere or site to the
  MuJoCo XML (the `.xml` file that defines the G1 and its world plane). Position it
  at a randomised but reachable 3-D coordinate relative to the robot's base each
  episode. How to add a body to a mjlab env's MJCF and how to update its position at
  reset time — via `model.body_pos`, `data.mocap_pos`, or a mjlab scene-asset API —
  must be confirmed in-container. This is the principal new engineering step.
- **End-effector-to-target distance reward:** each timestep, compute the Euclidean
  distance from the right (or left) wrist body position to the target sphere centre,
  and give a reward that increases as the distance shrinks — e.g. a negative-distance
  penalty or a Gaussian centred on zero distance. The mjlab API for reading a named
  body's world position at runtime (`data.xpos[body_id]` in raw MuJoCo) to confirm at
  run time.
- **Balance / uprightness reward:** retain or add a `torso_upright` term so the robot
  does not tip forward while reaching. The same concept as the get-up spec's
  uprightness term.
- **No velocity command:** remove `UniformVelocityCommand`; the sole goal is reaching
  the target.
- **Termination:** time-out only; do not terminate on near-miss. A generous episode
  length lets the policy explore.

**New-code flag:** Yes, the most of any tier — MJCF scene edits to add a free body /
target site + new reward terms against it. The MJCF editing API for mjlab (whether
via a Python asset builder, a YAML config, or direct XML patching) is the largest
unknown; confirm by reading an existing task config inside `mjlab-dev` that adds
non-robot bodies to the scene (search for `worldbody` or `freejoint` in the task
configs). Follow the `docker cp` / in-container-edit path from the get-up spec;
run `ruff format && ruff check --fix && pyright` then `pytest` before training.

**Train/record commands:**

```bash
# Step 1 — write the new task and MJCF scene extension inside mjlab-dev
# Edit path: docker cp or in-container editor (see the get-up spec "Concrete runs").
# Confirm mjlab's MJCF asset API by reading an existing multi-body task config:
ssh spark "docker exec mjlab-dev bash -lc \
  'grep -ril \"freejoint\\|worldbody\\|asset\" /workspace/mjlab/src/mjlab || true'"

# Step 2 — pre-commit checks (no uv, no make)
ssh spark "docker exec mjlab-dev bash -lc \
  'cd /workspace/mjlab && ruff format && ruff check --fix && pyright'"
ssh spark "docker exec mjlab-dev bash -lc \
  'cd /workspace/mjlab && pytest tests/'"

# Step 3 — smoke-train probe (confirm target body spawns and reward is non-NaN)
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Reach-Flat-Unitree-G1 \
  --agent.max-iterations 200 \
  --agent.num-envs 512 \
  --agent.seed 0'"

# Step 4 — full training run (iterations and num-envs to calibrate at run time)
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Reach-Flat-Unitree-G1 \
  --agent.max-iterations 5000 \
  --agent.num-envs 4096 \
  --agent.seed 42'"

# Step 5 — record (front and side views show arm extension and target approach)
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Reach-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_reach/<timestamp>/model_<best_iter>.pt \
  --no-shadows --no-reflections --no-debug-viz \
  --cameras front side \
  --output /workspace/clips/tier4_reach_{camera}.mp4'"

# Pull clips to Windows
scp spark:/workspace/clips/tier4_reach_*.mp4 docs/reports/assets/
```

Task name (`Mjlab-Reach-Flat-Unitree-G1`) and checkpoint directory (`g1_reach/`) are
placeholders — the `experiment_name` in the task class determines the log directory;
confirm from the startup printout. MJCF API and body-position query API (exact
attribute paths on `model` and `data`) to confirm in-container before writing any
code. Host-quiesce before training: `docker stop open-webui compose-arangodb-1 ollama-compose`
then `free -h`. SIGINT to the training child: `docker exec mjlab-dev bash -lc "kill -INT \$(pgrep -f mjlab.scripts.train)"`.

**Cost estimate:** ~4–8 hours new-code step (MJCF scene edit + reward terms + API
discovery + pre-commit cycle) + ~3–6 hours GPU training (reward-shaping iteration
likely; early policies may ignore the target entirely or collapse balance while
reaching). Highest per-task cost in the curriculum.

**Success criteria (visually verified):** Front-view clip shows the robot extending
one arm toward a visible target marker and holding the wrist within roughly 0.1–0.2 m
of it for at least 3 consecutive seconds; the robot remains upright throughout; the
behavior repeats across multiple episodes with randomised target positions. A policy
that reaches the correct general direction but never closes within visible range does
not pass.

---

## 2. Kick-a-Ball

**Goal:** Train the G1 to walk up to a ball placed on the ground and kick it toward
a goal area — the simplest whole-body object-interaction task. The robot must
approach, time a kick with one leg, and impart velocity to the ball in the target
direction. This requires coordinating locomotion with a contact interaction, which is
harder than pure reaching.

**Env + reward terms to touch:** A new task forked from
`Mjlab-Velocity-Flat-Unitree-G1` — call it `Mjlab-KickBall-Flat-Unitree-G1`. Key
additions:

- **MJCF scene edit — ball (free body):** add a sphere with a `freejoint` to the
  MuJoCo XML so it is a fully dynamic rigid body that can be kicked. Set mass and
  friction appropriate for a soccer-ball-like object (to calibrate at run time).
  Place it at a randomised starting position in front of the robot each episode. The
  MJCF free-body API in mjlab (how to add a sphere with a freejoint, how to reset its
  position and zero its velocity each episode) to confirm in-container — this is the
  same API question as in the reach task.
- **Ball-velocity-toward-goal reward:** each timestep, reward the component of the
  ball's velocity (`data.qvel` for the ball's freejoint) in the direction of a
  designated goal area. A ball at rest gives zero reward; a ball moving toward the
  goal gives positive reward proportional to speed. The goal direction can be fixed
  (world +x) or randomised per episode.
- **Approach sub-reward (optional curriculum stage):** in early training, give a small
  reward for the robot's foot being close to the ball — an approach incentive that
  encourages the robot to move toward the ball before learning to kick it. Remove this
  term or reduce its weight once kicking behaviour emerges, to avoid the robot
  hovering near the ball rather than kicking.
- **Uprightness / balance reward:** the robot must not fall over while kicking. Retain
  or add a torso-uprightness term.
- **Velocity command:** keep a low-range `lin_vel_x` command (e.g. 0.3–0.8 m/s) to
  encourage the robot to approach the ball rather than standing still.

**New-code flag:** Yes, the most of any tier — MJCF scene edits to add a free body
with a freejoint + new reward terms reading that body's velocity and position. The
free-body API (adding a freejoint sphere, resetting its pose/velocity each episode,
reading its state from `data.qpos`/`data.qvel`) is the central unknown; confirm by
checking whether any existing mjlab task or example uses a free body. Mjlab's episode
reset mechanism must be extended to reposition the ball each episode. Follow the same
`docker cp` / in-container-edit path and pre-commit checks as all Tier-3/4 tasks.

**Train/record commands:**

```bash
# Step 1 — write the new task and MJCF ball addition inside mjlab-dev
# Confirm free-body API before writing:
ssh spark "docker exec mjlab-dev bash -lc \
  'grep -ril \"freejoint\\|free_joint\\|ball\\|qvel\" /workspace/mjlab/src/mjlab || true'"

# Step 2 — pre-commit checks
ssh spark "docker exec mjlab-dev bash -lc \
  'cd /workspace/mjlab && ruff format && ruff check --fix && pyright'"
ssh spark "docker exec mjlab-dev bash -lc \
  'cd /workspace/mjlab && pytest tests/'"

# Step 3 — smoke-train probe (confirm ball spawns, freejoint DOFs appear in qpos,
# reward is non-NaN, and at least one episode shows foot-ball contact)
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-KickBall-Flat-Unitree-G1 \
  --agent.max-iterations 200 \
  --agent.num-envs 512 \
  --agent.seed 0'"

# Step 4 — full training run
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-KickBall-Flat-Unitree-G1 \
  --agent.max-iterations 8000 \
  --agent.num-envs 4096 \
  --agent.seed 42'"

# Step 5 — record (wide side view shows approach + kick arc most clearly)
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-KickBall-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_kickball/<timestamp>/model_<best_iter>.pt \
  --no-shadows --no-reflections --no-debug-viz \
  --cameras side chase \
  --output /workspace/clips/tier4_kickball_{camera}.mp4'"

# Pull clips to Windows
scp spark:/workspace/clips/tier4_kickball_*.mp4 docs/reports/assets/
```

Task name and checkpoint directory are placeholders — confirm from the startup
printout. Ball freejoint DOF count (7 for position+quaternion) must be accounted for
in any observation or reset code that indexes `data.qpos`. Higher iteration budget
than reach (8,000 vs 5,000) because approach + timing + kick is a longer credit
assignment chain. All host-quiesce and SIGINT procedures identical to other Tier-3/4
tasks.

**Cost estimate:** ~6–10 hours new-code step (free-body MJCF edit + episode reset +
reward terms + API discovery + smoke-test iteration) + ~5–8 hours GPU training
(credit assignment over approach→kick is longer than single-step tasks; curriculum
staging the approach reward is likely necessary). The highest combined cost of the
three Tier-4 tasks.

**Success criteria (visually verified):** Side-view clip shows the robot walking
toward the ball, making visible foot contact with it, and sending it rolling or
flying in the goal direction — at least 3 episodes in the recording. The kick must
be a deliberate contact event, not an accidental deflection from normal walking.
A policy that merely walks through the ball without directing it does not pass.

---

## 3. Carry an Object

**Goal:** Train the G1 to pick up (or start holding) a box or cylinder and walk
forward while keeping it off the ground — a simple carry task. The robot must
simultaneously maintain locomotion and hold an arm configuration that prevents the
object from falling. No dexterous grasping is attempted; the object can be
assumed to start already held or cradled (i.e. the task begins with the object
attached to the arms or in a grasped state), side-stepping the grasp-initiation
problem which would require contact-rich manipulation beyond this scope.

**Env + reward terms to touch:** A new task forked from
`Mjlab-Velocity-Flat-Unitree-G1` — call it `Mjlab-Carry-Flat-Unitree-G1`. Key
additions:

- **MJCF scene edit — carried object:** add a box or cylinder body to the MJCF. Two
  approaches to explore at run time: (a) a free body that is manually welded to the
  robot's wrist via a constraint each episode (constraint API in MuJoCo to confirm),
  or (b) a body attached to the wrist via a fixed joint from the start of each episode
  (simpler but less physically realistic). Approach (b) is recommended for a first
  attempt: add the object body to the robot's MJCF as a child of the wrist link with
  a fixed joint, at a position that simulates holding. This avoids the grasp-contact
  problem entirely.
- **Hold-arm-pose reward:** reward the wrist/elbow joint angles staying within a
  holding configuration — a `joint_pos` target term for the arm joints, centred on a
  pose that keeps the object lifted off the ground. The exact arm joint names in the
  G1 MJCF to confirm in-container (`grep joint /workspace/mjlab/src/mjlab/assets/g1.xml`
  or equivalent path).
- **Object-height reward:** reward the carried object body's height remaining above a
  minimum threshold (e.g. 0.5 m above ground). If the object is a fixed child of the
  wrist, this is equivalent to rewarding the wrist height, which is simpler.
- **Walk-while-carrying:** retain the `track_lin_vel` velocity-tracking reward so the
  robot must walk at a commanded speed while carrying. A low forward speed (0.3–0.8 m/s)
  is safer for a first run.
- **Uprightness / balance:** torso-uprightness term as in reach and kick.

**New-code flag:** Yes, the most of any tier — MJCF scene edits to attach an object
body to the robot's wrist, new reward terms for arm pose and object height, and
modifications to the episode reset to initialise the arm in the holding configuration.
The fixed-joint approach (object as wrist child) minimises contact API unknowns but
still requires MJCF editing and arm-joint targeting. The wrist joint name, object
attachment point, and the `joint_pos` reward term's parameter format (which joints,
which target angles) all to confirm in-container. Follow the standard `docker cp` /
in-container-edit path and pre-commit checks.

**Train/record commands:**

```bash
# Step 1 — write the new task; attach object body to G1 MJCF wrist link
# Confirm arm joint names first:
ssh spark "docker exec mjlab-dev bash -lc \
  'grep -i \"joint\" /workspace/mjlab/src/mjlab/assets/g1*.xml | head -60 || true'"

# Step 2 — pre-commit checks
ssh spark "docker exec mjlab-dev bash -lc \
  'cd /workspace/mjlab && ruff format && ruff check --fix && pyright'"
ssh spark "docker exec mjlab-dev bash -lc \
  'cd /workspace/mjlab && pytest tests/'"

# Step 3 — smoke-train probe (confirm object does not clip through the ground at
# episode start, arm joints do not explode, reward is non-NaN)
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Carry-Flat-Unitree-G1 \
  --agent.max-iterations 200 \
  --agent.num-envs 512 \
  --agent.seed 0'"

# Step 4 — full training run
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Carry-Flat-Unitree-G1 \
  --agent.max-iterations 6000 \
  --agent.num-envs 4096 \
  --agent.seed 42'"

# Step 5 — record (side and chase views show both the arm posture and forward walking)
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Carry-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_carry/<timestamp>/model_<best_iter>.pt \
  --no-shadows --no-reflections --no-debug-viz \
  --cameras side chase \
  --output /workspace/clips/tier4_carry_{camera}.mp4'"

# Pull clips to Windows
scp spark:/workspace/clips/tier4_carry_*.mp4 docs/reports/assets/
```

Task name and checkpoint directory are placeholders. G1 MJCF asset path
(`/workspace/mjlab/src/mjlab/assets/g1*.xml`) to confirm at run time — the actual
filename may differ. Object body naming, fixed-joint attachment point, and arm-joint
`joint_pos` target angles all to confirm before writing the task class. All
host-quiesce and SIGINT procedures identical to other Tier-3/4 tasks.

**Cost estimate:** ~4–8 hours new-code step (MJCF edit + arm-pose reward + reset
initialisation + pre-commit cycle) + ~4–6 hours GPU training (walk-while-carrying
combines two skills the policy must learn simultaneously; expect more shaping
iterations than pure locomotion). Overall second-highest combined cost after kick-a-ball.

**Success criteria (visually verified):** Side-view clip shows the robot walking
forward at the commanded speed with both arms in a visible carrying posture (arms
elevated, object held above the ground) for at least 5 consecutive seconds; the
object does not clip through the robot's body or the ground; the arm configuration
is stable, not flapping. A policy that walks but drops its arms to a natural position
(losing the carry posture) does not pass.
