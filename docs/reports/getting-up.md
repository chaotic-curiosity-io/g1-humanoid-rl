# Getting Up: Teaching a Robot to Stand From the Floor

> **Draft.** This report is populated after the S3 training run (see [the spec](../superpowers/specs/2026-06-19-spine-getup-recovery.md) for the plan). The structure below is the skeleton it will fill.

*This report is the From-Scratch Tasks track's entry. If you are reading this without reading the earlier reports, start with [00-primer.md](00-primer.md) for the vocabulary. Terms like [policy](00-primer.md), [reward](00-primer.md), and [episode](00-primer.md) are defined there.*

---

## What you are about to see

Every robot in this series so far has started each episode standing upright. Falling was the failure condition: fall, and the episode resets. The reward assumed the robot would try to stay standing, and it did, because staying standing was the only way to keep collecting reward.

This report removes that assumption entirely.

The robot starts each episode lying on the ground — face-up, face-down, or crumpled at a random angle. It has no velocity command to follow. It has no reference motion to imitate. It has only a reward function that gives higher scores when it is more upright, and a task that ends successfully only when it has managed to stand up and hold that position for a few continuous seconds.

Everything else — the strategy, the sequence of movements, the way it uses its arms and legs — the robot has to discover on its own, through trial and error.

**Why this is interesting and difficult in equal measure:** when you design the objective from scratch, the robot will find the shortest path to the highest score. And the shortest path is not always the one you intended. This task is the most likely in this entire series to produce "reward hacking" — strategies the robot finds that technically satisfy the reward but are completely unlike what you meant. Documenting those hacks honestly is as important as documenting the final successful behavior.

---

## The command

This is the only task in this series that requires writing new code before training begins. The task `Mjlab-Recovery-Flat-Unitree-G1` does not exist yet — it has to be built.

**Step 1: Write the new task (in-container)**

The new task is written as a Python class inside the mjlab source tree on the Spark. Because `mjlab/` is owned by root on the Spark host, do not `sudo chown` the tree — use `docker cp` from `/tmp` or edit directly inside the container:

```bash
# Copy the task file into the container:
scp scripts/recovery_task.py spark:/tmp/recovery_task.py
ssh spark "docker cp /tmp/recovery_task.py \
  mjlab-dev:/workspace/mjlab/src/mjlab/tasks/recovery_task.py"
```

The task must implement these reward terms: `base_height_rising` (reward the pelvis rising), `torso_upright` (reward vertical torso alignment), `stability_bonus` (reward holding the standing pose for consecutive seconds), `joint_velocity_penalty` (penalize thrashing), and `joint_torque_penalty` (penalize forceful movement). Critically, it must *not* terminate on falling — every episode starts fallen, so a falling-termination would kill every episode at step 1. Only two terminations are valid: success-hold (the robot has stood stably long enough) and timeout.

**Step 2: Pre-commit checks**

Before training, run the full pre-commit suite inside the container:

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && ruff format && ruff check --fix && pyright'"
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && pytest tests/'"
```

Both must pass before proceeding. Type errors in reward managers often signal wrong tensor shapes that produce NaN rewards at runtime.

**Step 3: Smoke-train probe**

Verify the task registers and steps without errors before committing to a full training run:

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Recovery-Flat-Unitree-G1 \
  --agent.max-iterations 200 \
  --agent.num-envs 1024 \
  --agent.seed 0'"
```

Kill after ~50 printed iterations. Verify: no import errors; rewards are finite (not NaN); episodes are not terminating at step 1 (which would mean the fallen-pose termination is mis-triggered on start).

**Step 4: Full training run (iterA)**

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Recovery-Flat-Unitree-G1 \
  --agent.max-iterations 5000 \
  --agent.num-envs 4096 \
  --agent.seed 42'"
```

**Step 5: Record and review**

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Recovery-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_recovery/<timestamp>/model_<best_iter>.pt \
  --no-shadows --no-reflections --no-debug-viz \
  --cameras chase side front top grid \
  --output /workspace/clips/s3_final_{camera}.mp4'"
scp spark:/workspace/clips/s3_*.mp4 docs/reports/assets/
```

See the [spec](../superpowers/specs/2026-06-19-spine-getup-recovery.md) for the full sequence including the host-quiesce bracket, the mjlab source-ownership rules, and the complete smoke-train verification checklist.

---

## Results

*This section is populated after the S3 training run and visual verification. Placeholders below describe what will appear here.*

### The training curve

**[Placeholder: reward curve plot — `assets/s3_reward_curve.png`]**

*Mean reward over training iterations. For this task, what matters in the curve is different from the walking curves: a flat-zero curve likely means the termination-on-falling bug (episodes ending at step 1, no learning signal). Any upward movement is evidence the robot is discovering how to lift itself.*

### The get-up behavior

**[Placeholder: final multi-camera clips — `assets/s3_final_side.mp4`, `assets/s3_final_grid.mp4`]**

*Four camera angles. What to look for: the robot starting flat on the ground, a recognizable sequence of movements toward upright, and the robot reaching and holding a stable stand. A viewer with no ML background should be able to watch this clip and say "yes, it figured out how to stand up."*

### The iteration log: what the robot tried first

**[Placeholder: prose iteration log — written after the run completes, while the context is fresh.]**

This section documents every distinct strategy the robot found during the shaping process — including the ones that were wrong:

- **IterA:** what reward design was tried first? What behavior emerged? Was it a genuine stand-up, or a hack (elbow-prop, height-without-uprightness, jitter-stand)? What was changed for iterB?
- **IterB (if needed):** same.
- How many iterations before the final design produced reliable stand-ups?

Each hack found during this process is documented here and cross-linked to [reward-hacking-gallery.md](reward-hacking-gallery.md), which collects the canonical record of hacks found across the entire program.

### Common hacks to watch for

The reward design for this task is the most prone to hacking of anything in this series. Three failure modes to check in the render before declaring success:

**Height without uprightness** — the robot raises its pelvis (by kicking its legs into the air, for example) without actually going vertical. This scores high on `base_height_rising` but the robot ends up upside-down, not standing.

**Elbow-prop** — the robot gets partially up by bracing on one elbow and satisfies the uprightness threshold at a shallow angle, never reaching full standing. Looks like progress in the reward curve; looks like a bad push-up in the video.

**Jitter-stand** — the robot flails to a near-standing pose just long enough to trigger the success-hold timer, then falls, earning the stability bonus repeatedly. Each episode ends in "success" even though the robot is falling immediately after the timer fires.

If the clips show any of these patterns, the reward design needs adjustment. The iteration log will document which hacks appeared and what terms closed them.

---

## Tweak this to explore

**Adjust the success-hold duration.** The task terminates successfully when the robot has stood for a continuous hold period (e.g. 2 seconds). Shorten this and the jitter-stand hack becomes easier to exploit. Lengthen it and the policy needs to demonstrate genuine, sustained balance — but convergence takes longer.

**Modify the reward weights.** The balance between `base_height_rising`, `torso_upright`, and `stability_bonus` determines what the robot prioritizes. Raise `torso_upright` relative to `base_height_rising` and the robot is penalized more for horizontal postures. Raise `stability_bonus` and the robot is rewarded more for holding a stand once it achieves it. Each change closes some hacks and potentially opens others.

**Try randomizing the fallen pose more aggressively.** The task spawns the robot in a randomized fallen pose. A wider randomization range (more varied starting angles, more crumpled joint configurations) produces a more robust policy that can get up from a wider variety of falls — but it also makes learning harder, especially in early iterations.

**Watch the connection to reward hacking.** This task is the live version of the [reward-hacking gallery](reward-hacking-gallery.md). Every hack the robot finds here is a specimen for that report. Keep notes during the iteration process — the freshest context produces the clearest explanations.

---

*All experiments use the Unitree G1 on flat terrain, trained with the MuJoCo-Warp simulator on a DGX Spark (NVIDIA GB10, aarch64). The spec for this run: [2026-06-19-spine-getup-recovery.md](../superpowers/specs/2026-06-19-spine-getup-recovery.md).*
