# Imitation Learning: Teaching G1 to Backflip

> **Draft.** This report is populated after the S2 training run (see [the spec](../superpowers/specs/2026-06-19-spine-backflip.md) for the plan). The structure below is the skeleton it will fill.

*This report is the Imitation track's spine task. It assumes you have read [imitation-cartwheel.md](imitation-cartwheel.md), which introduces motion imitation and documents all the lessons from the cartwheel campaign that this report applies from the start. Terms like [policy](00-primer.md), [reward](00-primer.md), and [episode](00-primer.md) are defined in [00-primer.md](00-primer.md).*

---

## What you are about to see

The cartwheel campaign taught us everything that can go wrong with motion imitation — and then showed us how to fix it. The backflip is the hardest test of those lessons: it requires a genuine airborne phase, full *backward* rotation (the body goes upside-down), and a stable landing. Each of those three phases is a distinct learning challenge.

Unlike the velocity-tracking experiments (reports [01](01-watching-it-learn.md)–[03](03-turning-the-knobs.md)), there is no reward here for speed, direction, or uprightness in the abstract. The policy is rewarded for one thing: making the robot's joints and body match a pre-recorded reference motion, frame by frame. If the robot matches the reference well enough throughout the full backflip — through takeoff, inversion, and landing — a backflip emerges. If it drifts too far from the reference, the episode resets and the attempt starts over.

**The lessons from the cartwheel, applied here from the start:**
- The termination thresholds are set to 0.5 m (not the default 0.25 m) before training begins — not discovered after the first failed iteration.
- The reference is visually inspected before training to confirm it is a single, feasible backflip.
- Every render uses `--termination-threshold 0.5` or `--disable-terminations` — never the default, which would cut every attempt mid-flip.
- The numerical scorer is informational only. A visual, frame-by-frame check is the only verdict.

---

## The command

The full pipeline for the S2 backflip, run from the Windows host into the `mjlab-dev` container on the Spark.

**Step 1: Retarget the reference motion**

```bash
# Convert the MimicKit backflip from SMPL format to G1 joints:
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && python scripts/smpl_backflip_to_g1.py'"

# Convert to the CSV format the NPZ converter expects:
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && \
  python scripts/pkl_to_csv.py \
  /workspace/pose-pipeline/outputs/gmr_pkl/smpl_backflip_to_g1.pkl \
  /workspace/pose-pipeline/outputs/gmr_pkl/smpl_backflip_to_g1.csv'"

# Convert to the NPZ format mjlab's tracking task expects:
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && \
  python -m mjlab.scripts.csv_to_npz \
  /workspace/pose-pipeline/outputs/gmr_pkl/smpl_backflip_to_g1.csv \
  /workspace/pose-pipeline/motions/backflip.npz'"
```

Before training, render the reference to confirm it is a single, feasible backflip (not a double, not truncated):

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/play_motion_npz.py \
  --motion /workspace/pose-pipeline/motions/backflip.npz \
  --output /workspace/clips/s2_reference_preview.mp4'"
scp spark:/workspace/clips/s2_reference_preview.mp4 docs/reports/assets/
```

Watch the preview before proceeding. It should show a single backward rotation from standing through inversion to landing, roughly 2–4 seconds. If it shows two flips or stops mid-motion, fix the reference first.

**Step 2: Train the tracking policy (iterA)**

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file /workspace/pose-pipeline/motions/backflip.npz \
  --env.terminations.anchor-pos.threshold 0.5 \
  --env.terminations.ee-body-pos.threshold 0.5 \
  --agent.num-envs 4096 \
  --agent.max-iterations 20000 \
  --agent.seed 42'"
```

This is the spine's heaviest run: approximately 8–12 hours on the GPU. The host-quiesce bracket (stopping co-tenant containers) must be run before starting. See the [spec](../superpowers/specs/2026-06-19-spine-backflip.md) for the full ops procedure.

**Step 3: Render the result**

```bash
# Render with terminations matching training (recommended first pass):
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Tracking-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_tracking/<timestamp>/model_19999.pt \
  --termination-threshold 0.5 \
  --no-shadows --no-reflections --no-debug-viz \
  --cameras chase side front top grid \
  --output /workspace/clips/s2_final_{camera}.mp4'"
scp spark:/workspace/clips/s2_*.mp4 docs/reports/assets/
```

Replace `<timestamp>` with the run directory printed by the trainer at startup. Both threshold flags must match the training values — using the default 0.25 m would cut every attempt mid-flip on screen.

---

## Results

*This section is populated after the S2 training run and visual verification. Placeholders below describe what will appear here.*

### Reference motion preview

**[Placeholder: reference preview clip — `assets/s2_reference_preview.mp4`]**

*The retargeted backflip reference, rendered before training began. Confirms: single backflip, feasible duration (~2–4 seconds), genuine backward rotation with inversion.*

### The training curve

**[Placeholder: reward curve plot — `assets/s2_reward_curve.png`]**

*Mean reward over 20,000 training iterations. The cartwheel precedent (iterA: stuck around 3–5; iterC from scratch: climbed to ~32 by iter 4,000) gives a rough expectation. A similar early rise signals genuine tracking progress; a flat near-zero curve signals the policy is not completing the motion.*

### The final backflip

**[Placeholder: multi-camera clip grid — `assets/s2_final_grid.mp4`]**

*Four camera angles (chase, side, front, top), rendered with `--termination-threshold 0.5` or `--disable-terminations`. The visual-verification gate: at least one episode in this clip must show all three phases — takeoff, inversion (head below hips), and landing — confirmed by stepping through the clip frame by frame.*

### Iteration log

**[Placeholder: prose iteration log — written after the run completes.]**

The backflip is expected to require multiple iterations, following the cartwheel precedent. Each iteration will be documented here:

- **IterA:** what did the initial training produce? Did the policy show genuine airborne phases, crash-rolls, or no rotation at all?
- **IterB (if needed):** what changed, and why? What did the visual review of iterA show?
- **IterC (if needed):** same.

For each iteration: the key change, what the video showed, and whether it passed the visual verification gate.

### Visual verification statement

**[Placeholder: the verdict — written after frame-by-frame review of the final clip.]**

This section will state clearly:
- How many episodes in the render show a completed backflip (all three phases visible).
- What the numerical scorer reported — and whether it matched the visual review.
- Whether any crash-rolls were present (which can fool a roll-angle-only scorer).

A confirmed backflip requires all three phases visible in the clip. A numerical score is not sufficient.

---

## Tweak this to explore

**Threshold tuning.** The 0.5 m termination threshold that worked for the cartwheel may need adjustment for the backflip. The backflip involves more vertical rotation and a different end-effector trajectory. Try rendering with `--disable-terminations` alongside `--termination-threshold 0.5` to see the difference — if the two clips look very different, the threshold is affecting the render in ways worth understanding.

**The reference matters most.** More than any reward weight or threshold, the quality and feasibility of the reference motion determines whether the policy can learn the backflip. If iterA produces a flop rather than a rotation, inspect `play_motion_npz.py` output again before adjusting training parameters. A bad reference is the hardest failure to diagnose because the training metrics may not reflect it.

**Try `--disable-terminations` for the render even if training used thresholds.** This shows the policy's full behavior without resets — including what happens *after* the backflip (does the robot recover to standing? walk a step? attempt another flip?). Comparing the threshold render to the no-termination render tells you whether the policy has genuinely learned the landing or just learned to stay close to the reference long enough to avoid episode-ending drift.

**Compare backflip to cartwheel.** Both use the same task and pipeline. Once both policies exist, render them side by side at the same camera angle. The structural difference — sideways rotation vs. backward rotation, different end-effector trajectories — will be visible in the timing and shape of the airborne phase.

---

*All experiments use the Unitree G1 on flat terrain, trained with the MuJoCo-Warp simulator on a DGX Spark (NVIDIA GB10, aarch64). The spec for this run: [2026-06-19-spine-backflip.md](../superpowers/specs/2026-06-19-spine-backflip.md).*
