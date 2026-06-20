# Imitation Learning: Teaching G1 to Cartwheel

> **Draft.** This report is populated after the cartwheel campaign is fully written up in report form (see [the spec](../superpowers/specs/2026-06-19-spine-backflip.md) for the Imitation track plan). The structure below is the skeleton it will fill. The detailed iteration-by-iteration log of the cartwheel campaign already exists as a separate document — linked prominently below.

*This report is the Imitation track's entry point. It introduces motion imitation as a training method and summarizes the cartwheel campaign. The backflip — the Imitation track's spine task — is covered in the next report: [imitation-backflip.md](imitation-backflip.md).*

---

## What you are about to see

The reports so far ([00](00-primer.md)–[03](03-turning-the-knobs.md)) trained the robot by writing down what we wanted in terms of numbers: "move forward at this speed," "stay upright," "don't fall." The algorithm then found a policy that maximized those numbers. We designed the objective; the robot found its own way of meeting it.

**Motion imitation is a different approach.** Instead of specifying what the robot should *achieve*, we show it what we want it to *look like* — frame by frame. We take a recording of a human cartwheel, mathematically translate it onto the G1's body (a process called "retargeting"), and then reward the robot for matching that recording as closely as possible at each step. The behavior that emerges is not "whatever scores highest on speed plus uprightness" — it is, if training works, something that actually resembles the reference motion.

This is more powerful than velocity tracking for expressive behaviors, and significantly harder to get right.

---

## The cartwheel campaign: a summary

The G1 cartwheel was trained using the `Mjlab-Tracking-Flat-Unitree-G1` task. The retargeted reference was a 2.73-second MimicKit cartwheel clip, re-expressed in terms of the G1's joints and proportions.

It took three iterations — labeled iterA, iterB, and iterC — to produce a policy that performs a genuine, visually confirmed cartwheel. Each iteration taught a hard-won lesson:

**IterA** failed because the simulation cut every attempt short. The tracking environment terminates an episode the moment the robot's pose drifts more than 0.25 m from the reference. During a cartwheel, the arms and legs sweep through wide arcs — staying within 0.25 m of the reference is almost impossible mid-flip. The policy never got to experience a completed cartwheel, so it never learned how to finish one.

**IterB** appeared to succeed — an automated scorer reported a 95% completion rate. It was wrong. Frame-by-frame inspection of the rendered video revealed the robot was face-planting and rolling along the ground, not cartwheeling. The scorer was counting any episode where the pelvis rolled through 180 degrees as an "inversion" — which a crash-roll also satisfies. Additionally, the render was using the old tight thresholds, so every attempt was cut short on screen before the face-plant even landed. And the reference motion was accidentally two cartwheels long (a tooling bug), making the target infeasible from the start.

**IterC** fixed all three issues: a clean, single-cartwheel reference; 0.5 m termination thresholds throughout training and rendering; and training from scratch (no warm-start from iterB's stuck policy). By iteration 4,500, the policy was already performing real, visually confirmed cartwheels with proper inversion and two-footed landings.

The final policy (`model_19999.pt` from iterC) performs at least four full cartwheels in a 40-second continuous rollout, each with a proper airborne inversion and recovery.

---

## The deep log

The cartwheel journey is documented in full — iteration by iteration, bug by bug, discovery by discovery — in a separate narrative:

**[cartwheel-journey.md](../cartwheel-journey.md)**

That document is the canonical record of what happened, including the scorer bug, the double-cartwheel reference, the render-vs-training mismatch, and the mid-run visual checks that confirmed iterC was working before training finished. If you want to understand *why* each iteration went the way it did, that is where to go.

This report, once populated, will be the reader-friendly version: the polished account, with the clips embedded, written for someone who has never seen motion imitation before.

---

## The command

The final training command for iterC (the successful run), executed inside the `mjlab-dev` container on the Spark:

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file /workspace/pose-pipeline/motions/mimickit_cartwheel_single.npz \
  --env.terminations.anchor-pos.threshold 0.5 \
  --env.terminations.ee-body-pos.threshold 0.5 \
  --agent.num-envs 4096 \
  --agent.max-iterations 20000 \
  --agent.seed 42'"
```

The two critical flags are the threshold overrides. The default 0.25 m is too tight for an aerial motion; 0.5 m gave the policy enough tolerance to experience and learn from a full rotation. This was the single most important change between iterA and iterC.

To render the final policy with terminations disabled (so the robot runs continuously without resetting mid-flip):

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Tracking-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_tracking/2026-04-19_20-54-47_cartwheel-iterC-single/model_19999.pt \
  --disable-terminations \
  --no-shadows --no-reflections --no-debug-viz \
  --cameras chase side front top grid \
  --output /workspace/clips/cartwheel_final_{camera}.mp4'"
```

---

## Results

*This section will be populated when the cartwheel campaign is written up as a full report. The clips already exist on the Spark (see [cartwheel-journey.md](../cartwheel-journey.md) for links to the Google Drive folder). The placeholders below describe what will be embedded here.*

### The final cartwheel: iterC

**[Placeholder: final multi-camera clip grid — `assets/cartwheel_final_grid.mp4`]**

*Four camera angles (chase, side, front, top), 40-second continuous rollout with terminations disabled. The final policy performs at least four full cartwheels in this window, each with a genuine airborne inversion and a two-footed landing.*

### What the failed iterations looked like

**[Placeholder: iterB clip for contrast — `assets/cartwheel_iterB_fail.mp4`]**

*IterB at model_25000: the policy the scorer called "95% success." Frame-by-frame review showed a crash-roll — the robot faces-plants and tumbles — not a cartwheel. This clip is the most direct illustration of why visual review is always the primary gate and numerical scores are only a first-pass filter.*

### The three key lessons

**[Placeholder: prose writeup — written once the full report form is complete.]**

The three lessons from the cartwheel campaign that apply to every future imitation run (and that the backflip report applies from the start):

1. **Termination thresholds are a two-edged knife.** Tight thresholds prevent learning by cutting every attempt short. Loose thresholds risk giving the policy a cheap escape: it learns to hit the reference loosely enough to avoid termination, but not accurately enough to produce the real motion.

2. **Quantitative scorers lie.** A scorer that measures a proxy for the behavior (roll angle through 180°) can be fooled by a completely different behavior (a crash-roll). Always pair any numerical metric with frame-by-frame visual review.

3. **The reference must be feasible and clean.** A double-cartwheel reference (the tooling bug that produced a 6-second clip instead of 2.73 seconds) asks the policy to do more than it can. A single, clean, correctly-timed reference unblocked the training more than any reward or threshold tweak.

---

## Tweak this to explore

**Try a different motion.** The same `Mjlab-Tracking-Flat-Unitree-G1` task and pipeline can run any retargeted motion. The backflip is the next step — covered in [imitation-backflip.md](imitation-backflip.md). A jump, a spin kick, or a dance move would use the same recipe.

**Adjust the termination threshold.** The 0.5 m threshold that worked for the cartwheel may be too tight or too loose for a different motion. A slower motion (a dance step) might work well at 0.3 m. A faster aerial motion (a flip) might need the threshold raised further, or terminations disabled entirely during training.

**Render at different thresholds to understand the policy.** After training, render the same checkpoint at `--termination-threshold 0.25` and at `--disable-terminations`. The difference between these two renders tells you how accurately the policy matches the reference in the difficult mid-motion frames — and whether what you are seeing is the policy's real behavior or an artifact of early-cut episodes.

---

*All experiments use the Unitree G1 on flat terrain, trained with the MuJoCo-Warp simulator on a DGX Spark (NVIDIA GB10, aarch64). The full iteration log: [cartwheel-journey.md](../cartwheel-journey.md).*
