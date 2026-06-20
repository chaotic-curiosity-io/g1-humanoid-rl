# Getting Up: Four Tries to Teach a Robot to Stand From the Floor

*This report is the From-Scratch Tasks track's entry. If you are new here, read [00-primer.md](00-primer.md) first for the vocabulary — [policy](00-primer.md), [reward](00-primer.md), [episode](00-primer.md). This is the deepest report in the series, because it is the one where **the behaviour is built entirely out of the reward function** — and getting that reward right took four attempts, each of which failed in a different, instructive way.*

---

## What makes this different

Every other robot in this series starts each episode **standing**. Falling is the failure condition. The reward never had to *teach* standing — it only had to keep the robot from losing a standing start.

This report throws that away. The robot starts each episode **lying on the ground** — on its back, face-down, or on either side, at a random heading. There is **no velocity command** to follow and **no reference motion** to imitate (unlike the [cartwheel](imitation-cartwheel.md)). The only guidance is a reward that scores higher as the robot gets its body up and vertical. Everything — the squirm, the push, the sequence of limb movements — has to be *discovered*.

That makes get-up the purest test of reward design in the whole program. And purity cuts both ways: with nothing but a reward to lean on, the optimiser will find the **shortest path to the highest score**, and that path is almost never the one you pictured. This report is the honest log of four such paths — three of them wrong.

---

## First, build the task

This is the only spine task that needed **new code** before any training. `Mjlab-Recovery-Flat-Unitree-G1` didn't exist; it was written by cloning the flat velocity task and changing four things:

1. **The starting state.** A custom reset function drops the robot into one of four genuinely-fallen poses (supine, face-down, left side, right side) at a random heading, pelvis ~0.1 m off the floor. This is the one truly novel piece of code.
2. **The fall termination is removed.** The robot *starts* fallen — a "you fell over" termination would end every episode at step 1.
3. **The velocity command is zeroed** (there's nowhere to walk).
4. **The reward** rewards rising height and an upright torso instead of locomotion.

`★ The first lesson, before any training:` in a from-scratch task, the **starting-state distribution and the success definition are as load-bearing as the reward.** The very first build had a bug — half the "fallen" robots actually spawned *upright but low*, and trivially sprang to standing under no action at all. A zero-action physics probe caught it (the robot must *stay down* unless the policy lifts it), and the reset was fixed so all four poses are genuinely on the ground. A task that's accidentally trivial teaches the policy nothing.

The new task passed `ruff`/`pyright`/`pytest` and a no-NaN smoke-train before the real runs. Then came the four attempts.

---

## The four attempts

Each attempt below is the *same* task with a *different reward specification*. Watching the failure modes is the entire point.

### Attempt 1 — the stillness hack

**Reward:** an upright term, a Gaussian "height near standing" bonus, and — fatefully — the velocity-tracking rewards kept from the walking task with their command pinned to **zero** (intended as a "hold still once you're up" incentive).

**What happened:** reward climbed to ~62, but the success rate **peaked early (~33%) and then decayed toward zero.** The robot learned to **lie perfectly still** — because a motionless body has zero velocity, which the zero-command tracking rewards pay handsomely, and the Gaussian height bonus was ≈0 with *no gradient* down on the floor. Lying still out-scored the risky business of getting up, so the policy gave up on standing.

**Fix:** delete the velocity rewards (they reward stillness, which a fallen robot satisfies trivially), and make the height reward a **monotonic ramp** — non-zero gradient at *every* height, so there's always a signal to rise.

### Attempt 2 — the success-termination trap

**Reward:** stillness terms gone; monotonic height ramp in. Now the height reward was *real* (the robot got its pelvis up to ~0.5 m). But the success rate **still peaked (~50%) then decayed**, and episodes ran nearly full-length.

**What happened:** the task ended the episode the moment the robot stood up (`stood_up` was a success *termination*). Standing at step 100 therefore *threw away* ~900 steps of future height reward, while hovering in a half-crouch farmed that reward for the entire episode. The optimiser correctly learned to get *almost* up and **deliberately not finish**.

**Fix:** remove the success termination entirely. With only a timeout, the way to maximise reward is to stand **and stay standing** for the full 20 seconds.

### Attempt 3 — the stable-crouch local optimum

**Reward:** as Attempt 2, success-termination removed.

**What happened:** genuine progress — the robot now **recovers from the floor to a stable, upright deep squat and holds it** (torso vertical, `upright` reward ~0.95). But it stalls at a pelvis height of ~0.52 m, short of the ~0.76 m full stand:

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/s3_getup_still.png">
  <source src="assets/s3_crouch_side.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/s3_crouch_side.mp4">download the clip</a> instead.
</video>

A deep, wide squat is a **stable, low-effort equilibrium**: low centre of mass, wide base, almost no ongoing action needed. Rising the last 0.24 m demands constant active balance for only a little more height reward — not worth it, so the policy settled.

**Fix:** tilt the trade-off. Double the height weight (5 → 10) so the last bit of standing is clearly worth it, and cut the action-rate penalty (−0.1 → −0.03) so the balancing effort is cheap.

### Attempt 4 — a full stand

It worked. The height reward climbed steadily to **9.7 out of a maximum 10** (pelvis ≈ 0.74 m, target 0.76) and — unlike every previous attempt — **held there with no decay**:

![stand-up height reward climbing to 9.7 and holding](assets/s3_standup.png)

The robot now recovers from a fallen pose all the way to a **full upright stand**, and stays there:

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/s3_getup_still.png">
  <source src="assets/s3_getup_side.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/s3_getup_side.mp4">download the clip</a> instead.
</video>

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/s3_getup_still.png">
  <source src="assets/s3_getup_chase.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/s3_getup_chase.mp4">download the clip</a> instead.
</video>

The overall reward shows the healthy shape of a hard problem being solved: a dip into the negatives early (exploration, dominated by penalties) before the climb to a high plateau.

![mean reward S-curve, dip then climb to ~198](assets/s3_reward.png)

---

## The command

The final (Attempt 4) training run, after the task was built and deployed into the container:

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Recovery-Flat-Unitree-G1 \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 2500 \
  --agent.run-name recovery-v4'"
```

The new task code is deployed in the container at `mjlab/src/mjlab/tasks/recovery/` (and mirrored in this repo's working notes for reproducibility). Because the mjlab tree is root-owned on the Spark, files are written to `/tmp` and `docker cp`'d into the container — never `sudo chown`'d. The recovery-specific reward weights live in `config/g1/env_cfgs.py`; the lying-down reset is in `mdp/events.py`.

---

## The lesson

Four attempts, four behaviours, one task. None of the failures was a bug in the learning algorithm — in each case PPO found the highest-scoring behaviour available, exactly as designed. The failures were all in the **specification**:

| Attempt | The misspecification | What the robot did |
|---|---|---|
| 1 | zero-command velocity rewards pay for *stillness* | lay still |
| 2 | success *ends the episode*, cutting off reward | half-crouched forever |
| 3 | a stable crouch is cheap; full standing isn't worth it | stood, but only halfway |
| 4 | — | **got up** |

Every one of these is a [reward hack](reward-hacking-gallery.md) — the optimiser exploiting a gap between what you measured and what you meant. Get-up makes them unavoidable precisely because there's no reference motion to anchor the behaviour: the reward, the starting state, and the success definition are *all* you have, and *all* of them have to be right.

The practical takeaway is not "RL is fiddly." It's that **designing a from-scratch objective is an iterative, empirical craft** — you write a reward, watch what the optimiser does with it (on video, never on the number alone), and the gap between intent and behaviour tells you what to fix next. Four iterations to a get-up is not failure; it *is* the method.

---

## Tweak this to explore

**Re-run the four attempts.** The four reward configs are the cleanest reward-design lesson in the repo. Watch the gait degrade and recover as you toggle the velocity rewards, the success termination, the height weight, and the action-rate penalty one at a time.

**Harder falls.** Widen the reset randomisation — more extreme crumples, harder initial joint scatter — for a policy robust to a wider variety of falls. Costs more training.

**Stricter standing.** The height ramp tops out at 0.76 m. Add an explicit "hold still once up" reward *gated on already being upright* (the version that does NOT pay a fallen robot) to get a calmer final stand.

**The whole task is a reward-hacking gallery.** Every attempt above is a specimen — see [reward-hacking-gallery.md](reward-hacking-gallery.md) for the cross-program collection. The freshest, clearest hack explanations come from writing them down the moment you see them, which is exactly what this report did.

---

*Unitree G1 on flat terrain, MuJoCo-Warp on a DGX Spark (NVIDIA GB10, aarch64). A new `Mjlab-Recovery-Flat-Unitree-G1` task, four reward iterations, 2048 parallel robots, up to 2500 iterations. The spec for this run: [2026-06-19-spine-getup-recovery.md](../superpowers/specs/2026-06-19-spine-getup-recovery.md).*
