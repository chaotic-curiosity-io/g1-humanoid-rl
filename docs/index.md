---
layout: default
title: Teaching a Humanoid to Walk (and Run, Flip, and Get Back Up)
---

# Teaching a Humanoid to Walk — and Run, Flip, and Get Back Up

How do you get a two-legged robot to *learn* to move — not by programming each
joint by hand, but by letting it practice millions of times in a simulator? This
is a hands-on, **zero-background** curriculum. No robotics or machine-learning
experience needed; every term is explained the first time it appears.

We train a [Unitree G1](https://www.unitree.com/g1) humanoid in a physics
simulator on a single GPU and watch what reinforcement learning actually does —
the successes *and* the instructive failures, every result confirmed on video,
not just by a number.

## New here? Start with walking

The four walking reports are the on-ramp — read them in order:

1. **[The Primer](reports/00-primer.md)** — the vocabulary with zero assumptions:
   simulator, policy, reward, episode, PPO, and why thousands of robots train in
   parallel.
2. **[Watching it Learn](reports/01-watching-it-learn.md)** — the robot goes from
   twitching on the ground to walking, frame by frame, plus how to read a reward curve.
3. **[Reproducing the Benchmark](reports/02-reproducing-the-benchmark.md)** — run
   training again from scratch and get the same curve (evidence it's real, not luck),
   and unpack exactly what the robot is rewarded for.
4. **[Turning the Knobs](reports/03-turning-the-knobs.md)** — change one setting,
   discover that *higher reward does not mean a better robot*, and get a menu of
   things to explore.

## Then explore the full curriculum

Once you have the vocabulary, the **[full syllabus](reports/)** branches into four
tracks — each a different way to teach a robot a skill, each with its own honest
story:

- **Locomotion** — shape a reward from scratch and the gait *emerges*. Beyond
  walking: [pushing for a running flight phase](reports/running-and-flight.md)
  (which reward-hacks into a dive!) and [more gaits](reports/more-gaits.md)
  (spin-in-place, backward walking).
- **Imitation** — give the robot a reference motion to match: a
  [cartwheel](reports/imitation-cartwheel.md) and a
  [backflip](reports/imitation-backflip.md) (landed in three attempts).
- **From-scratch tasks** — no reference, no command, behavior built entirely from
  the reward: [getting up off the floor](reports/getting-up.md) (a full stand,
  after four reward iterations).
- **Reward engineering** — the throughline of the whole series: the
  [reward-hacking gallery](reports/reward-hacking-gallery.md) (three ways a reward
  or metric misled us) and the [methods & techniques reference](reports/methods-reference.md).

The recurring lesson across all of it: **reinforcement learning is reward
engineering** — the robot does exactly what you reward, the metric can lie, and
the only real verdict is watching the robot move.

---

*A [Chaotic Curiosity](https://chaoticcuriosity.io) project. Built on
[mjlab](https://github.com/mujocolab/mjlab) (an Isaac-Lab-style RL API on
MuJoCo-Warp), trained on an NVIDIA DGX Spark. The full method, all the plots, and
the training runs behind these reports live in the
[repository](https://github.com/chaotic-curiosity-io/g1-humanoid-rl).*
