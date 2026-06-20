---
layout: default
title: Teaching a Humanoid to Walk
---

# Teaching a Humanoid to Walk

How do you get a two-legged robot to *learn* to walk — not by programming each
joint by hand, but by letting it practice millions of times in a simulator? This
is a four-part, **zero-background** guide. No robotics or machine-learning
experience needed; every term is explained the first time it appears.

We train a [Unitree G1](https://www.unitree.com/g1) humanoid in a physics
simulator on a single GPU, watch it go from random flailing to a confident walk,
reproduce the result from scratch to prove it's real, and then change one setting
to see how the robot's behavior shifts.

## Start here

1. **[The Primer](reports/00-primer.md)** — the vocabulary with zero assumptions:
   what a simulator, policy, reward, episode, and PPO are, and why thousands of
   robots train in parallel.
2. **[Watching it Learn](reports/01-watching-it-learn.md)** — the robot goes from
   twitching on the ground to walking, shown frame by frame, plus how to read a
   reward curve.
3. **[Reproducing the Benchmark](reports/02-reproducing-the-benchmark.md)** — we
   run the training again from scratch and get the same curve (evidence it's real,
   not luck), and unpack exactly what the robot is rewarded for.
4. **[Turning the Knobs](reports/03-turning-the-knobs.md)** — change one setting
   (how fast the robot is allowed to practice), discover that *higher reward does
   not mean a better robot*, and get a menu of things you can change to explore
   further.

---

*Built on [mjlab](https://github.com/mujocolab/mjlab) (an Isaac-Lab-style RL API
on MuJoCo-Warp), trained on an NVIDIA DGX Spark. The full method, all the plots,
and the training runs behind these reports live in the
[repository](https://github.com/chaotic-curiosity-io/g1-humanoid-rl).*
