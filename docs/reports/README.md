# G1 Walking — a hands-on learning series

Four short reports that take you from "what is this?" to running and tweaking your
own humanoid reinforcement-learning experiments — written for a reader with **no
robotics or machine-learning background**.

Read them in order:

1. **[00 — Primer](00-primer.md)** — the vocabulary with zero assumptions: what a
   simulator, policy, reward, episode, and PPO are, and why thousands of robots
   train in parallel.
2. **[01 — Watching it learn](01-watching-it-learn.md)** — the robot goes from
   random flailing to walking, shown frame by frame; how to read a reward curve.
3. **[02 — Reproducing the benchmark](02-reproducing-the-benchmark.md)** — we
   re-ran the training from scratch and got the same curve (evidence the result is
   real, not luck), and unpack what the robot is actually rewarded for.
4. **[03 — Turning the knobs](03-turning-the-knobs.md)** — a one-change experiment
   (slow vs. full speed range), the lesson that *higher reward ≠ better robot*, and
   a menu of settings you can change to explore further.

## Media

Plots and frame stills are committed under [`assets/`](assets). The full **videos**
(the learning progression and the A/B gait clips) are intentionally **not**
committed — they're large and git-ignored. They live on the training host under
`~/robotic-simulation/out/arc/` and are pulled to `assets/*.mp4` locally for
viewing. Each report shows the command to re-generate them.

## How these were produced

- The curve plots come from `scripts/plot_training_curves.py` (tensorboard → PNG).
- Full method and decisions: the design spec
  [`docs/superpowers/specs/2026-06-18-g1-walking-learning-arc-design.md`](../superpowers/specs/2026-06-18-g1-walking-learning-arc-design.md)
  and the implementation plan
  [`docs/superpowers/plans/2026-06-18-g1-walking-learning-arc.md`](../superpowers/plans/2026-06-18-g1-walking-learning-arc.md).
