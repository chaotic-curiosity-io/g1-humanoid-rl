# G1 Humanoid RL — Learning Series Syllabus

A growing collection of short reports that take a reader with **no robotics or
machine-learning background** from "what is this?" to running, imitating, and
engineering their own humanoid RL experiments. Every term is defined on first use;
each report is standalone and re-runnable from this checkout + the Spark.

The series is organized around a **Foundations primer** (shared vocabulary every
track builds on) and **four paradigm tracks** — the four ways this repo teaches a
robot to do something:

- **Locomotion** — shape a reward from scratch; the gait is emergent.
- **Imitation** — track a reference motion each step.
- **From-scratch tasks** — no reference, no command; behavior is 100% the reward
  terms you write.
- **Reward engineering** (cross-cutting) — understand and control what the reward
  actually incentivizes.

---

## Foundations

The keystone vocabulary every track builds on. Start here.

| Report | Hook |
|---|---|
| [00 — Primer](00-primer.md) | What a simulator, policy, reward, episode, and PPO are — and why thousands of robots train in parallel. |

---

## Locomotion

Shape a velocity reward from scratch; the gait emerges from the numbers.

| Report | Hook |
|---|---|
| [01 — Watching it learn](01-watching-it-learn.md) | The robot goes from random flailing to walking, shown frame by frame; how to read a reward curve. |
| [02 — Reproducing the benchmark](02-reproducing-the-benchmark.md) | We re-ran training from scratch and got the same curve — evidence the result is real, not luck — and unpack what the robot is actually rewarded for. |
| [03 — Turning the knobs](03-turning-the-knobs.md) | A one-change experiment (slow vs. full speed range), the lesson that *higher reward ≠ better robot*, and a menu of settings you can explore. |
| [Running and flight phase](running-and-flight.md) *(draft — populated after the S1 run)* | Push the velocity range until the robot develops a true flight phase; measure the gait change. |

---

## Imitation

Give the robot a reference motion to match each step, and watch it learn the
move from frame-level feedback.

| Report | Hook |
|---|---|
| [Imitation — cartwheel](imitation-cartwheel.md) *(draft — populated after the cartwheel narrative is finalized)* | A narrative entry point to the cartwheel training; links the full iteration log in [`../cartwheel-journey.md`](../cartwheel-journey.md). |
| [Imitation — backflip](imitation-backflip.md) *(draft — populated after the S2 run)* | Retarget a single feasible backflip reference onto the G1 and train a policy to complete it — applying every lesson from the cartwheel. |

---

## From-scratch tasks

No reference motion, no velocity command: the robot's behavior is entirely
determined by the reward terms you write.

| Report | Hook |
|---|---|
| [Getting up](getting-up.md) *(draft — populated after the S3 run)* | Train a policy to stand up from randomized fallen poses using only a shaped reward — the report most likely to surface a new reward hack. |

---

## Reward engineering

Understand and deliberately control what a reward actually incentivizes.

| Report | Hook |
|---|---|
| [Reward-hacking gallery](reward-hacking-gallery.md) *(draft — populated after the S4 run)* | A curated set of deliberately-naive rewards, each showing the cheat the robot found and the fix; the capstone lesson that "reward design *is* the job." |
| [Methods and techniques reference](methods-reference.md) | One lookup surface for the full toolkit: reward terms, terminations, the retargeting pipeline, curricula, recording gotchas, helper scripts, and ops/safety. |

---

## Ready-to-run (specs, not yet trained)

These candidate tasks have complete specs — enough to run without re-designing —
but have not yet been trained. They are grouped by tier and can be promoted to
full training campaigns as follow-ons.

| Tier | Bundle | What's inside |
|---|---|---|
| Tier 1 — Gait tweaks | [Compact spec](../superpowers/specs/2026-06-19-tier1-gait-tweaks-compact.md) | Crouched "Groucho" walk, tiptoe, energy-efficiency, spin-in-place, prescribed gait (hop/march), backward/sideways. All on the existing velocity env; no new mjlab code. |
| Tier 2 — Acrobatics | [Compact spec](../superpowers/specs/2026-06-19-tier2-acrobatics-compact.md) | Spinkick/martial-arts, jump, dance. All via the existing tracking env and retargeting pipeline; no new code. |
| Tier 3 — From-scratch tasks | [Compact spec](../superpowers/specs/2026-06-19-tier3-tasks-compact.md) | Push-recovery, single-leg (flamingo) balance. Both require a new mjlab task + reward manager. |
| Tier 4 — Object / whole-body (spec-only) | [Compact spec](../superpowers/specs/2026-06-19-tier4-objects-compact.md) | Reach-to-target, kick-a-ball, carry. Highest research risk; spec-only in this program — MJCF scene edits + new rewards required. |

---

## Media

Plots, frame stills, **and the videos** are committed under [`assets/`](assets) and
embedded directly in the reports: the learning progression (chase / side / 2×2 grid)
in report 01, and the A/B gait clips in report 03. The repo's `.gitignore` blanket-
ignores `**/*.mp4` but explicitly re-includes `docs/reports/assets/*.mp4` so these
publish to the Pages site while training-run videos elsewhere stay out of git. The
source clips live on the training host under `~/robotic-simulation/out/arc/`; each
report shows the command to re-generate them.

## How these were produced

- The curve plots come from `scripts/plot_training_curves.py` (tensorboard → PNG).
- Full method and decisions for the walking arc: the design spec
  [`2026-06-18-g1-walking-learning-arc-design.md`](https://github.com/chaotic-curiosity-io/g1-humanoid-rl/blob/main/docs/superpowers/specs/2026-06-18-g1-walking-learning-arc-design.md)
  and the implementation plan
  [`2026-06-18-g1-walking-learning-arc.md`](https://github.com/chaotic-curiosity-io/g1-humanoid-rl/blob/main/docs/superpowers/plans/2026-06-18-g1-walking-learning-arc.md).
- The skills curriculum extension (this syllabus and all new reports) follows the
  master spec
  [`2026-06-19-g1-skills-curriculum-master.md`](https://github.com/chaotic-curiosity-io/g1-humanoid-rl/blob/main/docs/superpowers/specs/2026-06-19-g1-skills-curriculum-master.md).
