# G1 Humanoid RL — Learning Series

A fifteen-chapter curriculum that takes a reader with **no robotics or
machine-learning background** from "what is a simulator?" to a humanoid robot
teaching itself to do a cartwheel and a backflip. Every term is defined the
first time it appears. **Read the chapters in order** — each one builds on the
concepts the previous one introduced, and the series is designed as one
cohesive arc, not a collection of standalone reports.

---

## Part I — What is this even?

The first two chapters answer the questions a newcomer brings before anything
else can land: what kind of project is this, and what are the eight words I
need to know before I read further?

| Chapter | Hook |
|---|---|
| [Chapter 1 — The Big Picture](01-the-big-picture.md) | A human-shaped robot taught itself to walk inside a computer, with no rules about how to move — here is why that is interesting and how it is possible. |
| [Chapter 2 — The Vocabulary](02-the-vocabulary.md) | The eight precise terms — environment, observation, action, policy, reward, episode, rollout, return — that let us talk about the practice loop without hand-waving. |

---

## Part II — How learning actually works

Three chapters on the engine under the hood: why trial and error can produce a
skill from nothing, how PPO turns that into a practical algorithm, and how to
read the numbers a training run produces.

| Chapter | Hook |
|---|---|
| [Chapter 3 — Trial and Error Made Precise](03-trial-and-error.md) | The single idea that turns thousands of falling robots into a walking one: nudge toward actions that earned more than expected, away from those that didn't — and repeat. |
| [Chapter 4 — PPO for Novices](04-ppo-for-novices.md) | The algorithm this project actually trains with: why a raw policy-gradient step over-corrects, how the value function gives it a baseline, and how clipping keeps the updates stable. |
| [Chapter 5 — Reading the Training](05-reading-the-training.md) | How to pick up a reward curve you have never seen before and answer the two questions that matter: is learning healthy, and is the number going up actually measuring what you think? |

---

## Part III — First real skill: walking

Three chapters on the concrete experiment: watch the robot go from flailing to
walking, verify the result is reproducible, then run your first controlled
experiment by pulling one knob.

| Chapter | Hook |
|---|---|
| [Chapter 6 — Watching It Walk](06-watching-it-walk.md) | The walking training run replayed frame by frame — from random collapse to smooth velocity tracking — and what the task actually asked the robot to do. |
| [Chapter 7 — Proving It's Real](07-proving-its-real.md) | Run the same training from scratch, get the same curve, then decompose the reward into its component terms to see what each one contributed: the reproducibility argument. |
| [Chapter 8 — Turning the Knobs](08-turning-the-knobs.md) | A one-change experiment (slow vs. full speed range) and the lesson that higher reward does not mean a better robot — plus a controlled-experiment discipline you will use for the rest of the series. |

---

## Part IV — Reward engineering

Three chapters that show, with live failures from this project, what happens
when the reward measures a proxy for the behavior you want rather than the
behavior itself.

| Chapter | Hook |
|---|---|
| [Chapter 9 — The Running Dive](09-the-running-dive.md) | Rewarding "air time" to force a flight phase backfires: the highest-scoring policy is a forward dive — the reward was telling the truth, but not the truth we wanted. |
| [Chapter 10 — More Gaits and the Command System](10-more-gaits-and-commands.md) | The robot's velocity command is a three-number vector, not just "go forward" — spin-in-place works on the first try; backward walking does not; the gap between them reveals a real limit. |
| [Chapter 11 — The Reward-Hacking Gallery](11-the-reward-hacking-gallery.md) | Three real failure modes from this project — proxy gaming (the dive), silent compensation (no-upright), and metric lying (the cartwheel scorer) — and why "the reward misled us" has more than one flavor. |

---

## Part V — Imitation

Two chapters on a completely different training paradigm: instead of writing
reward terms, show the robot a recording and reward it for matching each frame.

| Chapter | Hook |
|---|---|
| [Chapter 12 — Imitation and the Cartwheel](12-imitation-and-the-cartwheel.md) | Show the robot a real cartwheel, reward it for matching each frame, and watch it learn the move — plus the three hard-won lessons about termination thresholds and the scorer that lied. |
| [Chapter 13 — The Backflip in Three Attempts](13-the-backflip-in-three-attempts.md) | Three training runs, each fixing one diagnosed deficit: too-tight thresholds (grounded), too-loose thresholds (airborne but lands on its back), then a targeted landing reward that finally sticks the feet. |

---

## Part VI — Building a task from nothing

One chapter on the hardest paradigm: no reference motion, no velocity command,
just a reward you design from scratch for a behavior the robot has never seen.

| Chapter | Hook |
|---|---|
| [Chapter 14 — Building Get-Up from Scratch](14-building-get-up-from-scratch.md) | Four reward iterations — fixing a stillness hack, then a success-termination trap, then a stable-crouch local optimum, then finally a full stand — to teach the robot to stand up from any fallen pose. |

---

## Part VII — Synthesis

The capstone chapter names the single principle every previous chapter was
circling, and shows why it is a discipline, not a trick.

| Chapter | Hook |
|---|---|
| [Chapter 15 — Reward Engineering as Craft](15-reward-engineering-as-craft.md) | Strip fourteen chapters down to one sentence: reinforcement learning is reward engineering — the robot does exactly what you reward, not what you meant — and that sentence is the whole job. |

---

## Appendices

Reference material and deep logs for readers who want to go further.

| | |
|---|---|
| [Methods and techniques reference](methods-reference.md) | One lookup surface for the full toolkit: reward terms, terminations, the retargeting pipeline, curricula, recording gotchas, helper scripts, and ops notes. |
| [Cartwheel journey](../cartwheel-journey.md) | The full iteration-by-iteration log of the cartwheel training campaign — every attempt, every failure mode, every fix — for readers who want the unabridged story behind Chapter 12. |

---

## Media

Plots, frame stills, and videos are embedded throughout the chapters and
committed under [`assets/`](assets). The repo's `.gitignore` blanket-ignores
`**/*.mp4` but explicitly re-includes `docs/reports/assets/*.mp4` so these
publish to the Pages site while training-run videos elsewhere stay out of git.
The source clips live on the training host under
`~/robotic-simulation/out/arc/`; each chapter that embeds a clip shows the
command to re-generate it. Curve plots come from
`scripts/plot_training_curves.py` (tensorboard scalars → PNG).
