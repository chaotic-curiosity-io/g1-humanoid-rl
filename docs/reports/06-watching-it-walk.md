# Chapter 06 — Watching It Walk

*Chapter 05 gave you the tools to read a reward curve — the flat floor, the "aha" climb, the steady refinement. You saw the numbers. Now watch the body. This chapter replays the same training run frame by frame: you will see the robot go from collapsing in under a second to something that, by any reasonable standard, walks. Along the way you will learn what the task actually asked the robot to do, why the gait looks the way it does, and why nobody programmed a single step.*

---

## Part III: The first real skill

Parts I and II built the conceptual scaffold: what a simulator is, what a policy is, what PPO does, how to read a reward curve. Part III puts that scaffold to use on a concrete skill. This chapter is the visual story of the walking run. Chapter 07 stress-tests it. Chapter 08 starts pulling knobs.

The skill is **walking** — or more precisely, tracking a commanded velocity. The result is not a robot that has been told where to put its feet. It is a robot that discovered how to move its feet by trial and error, guided only by a reward signal designed to push it in the right direction. Understanding the difference between those two things is the chapter's deepest goal.

---

## The task: a command, not a choreography

Before watching the robot, understand what it was asked to do — because it was not asked to walk.

### "Go this fast, in this direction"

The task is called a **velocity tracking task** (or command tracking task). At the start of every episode, the simulator hands the policy a target: a forward speed, a sideways speed, and a turning rate. In the baseline walking run, the forward command was set to `1.0 m/s` — roughly a brisk human walking pace — with no sideways motion and no turning. The task is simple to describe: match that command as closely as possible, for as long as possible.

That is all the robot is asked to do. Not "take steps." Not "swing your arms." Not "keep your torso upright." Just: *make your body move at roughly 1 m/s forward.*

### Why is this not choreography?

In a choreographed system, an engineer writes out what every joint does at every moment. Step forward with the left foot, shift weight, lift the right foot, swing it through, land. A policy is *not* that. There is no sequence of moves written anywhere in the code. The policy is a neural network — a function with about 200,000 adjustable numbers — that maps the robot's current sensor readings to the current joint targets. Those joint targets are discovered by trial and error. The engineer designs the reward; the optimizer finds the behavior that earns it. The steps, the arm swings, the rhythm — none of that was written in. It emerged.

This distinction will matter a great deal in later chapters, when the gait does something the designer did not intend. For now, hold it as orientation: **the task specifies a goal, not a motion.**

---

## The reward signal: terms and weights

The policy can only learn to walk if the reward signal guides it toward walking. A single number per timestep — "you earned 0.3 this step" — has to encode all of that guidance. In practice, the total reward is not a single monolithic score. It is the sum of several **reward terms**, each targeting a specific aspect of behavior, each scaled by a **reward weight** that controls how loudly that term speaks.

Think of each reward term as one voice in a choir, and the reward weight as that voice's volume knob.

For the baseline walking run, the key terms are:

**`track_lin_vel`** — the primary term. At each timestep, compute the gap between the commanded forward speed and the robot's actual forward speed. The smaller the gap, the higher this term. Weight: high. This is the "please go forward at 1 m/s" signal.

**`upright`** — rewards the robot for keeping its torso close to vertical. If the torso tips past a threshold, this term drops. Weight: moderate. This is the "please do not face-plant" signal.

**`air_time`** — rewards each foot for spending time off the ground each stride. Weight: relatively low. This term nudges the robot toward actual steps rather than a shuffling slide. It is a small voice, but an important one: without it, the easiest way to track a forward velocity is to slide without lifting the feet.

**`action_rate_l2`** — penalizes large changes in joint targets from one timestep to the next. Weight: small negative (a penalty). This is the "please be smooth" signal — it discourages wild, jerky motions that happen to work once but are fragile.

The total reward at each timestep is the weighted sum of all these terms:

```
total_reward = w_track × track_lin_vel
             + w_upright × upright
             + w_air_time × air_time
             − w_rate × action_rate_l2
             + (other smaller terms...)
```

Every symbol: `w_*` is a weight (a number the designer chose), each named term is a score computed from the robot's state that timestep, and the sum is what the policy is trying to maximize over the episode.

> **Insight: the designer shapes the gait by choosing weights, not moves.**
> The designer never said "take a step every 0.6 seconds." They said "I will pay you for going forward, pay you a little for lifting your feet, pay you for staying upright, and charge you for being jerky." The optimizer found the best behavior under those constraints. That behavior is a gait. The weights are the design knobs — and as you will see in Chapter 08, changing a weight changes the gait, sometimes dramatically.

### What the robot does not know about the target gait

Nothing in the reward signal specifies arm swing, step length, step frequency, or the specific rhythm of a human walk. Those properties emerge because they happen to score well under the terms above. The robot swings its arms because that helps it stay upright and smooth. The robot takes steps of roughly the right size because that is the most reward-efficient way to earn `air_time` while tracking the speed. The designer did not choose these properties; the optimizer found them.

---

## Watching the progression

Now watch.

The baseline run trained for **2,050 iterations**, saving a snapshot of the policy every 50 iterations — 42 snapshots in total. The videos below replay 24 of those snapshots in order. Each clip is a time-lapse of learning: you see the robot at equally-spaced checkpoints from early training to convergence.

### Chase camera — the clearest view of heading

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/still_final.png">
  <source src="assets/chase.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/chase.mp4">download the clip</a> instead.
</video>

*24 checkpoints from the baseline walking run, replayed in training order. Chase camera follows from behind. The full arc — collapse, discovery, confident walk — is visible in one clip.*

---

### Side camera — posture and gait asymmetry

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/still_final.png">
  <source src="assets/side.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/side.mp4">download the clip</a> instead.
</video>

*Same 24 checkpoints from a side angle. The lateral view makes posture and weight shift easier to judge — watch the torso tilt change as training progresses.*

---

### 2×2 grid — four eras at once

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/still_mid.png">
  <source src="assets/grid.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/grid.mp4">download the clip</a> instead.
</video>

*Four checkpoints from different stages of training shown simultaneously. The contrast between top-left (early) and bottom-right (late) makes the scale of the change obvious at a glance.*

---

## Three stills, three stages

The clips above move fast. These stills freeze three specific moments from the same run and let you study them.

### Stage 1 — random twitching (early training)

![The G1 robot at early training, collapsed on the ground](assets/still_early.png)

At the very start of training, the policy's weights have never been adjusted — they begin as random numbers. Every joint target the network outputs is essentially random. The robot does not walk. It does not even stand. It collapses within a second, motors firing with no coordination, and the episode ends almost immediately. The robot earns almost no reward before the reset because there is no time to earn any.

This is not a malfunction. It is the starting line for every skill trained from scratch. There is nothing else here at iteration 0 — only noise.

---

### Stage 2 — discovering a gait (mid-training)

![The G1 robot mid-training, partially upright and stepping awkwardly](assets/still_mid.png)

Somewhere around iteration 500–800 the picture changes. The robot is upright. It is moving in roughly the right direction. The posture is hunched, the arms are stiff and odd-looking, the steps are irregular — but it is *moving*, and it is not falling within a second.

What happened? Two things came together. First, the policy crossed a threshold of balance: it learned just enough about not-falling to keep the episode alive long enough to earn `track_lin_vel` reward. Second, once the episode was alive, PPO got a much richer signal — dozens of timesteps of feedback instead of single-digit ones. That richer signal let it take better-aimed gradient steps, which let it survive slightly longer, which gave it even richer signal. The feedback loop from Chapter 05 kicked in.

The gait at this stage is a local solution, not a refined one. The optimizer found a strategy that scores well under the current reward terms, but has not yet found a *better* strategy. The hunched, stiff walk earns reward. A smoother walk would earn more. That refinement is still coming.

---

### Stage 3 — a confident walk (converged)

![The G1 robot at iteration 2050, walking forward confidently](assets/still_final.png)

By iteration 2,050, the picture has transformed. The robot walks with its torso upright, its weight shifting smoothly from foot to foot, its arms swinging in approximate counterbalance to its legs. It tracks the 1 m/s command reliably. Mean reward: **50.5**. Mean episode length: **995 out of 1,000 steps** — the robot is running out the full episode almost every time.

No engineer wrote any of this. The arm swing is there because it turned out to be the optimizer's best answer for satisfying `upright` while also satisfying `action_rate_l2` (a smooth, counterbalancing arm swing is lower-jerk than no arm swing at all). The step frequency settled where it did because that is the pace that best satisfies `air_time` and `track_lin_vel` together. The torso angle found its equilibrium because `upright` penalizes deviation.

This is **emergent gait**: a walking pattern that arose from optimizing reward terms, rather than being explicitly choreographed. No step was scripted. The designer wrote incentives; the optimizer wrote the motion.

---

## Why this counts as an "emergent" gait — and why that matters

The word *emergent* is doing real work here, not just dressing up a mundane result.

In classical robotics, a bipedal walker requires an explicit controller: a designer computes joint trajectories, specifies contact timing, solves for balance. The behavior is the direct product of design effort applied to motion. Change the desired behavior and you redesign the controller.

In the RL approach here, **the designer specified a goal, and the behavior appeared**. The three reward terms above did not fully determine the motion — they constrained the space of possible motions, and the optimizer found one that scored well within that constraint. A different optimizer, a different random seed, or even slightly different starting weights might have found a different gait that scored equally well. The outcome is a function of the incentives and the search process together, not a one-to-one product of design.

This has two practical consequences you will run into throughout this series:

**It makes the designer's job harder in one way.** You cannot directly specify what you want the robot to do. You have to specify what you want to *reward*, and then verify that the optimizer found something that satisfies your actual intent and not just your numerical proxy. Chapter 07 introduces how to do that verification. Chapter 08 shows how changing a weight changes the emergent gait. Later chapters show cases where the emergent gait was not the intended gait at all.

**It makes the system more powerful in another way.** The optimizer is not constrained to motions a human engineer could design. It searches a space of ~200,000-weight configurations that no human could explore analytically. The arm swing that emerged in this run was not designed; it was found, and it works. The series ahead includes a cartwheel that emerged from a reward signal. No engineer could have choreographed it step by step — but the optimizer found it.

> **Insight: emergence is not magic, it is search.**
> The gait did not spontaneously appear. PPO searched ~2,050 iterations × 2,048 parallel episodes worth of experience — millions of rollouts — evaluating and discarding motions that scored poorly and reinforcing ones that scored well. The result looks creative because the search space is large and the outcome was not prespecified. But the mechanism is gradient descent, and the signal is the sum of the reward terms. Understanding the terms is understanding where the gait came from.

---

## What the episode ending tells you

Occasionally in the clips above — especially early in training — you will see the robot *not* fall in any obvious way, but the episode ends anyway. The simulation cut it short. In the baseline run, this happens when the robot's body drops below a height threshold, indicating it has effectively collapsed even if not dramatically. The exact mechanics of when and why episodes end early will be the subject of Chapter 07. For now: an episode that ends before the time limit is one where the robot did something the environment decided was a failure. Watch for episodes getting *longer* as training progresses — that is the episode-length metric from Chapter 05 rising in real time.

---

## Reproducing the progression

The clips above were produced by the `record_learning_progression.py` script, run inside the `mjlab-dev` container on the Spark:

```bash
MUJOCO_GL=egl python -m mjlab.scripts.record_learning_progression \
  Mjlab-Velocity-Flat-Unitree-G1 \
  --run-dir logs/rsl_rl/g1_velocity/2026-04-17_18-46-23 \
  --num-checkpoints 24 \
  --cameras chase,side \
  --command-lin-vel-x 1.0
```

Note: the task ID (`Mjlab-Velocity-Flat-Unitree-G1`) must be supplied explicitly. Both G1 velocity tasks — flat terrain and rough terrain — write their logs under the same `g1_velocity` experiment name, so the script cannot auto-detect which task produced a given run directory.

**Parameters worth exploring:**

- `--command-lin-vel-x 0.3` — a slow, cautious pace. Watch whether the gait stays upright or collapses into a shuffle at low speed. Reveals how well the policy has generalized across the speed range it was trained on.
- `--command-lin-vel-x 1.5` — near the top of the trained speed range. The gait changes character noticeably.
- `--num-checkpoints 42` — include every saved checkpoint for fine-grained progression.
- `--cameras chase,side` — both camera angles (as used here). A side view makes gait asymmetries and posture obvious; a chase view shows heading drift.

---

## What you now understand

- The **velocity tracking task** gives the policy a target speed and direction — not a motion to replicate, but a goal to achieve. The robot figures out how to move by trial and error under that goal.
- A **reward term** is a single component of the total reward signal, targeting one specific aspect of behavior — forward speed, upright posture, foot lift, motion smoothness. The total reward is a weighted sum of all terms.
- A **reward weight** is the scalar multiplier on each term. It controls how loudly that term's incentive speaks relative to the others. The designer tunes weights, not motions.
- An **emergent gait** is a walking pattern that arose from optimizing reward terms, not from explicit choreography. The arm swing, step frequency, and posture visible in the final clip above were not specified anywhere — they appeared because they happen to score well under the chosen reward terms. The optimizer found them through search.
- Early in training, episodes end almost instantly. As the policy improves, episodes grow longer. The stills mark three distinct stages: the collapse of random weights, the awkward discovery of a gait, and the confident walking of a converged policy.

In Chapter 07 you will stress-test this result: run the same training from a different seed, inspect each reward term individually, and build real evidence that what you saw in the clips is consistent and real — not a lucky one-off. The lesson from Chapter 05 that the metric is a proxy, not the behavior, will come back with teeth.

Continue to [Chapter 07 — Proving It's Real](07-proving-its-real.md).

---

*Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>*
*Claude-Session: https://claude.ai/code/session_01D6dhn7JiNfx8tpFbitRmgN*
