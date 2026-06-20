# Turning the Knobs: One Change, Two Very Different Robots

*This is report 03 of four — the payoff. Terms like [policy](00-primer.md), [reward](00-primer.md), [episode](00-primer.md), and [PPO](00-primer.md) are defined in [00-primer.md](00-primer.md). Report [01](01-watching-it-learn.md) showed the baseline learning curve; report [02](02-reproducing-the-benchmark.md) reproduced the training and unpacked the reward terms. This report asks: what happens when you change exactly one thing?*

---

## The controlled experiment

The most direct way to understand any system is to change one thing and hold everything else constant. So we trained two [policies](00-primer.md) from scratch with an **identical recipe** — same [PPO](00-primer.md) algorithm, same [reward](00-primer.md) function, same neural network, same random seed (42), same 1,500 iterations — and changed exactly **one thing**: the range of forward speeds the robot practiced.

- **Control** — the standard setup: commanded forward speed drawn from **−1 to +1 m/s**. The robot practiced the full range of walking and slow-backing speeds.
- **Tweak ("slow")** — a restricted setup: commanded forward speed drawn from only **−0.5 to +0.5 m/s**. The robot only ever practiced slow speeds — never asked to walk faster than half a meter per second.

Everything else was left untouched. Same number of parallel robots (2,048). Same reward weights. Same network size.

One honest aside: the flat-terrain task includes a built-in speed curriculum — it automatically ramps the target speed range upward as training progresses. Pinning the tweak to the slow range meant also overriding those curriculum stages, so the one-line form of the command had two parts: `--env.commands.twist.ranges.lin-vel-x=(-0.5, 0.5)` plus the same override on each curriculum stage. The principle remains the same — one intended change — but it required two flags in practice.

---

## Result 1: the training curves

![Mean reward over training — control vs. slow-trained tweak](assets/ab_Train_mean_reward.png)

At the end of 1,500 iterations:

| Run | Final reward |
|-----|-------------|
| Control (speed range −1 to +1 m/s) | 38.4 |
| Tweak (speed range −0.5 to +0.5 m/s) | **57.8** |

The slow-trained policy reached a **higher reward** than the control. If you glance only at the number, the tweak looks like the "better" training run.

**It is not. This is the most important idea in this report, so it is worth stating plainly:**

> **A higher reward does not mean a better robot. It means the policy did better at the task it was given — and the slow task is simply easier.**

Tracking a speed of 0.3 m/s is easier than tracking a speed of 0.8 m/s. The [reward](00-primer.md) for velocity tracking is higher when the target is easier to hit. Nothing about the slow policy is more capable — it just got more points because it played an easier game. **Reward numbers are only meaningful when compared between runs that share the same task.**

A sprinter's score in a 100-meter dash cannot be compared to a jogger's score in a 10-kilometer race. Same principle.

![Speed-tracking error during training — control vs. slow-trained tweak](assets/ab_speed_tracking_error.png)

The velocity-tracking error plot above confirms this. The tweak's tracking error is lower throughout training — it is better at matching its commanded speed, because it was commanded a speed it could easily match. Lower error is better tracking; but the tweak achieved lower error by being given an easier target, not by being a more capable walker.

---

## Result 2: what happens when you ask both to go fast

A reward number is one window into a policy. To understand what the policy actually *does*, you have to watch it move. We commanded **both** final policies to walk forward at **1.0 m/s** — a brisk walk that is comfortably within the control's training range, but **double the tweak's training maximum**. The tweak had never, in its entire 1,500-iteration training history, been asked to go this fast.

The videos (`assets/ab_control.mp4`, `assets/ab_tweak.mp4`) tell the story most clearly. The stills below give the gist.

**Control policy — asked to walk at 1.0 m/s:**

![Control policy walking at 1.0 m/s](assets/ab_control_still.png)

The control strides forward confidently and stays on-heading. The robot's posture is upright and the motion looks purposeful. This is exactly what the control was trained to do — 1.0 m/s was in the middle of its practice range.

**Slow-trained policy — asked to walk at 1.0 m/s:**

![Slow-trained (tweak) policy walking at 1.0 m/s](assets/ab_tweak_still.png)

The tweak does not fall over — that is worth saying honestly. The slow training still produced a policy that knows how to stay upright, and the basic stabilization skills generalize to some degree. But the tweak drifts off-heading and moves hesitantly. It was never taught to coordinate its legs at this speed, so it handles the command poorly. The videos make the heading drift especially clear.

This is a concrete example of **out-of-distribution behavior**: when a [policy](00-primer.md) is asked to do something outside its training experience, it does not extrapolate gracefully. The lesson transfers directly to real robots: a policy trained only on easy conditions will behave unreliably when deployed in harder ones, even if its training metrics look impressive.

---

## The explore menu

Changing the commanded speed range is one knob. Here is a menu of others — each with a one-line description of what to expect and, where useful, the actual flag to use. All training commands run inside the `mjlab-dev` container:

```bash
python -m mjlab.scripts.train <task-id> [flags]
```

### Commanded speed range (what we just did)

```
--env.commands.twist.ranges.lin-vel-x=(min, max)
```

Narrow the range and the policy has an easier practice target — reward will look better, but the robot won't handle the full speed range. Widen it beyond ±1 m/s and the robot practices a faster walk; expect slower initial learning but better high-speed behavior. See above for what this trade-off looks like in practice.

### Reward weights

```
--env.rewards.<term>.weight <value>
```

Each term in the [reward](00-primer.md) has a weight that controls how much it contributes to the total score. Familiar terms from report [02](02-reproducing-the-benchmark.md):

- Lower the weight on the upright/fall term → the robot tolerates more lean, producing a sloppier gait that falls more often.
- Raise the weight on `action_rate_l2` (the smoothness penalty) → the robot learns to issue very gradual joint commands, producing smoother but potentially slower motion.
- Raise the weight on `air_time` → the robot takes more exaggerated, high-stepping strides.

Reward weights are the fastest way to shape *what kind* of walking emerges without changing the task itself.

### Train longer

```
--agent.max-iterations 5000
```

The baseline run in report [01](01-watching-it-learn.md) was still improving at 2,050 iterations — reward had not plateaued. More iterations push the reward higher and refine the gait. On the DGX Spark, 1,500 iterations takes about 26 minutes; 5,000 would take roughly 85 minutes. Returns are diminishing but the policy keeps improving.

### Harder worlds (different task IDs)

The task ID is the first argument to the training command. Three worth trying:

- **`Mjlab-Velocity-Rough-Unitree-G1`** — procedural bumpy terrain with a difficulty curriculum that ramps the terrain roughness as the policy improves. A policy trained here generalizes better to uneven ground.
- **`Mjlab-Velocity-Stairs-Unitree-G1`** — teaches the robot to climb and descend stairs. Learning is slower and the curriculum matters a lot.
- **`Mjlab-Velocity-Sprint-Unitree-G1`** — trains the robot to run at high speed. The gait that emerges looks markedly different from a walking policy — longer strides, more forward lean, higher energy.

### More or fewer parallel robots

```
--env.scene.num-envs 4096
```

More parallel robots means more experience per [PPO](00-primer.md) update and generally faster learning. It also requires more GPU memory. The baseline used 2,048; try 4,096 if memory allows, or 1,024 on a smaller machine.

### A different random seed

```
--agent.seed 7
```

Changing the seed runs the same recipe with different starting weights and different episode-reset randomness. Comparing two or three seeds is the most direct way to characterize how much of the result is the recipe versus luck. If the curves from different seeds nearly overlap, the recipe is robust; if they diverge widely, the result is sensitive to initialization.

---

## Good next adventures (out of scope here)

Two directions worth knowing about that go beyond tuning the velocity-walking task:

**Motion imitation.** The `Mjlab-Tracking-Flat-Unitree-G1` task trains the robot to reproduce a reference motion clip — a cartwheel, for example — rather than simply walking at a commanded velocity. The pipeline is documented in [`docs/cartwheel-journey.md`](../cartwheel-journey.md). It is a step up in complexity, but the same [PPO](00-primer.md) loop drives it.

**A different robot.** The `Mjlab-Velocity-Flat-Unitree-Go1` task trains a Unitree Go1 — a four-legged dog — on the same flat-terrain velocity task. The same algorithm and reward produce a trotting gait instead of a walking one because the morphology is different. Swapping the task ID is the only change needed.

---

## Summing up

Report [00](00-primer.md) gave you the vocabulary. Report [01](01-watching-it-learn.md) showed the arc from random twitching to walking. Report [02](02-reproducing-the-benchmark.md) showed the arc reproduces and unpacked the reward terms. This report showed that one number changed produces a robot that *looks* better on paper but performs worse when pushed outside its training range. The reward is a signal, not a scoreboard — understanding what it rewards, and what it does not, is the real skill.

The knobs above are the starting point. Turn one at a time, and watch what changes.

---

*All experiments use the Unitree G1 on flat terrain, trained with the MuJoCo-Warp simulator on a DGX Spark (NVIDIA GB10, aarch64).*
