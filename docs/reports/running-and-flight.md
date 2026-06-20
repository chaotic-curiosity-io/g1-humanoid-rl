# Running and the Flight Phase: When Both Feet Leave the Ground

> **Draft.** This report is populated after the S1 training run (see [the spec](../superpowers/specs/2026-06-19-spine-running-flight.md) for the plan). The structure below is the skeleton it will fill.

*This report is part of the Locomotion track. If the terms below — [policy](00-primer.md), [reward](00-primer.md), [episode](00-primer.md), [PPO](00-primer.md) — are new to you, read [00-primer.md](00-primer.md) first. Report [03](03-turning-the-knobs.md) is the direct predecessor: it showed what happens when you restrict the speed range. This report asks the opposite question — what happens when you push it much higher?*

---

## What you are about to see

Walking and running feel like points on a continuum, but they are physically different in one specific way: when you walk, at least one foot is always on the ground. When you run, there is a moment — brief, repeated — when both feet are off the ground at the same time. That brief airborne moment is called the **flight phase**, and it is the physical signature that separates running from walking.

The baseline walker from Session 1 (`model_2050.pt`) never achieves a flight phase. Even at a brisk pace, its step pattern stays planted: at least one foot is always in contact with the floor. This report asks what happens when we train a fresh policy with a much higher commanded speed and an explicit reward for getting feet off the ground — and whether a true running gait emerges on its own.

The experiment follows the same "change exactly one thing" discipline from [report 03](03-turning-the-knobs.md). Here, instead of restricting the speed range, we widen it substantially — and add one extra push in the reward function to encourage airtime.

---

## The command

The full training command for the fast runner (B policy), run inside the `mjlab-dev` container on the Spark:

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Unitree-G1 \
  \"--env.commands.twist.ranges.lin-vel-x=(0.0, 3.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.0.lin-vel-x=(0.0, 3.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.1.lin-vel-x=(0.0, 3.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.2.lin-vel-x=(0.0, 3.0)\" \
  --env.rewards.feet_air_time.weight 2.0 \
  --agent.max-iterations 2000 \
  --agent.seed 42'"
```

**Why the four speed-range lines?** The flat-terrain task has a built-in speed curriculum that resets the commanded speed range at every episode boundary. To pin the range at `(0.0, 3.0)`, you must override not just the command range itself but all three of the curriculum's stages — otherwise the curriculum quietly overwrites your setting every reset. This was the key lesson from [report 03](03-turning-the-knobs.md), and it applies here too. See the [spec](../superpowers/specs/2026-06-19-spine-running-flight.md) for full details on the curriculum-clobber gotcha.

The control for comparison is the existing walker: `logs/rsl_rl/g1_velocity/2026-04-17_18-46-23/model_2050.pt` — already on disk, no retraining needed.

---

## Results

*This section is populated after the S1 training run. Placeholders below describe what will appear here.*

### Training curves

**[Placeholder: reward curve plot — `assets/s1_reward_curve.png`]**

The training curve for the fast runner will be shown here, alongside the control run's curve for comparison. Key numbers to look for: how quickly does the fast runner's reward climb relative to the baseline? Does it plateau at the same level, higher, or lower?

**[Note for the reader: reward numbers are not directly comparable between the two runs because they are trained on different speed distributions. A higher reward for the fast runner does not mean a better robot — it means a robot that scored well on a harder task. See [report 03](03-turning-the-knobs.md) for a full explanation of why this distinction matters.]**

### The A/B clips: walker vs. runner

**[Placeholder: control clip — `assets/s1_A_walker_1p5ms.mp4`]**

*Control walker (A) — commanded at 1.5 m/s: this will show the existing `model_2050.pt` walking at a brisk pace. Watch the feet — at least one always touches the ground.*

**[Placeholder: fast runner clip — `assets/s1_B_runner_1p5ms.mp4`]**

*Fast runner (B) — commanded at the same 1.5 m/s: this will show the newly trained policy at the same speed. The question is visible in the clip: do both feet leave the ground?*

### Cadence and step-length plot

**[Placeholder: cadence-vs-speed plot — `assets/s1_cadence_vs_speed.png`]**

This plot will show step cadence (steps per second) or air-time fraction versus commanded forward speed, overlaid for both the control and the fast runner. If a flight phase has emerged, the fast runner's curve will show a measurably different relationship — longer strides, more time in the air per step — at the same commanded speed.

### What the clips showed

**[Placeholder: prose description of the visual result — written after the run.]**

The key question answered here: did a visible flight phase emerge? Could you freeze the clip on a frame and see both feet off the ground simultaneously? And was the behavior attributable to the changed knobs — the wider speed range and the increased airtime reward — rather than to any other change?

---

## Tweak this to explore

Once the S1 run lands and this report is populated, here are the knobs worth turning:

**Speed ceiling** — the fast runner was trained up to 3.0 m/s. Try pushing to 4.0 m/s (or pulling back to 2.0 m/s) to see where the flight phase emerges and disappears. Remember to override all three curriculum stages, or the curriculum will quietly reset the range every episode.

```bash
"--env.commands.twist.ranges.lin-vel-x=(0.0, 4.0)"
"--env.curriculum.command-vel.params.velocity-stages.0.lin-vel-x=(0.0, 4.0)"
# ... (repeat for stages 1 and 2)
```

**Airtime weight** — the spec used `--env.rewards.feet_air_time.weight 2.0` as a starting point. Crank it higher and the robot will prioritize getting its feet off the ground even more aggressively — possibly at the expense of forward speed or stability. Drop it to zero and see how much of the running gait survives without that explicit push.

**Training duration** — the fast run used 2,000 iterations. The baseline walker was improving at 2,050 iterations and had not plateaued. Does the fast runner's flight phase become cleaner with more training, or does it emerge early and stay roughly stable?

**A/B speed test** — once you have both policies, try commanding them both at a speed that was inside the fast runner's training range but outside the walker's (say, 2.0 m/s). The walker from [report 03](03-turning-the-knobs.md) drifted off-heading when pushed outside its training range. Does the same happen here, and does the fast runner handle it better?

---

*All experiments use the Unitree G1 on flat terrain, trained with the MuJoCo-Warp simulator on a DGX Spark (NVIDIA GB10, aarch64). The spec for this run: [2026-06-19-spine-running-flight.md](../superpowers/specs/2026-06-19-spine-running-flight.md).*
