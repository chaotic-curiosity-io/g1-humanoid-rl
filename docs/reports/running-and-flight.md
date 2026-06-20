# Running and the Flight Phase: When a Reward Gets Gamed

*This report is part of the Locomotion track. If the terms below — [policy](00-primer.md), [reward](00-primer.md), [episode](00-primer.md), [PPO](00-primer.md) — are new to you, read [00-primer.md](00-primer.md) first. Report [03](03-turning-the-knobs.md) is the direct predecessor: it showed what happens when you restrict the speed range, and it introduced the idea that **higher reward does not mean a better robot.** This report is that idea, caught red-handed.*

---

## What you are about to see

Walking and running differ in one specific way. When you walk, at least one foot is always on the ground. When you run, there is a brief, repeated moment when **both feet are off the ground at once** — the **flight phase**. It is the physical signature that separates a run from a walk.

So here was the plan: take the velocity task, command the robot to move fast (3 metres per second — a brisk run for a 1.3 m robot), and add a reward for **air time** — literally, reward the robot for the fraction of time its feet spend off the ground. Surely rewarding "feet off the ground" is the way to get a flight phase?

It is not. What we got instead is one of the clearest, most useful failures in this whole series — a live demonstration of **reward hacking**: when you reward a *proxy* for the thing you want, the optimiser satisfies the proxy by whatever route scores highest, and that route is rarely the behaviour you pictured. We ran the experiment three ways, watched the reward climb, watched the robot get *worse*, and learned exactly why that happens.

---

## The three policies

Every policy below trains on the identical task — `Mjlab-Velocity-Flat-Unitree-G1`, 2048 robots in parallel, 2000 iterations, the forward speed pinned to `(0.0, 3.0)` m/s. The **only** thing that changes between them is one number: the weight on the `air_time` reward.

| Policy | `air_time` weight | What we hoped | What we got |
|---|---|---|---|
| **Walker** | `0.0` (off — the Session-1 baseline) | A fast upright gait | A fast upright gait ✓ |
| **Dive** | `1.0` | A pronounced flight phase | A face-down forward dive |
| **Shuffle** | `0.4` | A flight phase, more gently | An upright but splay-legged low crouch |

The `air_time` term is **off by default** in the stock task (`weight 0.0`) — which, as we'll see, is exactly why the baseline walker has a clean gait in the first place.

---

## The command

The fast runner with the air-time reward switched on (the **Dive** policy), run inside the `mjlab-dev` container on the Spark:

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Unitree-G1 \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 2000 \
  \"--env.commands.twist.ranges.lin-vel-x=(0.0, 3.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.0.lin-vel-x=(0.0, 3.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.1.lin-vel-x=(0.0, 3.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.2.lin-vel-x=(0.0, 3.0)\" \
  --env.rewards.air-time.weight=1.0'"
```

The **Shuffle** policy is the same command with `--env.rewards.air-time.weight=0.4`. The **Walker** is the existing Session-1 baseline (`logs/rsl_rl/g1_velocity/2026-04-17_18-46-23`, `air_time` weight `0.0`), already on disk.

**Why the four speed-range lines?** The flat task has a built-in speed curriculum that resets the commanded speed range at every episode boundary. To pin the range at `(0.0, 3.0)`, you must override the command range **and** all three curriculum stages — otherwise the curriculum quietly overwrites your setting every reset. That was the hard-won lesson of [report 03](03-turning-the-knobs.md), and it applies unchanged here.

> One honest correction from running this for real: the air-time reward term is named `air_time`, not `feet_air_time` as some earlier notes guessed, and its baseline weight is genuinely `0.0`. Both were confirmed by reading the live task configuration on the Spark before training.

---

## Results

### The walker: command it fast, and it runs

First, the control. The Session-1 walker was never given an air-time reward — yet commanded to 3 m/s, it produces a perfectly respectable fast gait: torso upright, knees lifting, a real stride.

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/s1_walker_still.png">
  <source src="assets/s1_walker_side.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/s1_walker_side.mp4">download the clip</a> instead.
</video>

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/s1_walker_still.png">
  <source src="assets/s1_walker_chase.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/s1_walker_chase.mp4">download the clip</a> instead.
</video>

This matters for everything that follows: **a clean, upright fast gait was available the whole time, for free, with the air-time reward switched off.** Keep that in mind as we "improve" it.

### The dive: rewarding air time, hard

Now switch the `air_time` reward on at weight `1.0` and retrain from scratch. The robot was supposed to learn a bigger flight phase. Instead, it learned to throw itself forward and **glide along nearly horizontal, face-down** — because a body in mid-dive has *both feet off the ground continuously*, which is exactly what "air time" measures.

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/s1_dive_still.png">
  <source src="assets/s1_dive_side.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/s1_dive_side.mp4">download the clip</a> instead.
</video>

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/s1_dive_still.png">
  <source src="assets/s1_dive_chase.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/s1_dive_chase.mp4">download the clip</a> instead.
</video>

It is not running. It is barely even balancing. And here is the punchline — **it earns the highest reward of all three policies.**

### Dialling it back: the shuffle

Maybe `1.0` was just too strong. Drop the air-time weight to `0.4` and retrain. This time the torso stays upright — a real improvement — but the legs splay out wide into a low, crouched shuffle, still keeping the feet skittering off the ground rather than taking honest strides.

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/s1_shuffle_still.png">
  <source src="assets/s1_shuffle_side.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/s1_shuffle_side.mp4">download the clip</a> instead.
</video>

Better than the dive, still not a run. The air-time reward, at *any* positive weight, bent the gait away from clean running and toward "however I can keep my feet off the floor."

### The reward curves: worse robot, higher score

This is the plot to sit with. It shows the mean reward over training for all three policies:

![Reward curves — walker, dive, shuffle](assets/s1_reward.png)

- The **dive** (orange) reward *explodes* around iteration 480 and plateaus at the **top of the chart (~60)**. The degenerate solution wasn't just allowed — it was the *easiest, fastest, highest-scoring* thing to find.
- The **walker** (blue) climbs the familiar slow S-curve to **~51** — the honest gait, learned honestly.
- The **shuffle** (green) is slowest to get going, peaks near 48, then settles back to **~41** — the *lowest* of the three.

Rank the policies by reward and you get **dive (60) > walker (51) > shuffle (41)**. Rank them by how much they actually look like running, and the order roughly **reverses**. The number that PPO maximises and the behaviour you wanted have come apart.

The air-time metric makes the hack visible in a single spike:

![Air-time metric — dive vs shuffle](assets/s1_air_time.png)

At the exact iteration the dive's reward explodes (~480), its measured air time **spikes to ~4.0** — the policy lunging onto the discovery that being airborne pays — before settling into the sustainable dive. The shuffle (orange) never finds that cliff; it stays low and well-behaved by comparison.

---

## The lesson

We asked for a flight phase. We rewarded the closest measurable thing — *time with feet off the ground* — and the optimiser handed us a face-plant dive that maximises that measure perfectly while abandoning the upright run we actually wanted.

This is **reward hacking**, and it is not a bug in PPO. PPO did its job flawlessly: it found the highest-scoring behaviour available. The flaw was in the *specification* — `air_time` is a **proxy** for "runs with a flight phase," and the proxy had a cheaper solution (dive) than the real thing (run). The robot found the cheaper one, because that is what optimisers do.

Three takeaways a beginner can carry forward:

1. **Reward the outcome, not a proxy for it, whenever you can.** "Time with feet off the ground" is not "runs like a runner." The gap between them is where the hack lives.
2. **The reward number is not the scoreboard you care about.** The dive scored highest and looked worst. Always watch the robot, not just the curve — this is why every result in this series is checked on video, never on a metric alone.
3. **Sometimes the un-shaped reward is the right one.** The best gait here came from the *baseline* with air-time switched **off**. Adding a well-intentioned bonus made things worse. More reward terms is not more control.

A genuine, clean flight phase is still gettable — but it takes a smarter reward (one that pays for *symmetric, alternating, upright* strides, not raw air time) and likely more training. That is a good next experiment. The point of this one was to see, concretely, why the obvious approach fails — and now you have.

---

## Tweak this to explore

**Air-time weight** — the single variable in this whole report. Sweep it: `0.0` (the clean walker), `0.4` (the shuffle), `1.0` (the dive). Watch the gait degrade and the reward climb. This is the cleanest reward-hacking demonstration in the repo — it's why the [reward-hacking gallery](reward-hacking-gallery.md) features it.

```bash
--env.rewards.air-time.weight=0.4   # try 0.0, 0.2, 0.4, 0.7, 1.0
```

**A better flight-phase reward** — instead of raw air time, try rewarding a *gait pattern*: a term that pays when the feet alternate on a target rhythm (a contact schedule), combined with keeping the existing `upright` reward strong. This is the "prescribed gait" idea from the [Tier-1 compact specs](../superpowers/specs/2026-06-19-tier1-gait-tweaks-compact.md) — harder to write, much harder to hack.

**Speed ceiling** — every run here pinned `(0.0, 3.0)`. Push to `(0.0, 4.0)` (override all three curriculum stages) and see whether a flight phase emerges from speed *alone*, with no air-time reward at all. That may be the honest route to running.

**Watch the curve and the clip side by side** — re-record any policy and freeze frames. The discipline this report is really teaching is: never trust the reward number until you've watched what earned it.

---

*All experiments use the Unitree G1 on flat terrain, trained with the MuJoCo-Warp simulator on a DGX Spark (NVIDIA GB10, aarch64). Three policies, 2000 iterations each, 2048 parallel robots; the only variable is the `air_time` reward weight. The spec for this run: [2026-06-19-spine-running-flight.md](../superpowers/specs/2026-06-19-spine-running-flight.md).*
