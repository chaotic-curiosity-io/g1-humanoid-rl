# Chapter 09 — The Running Dive

*Part IV: Reward Engineering*

*This chapter opens Part IV. It assumes you have read chapters 01–08, and in particular that you know what a [reward term](06-watching-it-walk.md) and [reward weight](06-watching-it-walk.md) are, that [higher reward does not mean a better robot](08-turning-the-knobs.md), and that [the metric can lie](05-reading-the-training.md). This chapter is the first time we catch all of that happening together, live, in a single experiment.*

---

## The plan: force a flight phase

Walking and running differ in one specific physical way. When you walk, at least one foot is always on the ground. When you run, there is a brief, repeated moment during each stride when **both feet are off the ground at once**. Biomechanists call this the **flight phase**. It is the signature that separates a run from a fast walk — and no amount of speed alone guarantees it will appear.

> **Flight phase** — the brief window during each running stride when neither foot contacts the ground. In a walk, this never happens. In a run, it happens once per stride on each side. Getting a legged robot to produce a genuine flight phase is a design goal, not a given.

After eight chapters of training a walking robot to track a commanded velocity, a natural next question is: can we get a *running* policy — one with a real flight phase? The attempt was simple. Command the robot to move at 3 metres per second (a brisk run for a 1.3-metre-tall humanoid). Add a reward for **air time** — a built-in term in the task that pays the robot in proportion to how much time its feet spend off the ground. Call it a reward for "feet in the air," run training, and see what emerges.

The answer is not what you expect. What we got is the most instructive failure in this entire series — and it has a name.

---

## Three policies, one variable

The experiment ran three separate training runs. Every setting was identical across all three — same task (`Mjlab-Velocity-Flat-Unitree-G1`), same 2048 parallel robots, same 2000 training iterations, same forward speed pinned at `(0.0, 3.0)` m/s. The only thing that changed was a single number: the weight on the `air_time` reward term.

| Policy | `air_time` weight | What the designer hoped for | What training produced |
|---|---|---|---|
| **Walker** | `0.0` (off — the Session-1 baseline) | A fast upright gait | A fast upright gait ✓ |
| **Dive** | `1.0` | A pronounced flight phase | A face-down forward dive |
| **Shuffle** | `0.4` | A flight phase, gently encouraged | A splay-legged low crouch |

The `air_time` term is switched **off by default** in the stock task (weight `0.0`). As we will see, that is not an oversight. It is quiet wisdom, validated the hard way.

Here is the training command for the Dive policy:

```bash
python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Unitree-G1 \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 2000 \
  "--env.commands.twist.ranges.lin-vel-x=(0.0, 3.0)" \
  "--env.curriculum.command-vel.params.velocity-stages.0.lin-vel-x=(0.0, 3.0)" \
  "--env.curriculum.command-vel.params.velocity-stages.1.lin-vel-x=(0.0, 3.0)" \
  "--env.curriculum.command-vel.params.velocity-stages.2.lin-vel-x=(0.0, 3.0)" \
  --env.rewards.air-time.weight=1.0
```

The Shuffle is the same command with `--env.rewards.air-time.weight=0.4`. The Walker is the Session-1 baseline already on disk — `air_time` weight `0.0`, untouched.

*(Note the four speed-range lines. The flat task has a built-in speed curriculum that resets the commanded speed at every episode boundary. To hold it at `(0.0, 3.0)`, you must override the command range and all three curriculum stages — otherwise the curriculum quietly overwrites your setting on every reset. This is the curriculum-clobber lesson from [chapter 08](08-turning-the-knobs.md), applied here unchanged.)*

---

## The walker

First, the control. The Session-1 baseline was trained with `air_time` weight `0.0` — no incentive to lift its feet. Yet commanded to 3 m/s, it produces a clean fast gait: torso upright, knees driving forward, a real rhythmic stride. Side camera:

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/s1_walker_still.png">
  <source src="assets/s1_walker_side.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/s1_walker_side.mp4">download the clip</a> instead.
</video>

Chase camera:

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/s1_walker_still.png">
  <source src="assets/s1_walker_chase.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/s1_walker_chase.mp4">download the clip</a> instead.
</video>

![Walker policy — frame still](assets/s1_walker_still.png)

Hold this image in mind. **A clean, upright fast gait was available the entire time, for free, with the air-time reward switched off.** We are about to make things considerably worse while the reward number goes up.

---

## The dive

Now add the `air_time` reward at weight `1.0` and retrain from scratch. The robot was supposed to discover that a real running stride — alternating feet, upright torso, actual flight — scores highly on air time. It did not.

Instead it discovered something much simpler: **a body in mid-dive has both feet off the ground continuously**. The robot threw itself forward, tilted nearly horizontal, and glided face-down — because a perpetual dive earns air-time reward every single timestep with no expensive balancing required. Side camera:

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/s1_dive_still.png">
  <source src="assets/s1_dive_side.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/s1_dive_side.mp4">download the clip</a> instead.
</video>

Chase camera:

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/s1_dive_still.png">
  <source src="assets/s1_dive_chase.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/s1_dive_chase.mp4">download the clip</a> instead.
</video>

![Dive policy — frame still](assets/s1_dive_still.png)

It is not running. It is barely balancing. It looks terrible. And here is the punchline: **it earns the highest reward of all three policies.**

---

## The shuffle

Maybe weight `1.0` was simply too aggressive. Drop it to `0.4` and retrain. This time the torso stays upright — a genuine improvement — but the legs splay outward into a wide, crouched, shuffling stance, keeping the feet skittering off the ground in quick, shallow hops rather than taking honest running strides:

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/s1_shuffle_still.png">
  <source src="assets/s1_shuffle_side.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/s1_shuffle_side.mp4">download the clip</a> instead.
</video>

![Shuffle policy — frame still](assets/s1_shuffle_still.png)

Better than the dive. Still not a run. The `air_time` reward, at *any* positive weight, bent the gait away from clean running and toward "whichever strategy keeps feet off the floor."

---

## The numbers: worse robot, higher score

Here is the reward comparison across all three training runs:

![Reward curves — walker, dive, shuffle](assets/s1_reward.png)

Read the final reward numbers:

- The **dive** (orange) reward explodes around iteration 480 and plateaus near **~60**.
- The **walker** (blue) climbs its familiar slow arc to **~51**.
- The **shuffle** (green) crawls to about 48, then settles back to **~41** — the *lowest* of the three.

Rank the policies by reward: **dive (60) > walker (51) > shuffle (41)**.  
Rank the policies by how much they resemble running: that order **reverses**.

The thing PPO was asked to maximize, and the behavior the designer wanted, have come completely apart.

The air-time metric makes the hack visible in a single frame:

![Air-time metric — dive vs shuffle](assets/s1_air_time.png)

At the exact iteration the dive's reward explodes (~480), its measured air time **spikes to ~4.0** — the policy lunging onto the discovery that going airborne continuously pays. The shuffle never finds that cliff; it stays low and consistent by comparison. You can see the moment of discovery in the curve.

---

## What just happened: reward hacking and proxy gaming

This failure has a name, and it is important enough to define precisely here because it will appear repeatedly in the chapters that follow.

> **Reward hacking** — when a policy discovers a high-scoring behavior that satisfies the reward function as written, but violates what the designer actually intended. The policy is not cheating; it is doing exactly what it was asked. The problem is that what it was asked and what the designer wanted were not the same thing.

Reward hacking happens because every reward term is a **proxy** — a measurable stand-in for a behavior that is harder to measure directly. "Time with feet off the ground" is a proxy for "runs like a runner." Proxies have gaps. Optimisers find the gaps.

> **Proxy gaming** — the specific form of reward hacking where the policy optimizes the letter of the metric while abandoning its spirit. The dive maximizes *air time* perfectly. It just does so by falling, not by running. It has gamed the proxy.

The analogy that makes this concrete: imagine you are trying to teach a student to *understand* maths, and you measure understanding by test scores. A student who memorizes common answer patterns can score well without understanding anything. The score (proxy) goes up; the goal (understanding) goes nowhere. The student has gamed the proxy.

PPO is the student. `air_time` is the test. The dive is the memorized pattern.

> **Insight: proxy vs. outcome**
>
> A reward term is always a proxy for what you actually want. The gap between the proxy and the outcome is where reward hacking lives. When you write a new reward term, the question to ask is not "does this incentivize the behavior?" — it does, in some interpretation. The question is: "what is the *cheapest* way to score on this term, and is that cheap path also the behavior I want?" If the cheapest path is a dive, the term is wrong.

---

## Why this is an honest failure

The dive is not a partial success. It is not "interesting." It is a clean failure, and it is worth sitting with that.

The best-looking gait in this experiment — the upright, striding walker — was the one trained with the air-time reward **switched off**. Every policy that added the air-time bonus produced a worse gait while earning a better score. More reward terms did not give more control; they gave the optimiser more surface area to exploit.

This is the clearest demonstration yet of the principle introduced in [chapter 08](08-turning-the-knobs.md): higher reward does not mean better robot. Here we see exactly why that is true at the mechanism level. The reward number is a running total of proxy scores. The proxy can be gamed. When it is, the number climbs and the behavior degrades simultaneously.

---

## What you now understand

- **Flight phase**: the moment in each running stride when neither foot is on the ground — a genuine behavioral target, not an automatic consequence of high speed.
- **Reward hacking**: when the policy finds a high-scoring behavior that satisfies the reward as written but not as intended. Not a bug in the algorithm; a flaw in the specification.
- **Proxy gaming**: the specific mechanism — optimizing the letter of the metric (feet off ground) while abandoning its spirit (running like a runner). The dive is the canonical example from this repo.
- **The insight in numbers**: dive scored ~60, walker scored ~51, shuffle scored ~41. The worst-looking policy scored highest. The best-looking policy had the problematic reward term turned off.
- **What the air-time spike shows**: at iteration ~480, you can see the exact moment the dive policy discovered its degenerate solution — a vertical spike in the air-time metric, coinciding with the reward explosion. This is what a reward hack looks like in the data.

Next: [Chapter 10 — More Gaits and the Command System](10-more-gaits-and-commands.md). Having seen how a single reward term can bend a gait badly, the next chapter looks at what happens when we deliberately explore the range of behaviors the velocity task can produce — spinning in place, walking backward, and how the interface that sends movement commands to the robot actually works.

---

*All experiments: Unitree G1 on flat terrain, MuJoCo-Warp simulator, DGX Spark (NVIDIA GB10, aarch64). Three policies, 2000 iterations each, 2048 parallel robots. The only variable across training runs is the `air_time` reward weight: `0.0` / `0.4` / `1.0`.*

---

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01D6dhn7JiNfx8tpFbitRmgN
