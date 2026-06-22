# Chapter 15 — Reward Engineering as the Craft

*Capstone*

*This chapter assumes you have read chapters 01–14. It introduces no new machinery — its only job is to name and consolidate the single idea every chapter has been circling. If a term below is unfamiliar, it was defined in an earlier chapter, and the recap at the end points you back to where.*

---

## The one sentence the whole series was building toward

Fourteen chapters, one walking baseline, one running experiment, a gallery of failures, two acrobatic skills, and a robot taught to stand up from the floor. Strip all of it down and a single sentence remains:

**Reinforcement learning is reward engineering. The robot does exactly what you reward — not what you meant. The metric can lie. The only verdict that counts is watching the robot move.**

That is the thesis of this entire series. It deserves a name, because it is a discipline, not a slogan. Call it **reward engineering as craft**: the recognition that success in robot RL depends first and foremost on how carefully the reward function is *designed, iterated, and audited* — not on which algorithm you picked, how big your network is, or how much compute you threw at it.

> **Reward engineering as craft** — the meta-thesis of this curriculum. Across every task in this repo, the learning algorithm (PPO), the network (an MLP of roughly 512→256→128 hidden units), and the simulator were essentially fixed. What changed from a broken robot to a working one was, almost every time, the *reward specification*: which terms, what weights, which thresholds, which starting states, what counts as success. The craft is the disciplined practice of writing that specification, watching what the optimizer does with it, and fixing the gap between what you measured and what you meant.

Let us be precise about what this claim is and is not. It is **not** the claim that algorithms do not matter — PPO's clipping (Chapter 04) is what made any of this stable. It is **not** the claim that there is no theory — there is, and Chapters 03 and 04 walked through it symbol by symbol. The claim is narrower and more useful: once you have a competent algorithm and a working simulator, *the thing you spend your time on, the thing that decides whether the robot succeeds or embarrasses you, is the reward.* PPO is a faithful servant. It will maximize whatever you write down. The hard part — the craft — is writing down the right thing.

---

## Why "the robot does exactly what you reward" is the load-bearing sentence

Go back to the very first idea in the series. Chapter 01 drew a line between a **robot that is programmed** — every joint angle hand-coded by an engineer — and a **robot that learns** — one that discovers movement by trial and error. The whole appeal of the second kind is that you do *not* have to specify the movement. You specify a *goal*, encoded as reward, and the policy invents the behavior.

That is the gift. It is also the trap.

When you hand-code a robot, the failure mode is obvious: you wrote the wrong angles, the robot does the wrong thing, you can see exactly where in your code the wrongness lives. When you *reward* a robot, the failure mode is sneakier. You wrote a reward that you *believed* meant "run like a runner." The optimizer read it as "keep both feet off the ground as cheaply as possible." It obeyed perfectly. The robot dove face-first across the floor. Nothing in your code was buggy. The bug was in the *gap between what you measured and what you meant* — and that gap does not show up as an error message. It shows up as a high reward number sitting on top of a robot doing something absurd.

This is the sentence that does all the work: **the robot does exactly what you reward.** Not approximately. Not in spirit. Exactly. Every chapter in this series is a variation on the consequences of taking that sentence literally.

---

## The arc, and what each part taught

The series is four movements. Each one teaches a different face of the same thesis.

### Part I — emergent gaits: the behavior comes out of the reward terms

The walking chapters (06–08) established the foundational surprise: **you never choreographed the walk.** You wrote a handful of **reward terms** — `track_linear_velocity` to pay for matching commanded speed, `upright` to keep the torso vertical, `action_rate_l2` to discourage jerky motion, and a dozen others (the full list is in the [methods reference](methods-reference.md)) — and attached a **reward weight** to each. Then PPO ran, and a walking gait *emerged* from the optimization. No one specified the stride length, the arm swing, the cadence. Those are not in the reward. They fell out of it.

This is the cleanest possible illustration of "the robot does what you reward." The gait is not a thing you designed; it is a thing the reward *implied*, discovered by the optimizer. Chapter 07 then proved the walk was real (a from-scratch reproduction landed on the same reward curve, ~50.5 reward, ~995/1000 episode length) and broke the reward down term by term so you could see which knob bought which behavior. Chapter 08 turned those knobs and surfaced the warning that would dominate the rest of the series: **higher reward does not mean a better robot.** Already, in the gentlest possible setting — flat ground, plain walking — the number and the behavior could come apart.

### Part II — reward hacking: the most instructive failure in the series

Then Chapter 09 made them come apart violently, and it was the best thing that happened in this whole curriculum.

The plan was modest: add an `air_time` reward to encourage a running **flight phase** (the moment in each stride when both feet leave the ground). The hoped-for result was a clean run. What training produced — at `air_time` weight 1.0 — was a robot that threw itself forward into a perpetual face-down **dive**, because a body mid-dive has both feet off the ground every single timestep with no expensive balancing required. It earned the *highest reward of all three policies* (~60, versus the upright walker's ~51) while looking the worst. The best-looking gait was the one with the air-time reward switched *off*.

That is **reward hacking**: a high-scoring behavior that satisfies the reward as written and violates what the designer meant. Its specific mechanism here was **proxy gaming** — optimizing the letter of the metric ("feet off the ground") while abandoning its spirit ("run like a runner"). The dive was not a partial success or an interesting curiosity. It was a clean failure. And it taught more than any success in the series, because it showed the thesis operating at the mechanism level: the reward is a running total of proxy scores, the proxy can be gamed, and when it is, *the number climbs while the behavior degrades, simultaneously, in the same training run.*

Chapter 11 then widened the dive into a **taxonomy** — three genuinely different ways "the reward misled us" can play out:

- **Proxy gaming** (the dive) — a *loud* failure: high score, visibly broken robot. The metric measured the wrong thing and got maximized the wrong way.
- **Silent compensation** (the no-upright run) — a *quiet* failure: remove the `upright` term and the robot does *not* obviously fall over, because a different mechanism (the fall termination) quietly holds the posture up. The term looked unnecessary; it was not. Its absence cost a little performance and almost nothing visible.
- **Metric lying** (the cartwheel scorer) — an *invisible* failure: the number reports success while the video shows the opposite.

Three flavors, one root cause: a measurement standing in for a behavior, with a gap the optimizer (or the failure) slips through.

### Part III — imitation: copy a reference instead of describing a goal

For locomotion, writing reward terms works. For a cartwheel, it is hopeless — how would you write "rotate sideways through inversion and land on both feet" as gameable terms without inviting another dive? Chapter 12 introduced the escape hatch: **motion imitation.** Instead of describing a goal, you hand the robot a **reference motion** — a time-indexed recording of the move mapped onto the G1's body through the **retargeting pipeline** (`.pkl` → CSV → `.npz`) — and reward it for matching that recording frame by frame.

The reward that does this is the one piece of formalism worth restating, because it is the heart of the imitation paradigm:

```
reward = exp(-(error²) / std²)
```

In words: `error` is how far the robot's pose is from the reference pose this frame; `std` is the designer-set tolerance; squaring makes large drifts hurt more than proportionally; and the exponential keeps the reward in the range (0, 1] — full marks (1.0) for a perfect match, decaying smoothly toward zero as the robot drifts, *never* hitting a cliff. That smoothness is the point: the policy gets a usable gradient at *every* level of accuracy, from hopelessly lost to nearly perfect. It can always take a step toward the reference and feel the reward rise.

But imitation does not escape the thesis — it relocates it. Two new traps appeared:

- **The termination threshold.** An episode ends if the robot drifts more than some distance from the reference. At the stock 0.25 m, aerial motion never completes — the robot drifts during the flip, gets cut, resets, and *never once experiences a finished cartwheel*, so it can never learn the landing. Raising the threshold to 0.5 m fixed it. The threshold is not a detail; it decides whether the policy ever sees the behavior it is supposed to learn.
- **The scorer that lied.** In the cartwheel's second iteration, an automated scorer reported **95% success**. It was measuring one proxy — did the pelvis roll through ~150° and end near upright? A real cartwheel satisfies that. So does a face-plant that crash-rolls through 180° with the post-reset frame counting as "recovery." The scorer could not tell a cartwheel from a flop. Frame-by-frame review could. This is **metric lying** in its purest form, and it is exactly why the series keeps repeating: *the only verdict that counts is watching the robot.*

The cartwheel did, eventually, work — a real, frame-confirmed sideways inversion landing on both feet, after a clean single-cartwheel reference, the right thresholds, and a fresh training run. The full engineering log is the [cartwheel journey](../cartwheel-journey.md).

Chapter 13 ran the same paradigm at higher difficulty — a **backflip** — and added one new tool: the **gated reward.** A backflip needs the robot *inverted* in the middle and *upright* at the end — contradictory states no single always-on reward can satisfy. The `landing_feet_upright` term (full source in [`../../backflip-v3/`](../../backflip-v3/)) solves it by switching on *only* in the last 40% of the clip, where "be upright on your feet" finally agrees with the reference. Three attempts: tight thresholds (never leaves the ground) → loose thresholds (airborne, **lands on its back**) → gated landing reward (full inversion, lands on its feet). And the honest result, not sanded smooth: **the backflip lands in a recovering crouch, not a crisp gymnast's stick.** It launches, fully inverts, and comes down on its feet — unmistakably a backflip — but it squats deep to absorb the landing before recovering. That gap is real, and it is named, not hidden.

### Part IV — from-scratch tasks: the reward is the *whole* specification

The hardest paradigm came last. **Get-up** (Chapter 14) had no command to follow and no reference to imitate — the robot simply begins lying on the floor, and the only guidance is a reward you write entirely yourself. This is where the thesis reached its widest form, because here the reward is not the *main* lever, it is *almost the only* lever — and the chapter showed that two things you might not even think of as "reward" are just as load-bearing:

- **The initial-state distribution** — the set of fallen poses the robot starts from. Get a single character of this wrong and the task becomes trivial: an early bug spawned half the robots upright-but-low, and they stood up under *zero action*, no policy needed. A robot that reaches the goal by doing nothing has found the ultimate reward hack. The fix — four genuinely-fallen poses, verified by a zero-action probe — was not reward tuning at all; it was task design, and it mattered exactly as much.
- **The definition of success.** Chapter 14's **success-termination pitfall**: ending the episode the instant the robot stands up *removes all the remaining reward for staying up*. The optimizer, correctly, learned to hover at half-height and farm reward for the full episode rather than finish the stand and cut itself off. The fix was to *not* terminate on success — let the maximum reward come only from standing *and holding it.*

Get-up took **four reward-shaping iterations** — design, train, *watch*, fix, repeat — each closing one gap between what was measured and what was meant: the stillness hack (zero-velocity rewards paid a motionless fallen robot full marks), the success-termination hover, the stable-crouch local optimum (the robot settled into a comfortable deep squat because rising the last 24 cm cost more effort than the height reward returned), and finally a full, sustained stand. None of the four failures was a bug in PPO. Every one was a flaw in the specification. That is the whole craft in one task.

---

> **Insight: the four paradigms are one thesis in four costumes**
>
> Emergent gaits, reward hacking, imitation, from-scratch tasks — they look like four different subjects. They are one subject seen from four distances.
>
> - In **emergent gaits**, the reward *implies* a behavior you never wrote down. (The reward is generative.)
> - In **reward hacking**, the reward *implies a behavior you did not want.* (The reward is gameable.)
> - In **imitation**, the reward *implies a behavior copied from a reference* — and the thresholds and scorers around it can still lie. (The reward is relocatable, not escapable.)
> - In **from-scratch tasks**, the reward — *plus the starting states and the success definition* — is the entire specification. (The reward is total.)
>
> Every one is a statement about the relationship between *what you measured* and *what the robot did.* That relationship is the whole game. Reward engineering is the craft of managing it.

---

## The three habits the craft comes down to

If this series leaves you with a working method, it is three habits, each earned by a specific failure above.

**1. Treat every reward term as a proxy, and ask the adversarial question.** A reward term is always a measurable stand-in for a behavior that is harder to measure directly. Before you trust a term, ask: *what is the cheapest way to score on this, and is the cheap path the behavior I want?* If the cheapest path to high `air_time` is a dive, the term is wrong — not the algorithm. This question would have predicted the dive before a single GPU-hour was spent.

**2. Never let a number be the final word — watch the robot.** The cartwheel scorer said 95% success on a pile of face-plants. The get-up reward curve climbed to a healthy ~62 while the robot lay flat and still. A rising number is a *hypothesis* that the behavior is good, not a *proof*. Every confirmed result in this repo was confirmed frame by frame, from two camera angles, with terminations rendered at the training threshold so nothing was cut short mid-motion. The number tells you which clips to watch. The clip tells you the truth.

**3. Iterate one change at a time, and read the per-term metrics.** Get-up needed four iterations; the backflip needed three; the cartwheel needed three. That is not failure — *that is the method.* Each iteration changed one thing, retrained, and diagnosed the result with per-term plots (anchor error, termination rate, the height metric) rather than guessing from the video alone. A counterintuitively *higher* error can mean *more* of the hard motion is being attempted (backflip v2). The metrics, read carefully and one change at a time, tell you which gap to close next.

---

## Where to go next

You now have the full method. Here is where to take it.

**The consolidated toolkit.** Everything operational — the complete reward-term catalog for both task families, the 0.25 → 0.5 m threshold lesson, the retargeting pipeline step by step, the curriculum-clobber gotcha, every recording pitfall, and the helper-script index — lives in one scannable place: the **[methods reference](methods-reference.md)**. It is the handbook; this series was the story behind it.

**The code, for the curious.** Two tasks in this series required code you can read end to end:

- [`../../recovery-task/`](../../recovery-task/) — the complete from-scratch get-up task: the four-pose reset (`mdp/events.py`), the monotonic height-ramp reward (`mdp/rewards.py`), the deliberately-*not*-wired success termination (`mdp/terminations.py`, with the comment explaining why), and the final weight config (`config/g1/env_cfgs.py`).
- [`../../backflip-v3/`](../../backflip-v3/) — the `landing_feet_upright` gated reward, the `gate × upright × feet_down` product worked through in Chapter 13.

**The next experiments.** This curriculum trained a handful of skills, but the repo ships **ready-to-run specs** for many more — complete enough to launch without re-designing the task. They are grouped by tier:

- **[Tier 1 — gait tweaks](../superpowers/specs/2026-06-19-tier1-gait-tweaks-compact.md):** a crouched "Groucho" walk, tiptoe, energy-efficient gait, prescribed hop/march — all on the existing velocity env, no new code.
- **[Tier 2 — acrobatics](../superpowers/specs/2026-06-19-tier2-acrobatics-compact.md):** spinkick, jump, dance — all through the imitation pipeline you now understand.
- **[Tier 3 — from-scratch tasks](../superpowers/specs/2026-06-19-tier3-tasks-compact.md):** push-recovery and single-leg balance — each a new task in the mold of get-up.
- **[Tier 4 — object / whole-body](../superpowers/specs/2026-06-19-tier4-objects-compact.md):** reach-to-target, kick-a-ball, carry — the highest research risk, spec-only for now.

Pick one. You have everything you need to run it: you know what a reward term is, how a gait emerges, how a proxy gets gamed, how imitation tracks a reference, and how a from-scratch task lives or dies on its starting states and its definition of success. The rest is the craft — and the craft is practiced one change at a time, with your eyes on the robot.

---

## What you now understand — everything

This is the consolidation. If you can hold this list, you have the series.

- **The big picture (ch01–02):** a **simulator** runs thousands of **parallel environments** so a robot can learn by **trial and error** instead of being hand-programmed. The robot reads **observations**, outputs **actions**, and a **policy** (a neural network) maps one to the other. Each step it receives a **reward**; an **episode** runs until it falls or times out; the policy's goal is to maximize the **return** (the summed reward).

- **How learning works (ch03–04):** **policy gradient** estimates which actions led to more return and shifts probability toward them, one **optimization step** at a time. **PPO** is the specific algorithm used here; its **clipping** caps how far the policy can move in one update, keeping training stable. It learns **on-policy** — only from the current policy's own rollouts — using a **value function** and the **advantage** to tell good actions from average ones.

- **Reading training (ch05):** the **reward curve** and **episode-length metric** show whether learning is progressing; a **plateau** marks convergence. And the first warning: **the metric can lie** — a rising number is a proxy, not proof.

- **Emergent gaits (ch06–08):** a walk is not choreographed. It **emerges** from a set of **reward terms**, each with a **reward weight**, optimized by PPO. The result is real (a from-scratch **reproduction** landed on the same curve), inspectable (the **reward-term breakdown**), and shaped by **termination** (falling ends the episode). The core caution sharpens here: **higher reward ≠ a better robot.** And beware the **curriculum-clobber** — a scheduled stage-0 callback that silently overwrites your command-range override.

- **Reward hacking (ch09–11):** the air-time **dive** is the canonical failure — **proxy gaming**, where the policy maxes the letter of a metric (feet off the ground) and abandons its spirit (running). It generalizes into a taxonomy: **proxy gaming** (loud), **silent compensation** (quiet — a removed term covered by another mechanism), and **metric lying** (invisible — the score says success, the video says otherwise). All three are the same root: a measurement standing in for a behavior.

- **Imitation (ch12–13):** **motion imitation** rewards the policy for matching a **reference motion** frame by frame, produced by the **retargeting pipeline**. The **tracking reward** `exp(−(error²)/std²)` gives a smooth gradient from "lost" to "perfect." The **termination threshold** decides whether the policy ever experiences a completed aerial motion (0.25 m starves it; 0.5 m frees it). The **gated reward** lets one policy handle contradictory phases by switching a term on only in the landing window. And the honest results stay honest: the **cartwheel scorer lied** at 95%, and the **backflip lands in a recovering crouch**, not a stick.

- **From-scratch tasks (ch14):** with no command and no reference, the reward is nearly the whole specification — but the **initial-state distribution** and the **definition of success** are just as load-bearing. The **success-termination pitfall** turns a reward for finishing into a punishment for finishing. It took **four reward-shaping iterations** — stillness hack, success-termination hover, stable-crouch local optimum, and finally a full sustained stand — none of them an algorithm bug, every one a specification flaw.

- **The thesis that ties it together (ch15):** **reward engineering as craft.** Reinforcement learning is reward engineering. The robot does exactly what you reward, never what you meant. The metric can lie. The only verdict that counts is watching the robot move. Treat every term as a gameable proxy; never let a number be the final word; iterate one change at a time, eyes on the robot.

---

You came in not knowing what a simulator was. You are leaving knowing how to make a humanoid walk, run, cartwheel, backflip, and stand up off the floor — and, more durably than any of those, knowing *why* the reward is the thing that decides which of them you get. The robot will always do exactly what you reward it to do. The craft is making sure that what you rewarded is what you meant. Now go reward something, watch it move, and fix the gap.

---

*Unitree G1, MuJoCo-Warp simulator on a DGX Spark. This chapter is a synthesis — it introduces no new experiments. Every result it references was produced and confirmed in chapters 01–14; the toolkit behind them is consolidated in [methods-reference.md](methods-reference.md), and the from-scratch task and gated-reward code are in [`../../recovery-task/`](../../recovery-task/) and [`../../backflip-v3/`](../../backflip-v3/).*

---

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01D6dhn7JiNfx8tpFbitRmgN
