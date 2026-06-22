# Chapter 04 — PPO for Novices

*Chapter 03 gave you the engine: the policy-gradient rule. Nudge the weights θ toward actions that earned higher-than-expected return, away from the ones that fell short, repeat a few thousand times. It also left two soft spots hanging — a hand-wavy "expected return" baseline, and the tension between nudges that are too small (slow) and too big (unstable). This chapter closes both. It introduces the algorithm this project actually trains with — **PPO** — and the two ideas it leans on: the **value function** and the **advantage**. This is the deepest theory chapter in the series. Everything here refines the Chapter 03 rule; nothing replaces it. Take it slowly, and keep the soup analogy nearby.*

---

## First, the honest disclaimer

Before anything else: **we did not invent PPO, and we did not write it.** PPO (Proximal Policy Optimization) is a standard, widely-used reinforcement-learning algorithm published years ago. In this project it arrives pre-built, inside a library called **rsl_rl** that sits underneath the `mjlab` training code. When you run a training command in this repo, rsl_rl is the thing quietly doing the policy-gradient math, the value-function math, and the clipping you are about to learn. We *configured* it and *fed it rewards*; we did not author the optimizer.

That matters for how you read this chapter. Your job is not to be able to re-implement PPO. Your job is to understand *what it is doing on your behalf* well enough to read a training run and know whether it is working. That understanding is what this chapter delivers.

---

## The problem with the raw rule: over-correcting the steering

Recall the end of Chapter 03. The raw policy-gradient rule works, but it is fragile, and the fragility has a specific shape: **how far do you step?**

Here is the analogy. You are driving down a straight road, and you drift a little to the right. You correct by turning the wheel left. The question is *how hard*.

- Turn the wheel a tiny bit, and you barely close the gap — you will drift along the shoulder for miles before you are centered again. Safe, but agonizingly slow.
- Turn the wheel hard, and you over-correct: now you are veering into the oncoming lane, so you yank it back the other way, over-correct again, and start fishtailing. Each correction is bigger than the error it was fixing. A few oscillations later you are in the ditch.

The good driver makes corrections that are *as firm as possible without ever overshooting* — big enough to make real progress, small enough that you never swing past center.

The raw policy-gradient rule is the bad driver. Remember *why* its steps have to be kept small (the Chapter 03 insight): a single rollout's return is noisy. A robot can earn high return partly by luck — a fortunate stumble that happened to stay upright. If the update steps θ hard toward whatever scored highest in *one* noisy batch, it over-corrects toward a fluke, and the next batch over-corrects back. The policy fishtails. In the worst case it leaps so far that a brain which was steadily improving is wrecked in a single update, and the return curve collapses instead of climbing.

So the raw rule forces an unhappy choice: keep the learning rate tiny and crawl, or crank it up and risk a catastrophic step. Neither is good. PPO is the engineering answer to "be the good driver" — make each step as bold as possible *without ever overshooting into the ditch.* The rest of this chapter builds that answer in two pieces: first a better measuring stick (advantage, via the value function), then the mechanism that caps the step size (clipping).

> **Insight: instability is not a small bug, it is the whole difficulty.**
> A beginner imagines the hard part of RL is "finding the right answer." It is not — the policy-gradient rule already points roughly the right way. The hard part is *taking steps in that direction without destroying what you already have.* Most of the cleverness in modern RL algorithms, PPO included, is machinery for being aggressive and safe at the same time. Hold that frame: PPO is not a smarter direction, it is a safer step.

---

## Advantage — "was this action better than expected?"

Chapter 03 kept saying *higher-than-expected return* and *lower-than-expected return*, and quietly admitted it never said how the algorithm knows what to *expect*. Time to fix that. It takes two new ideas, in order: first the **value function** (the thing that computes "expected"), then the **advantage** (the comparison against it).

### The value function — a learned guess of expected return

**Plain language first.** Imagine a second observer sitting beside the robot. This observer never moves a joint. Its only job is to look at the robot's current situation — its pose, its velocity, how upright it is — and answer one question: *"From a spot like this, how much total return does a robot like me usually go on to earn?"* It is a forecaster. Show it a robot standing balanced and centered, and it forecasts a high number ("situations like this usually end well"). Show it a robot already tipping over at a bad angle, and it forecasts a low number ("situations like this usually end in a fall"). The forecaster does not know the future; it has just watched thousands of episodes and learned the pattern of which *situations* tend to pay off.

That forecaster is the **value function**.

**Now the formalism, gently.** The value function is written `V(s)`, and in this project it is a second neural network, trained alongside the policy:

```
V(s)  =  the expected total future return, starting from situation s
```

Reading it symbol by symbol:

- **`V`** — short for *value*. It is a function: feed it a situation, it returns one number.
- **`s`** — the *state*, i.e. the robot's current situation. For our purposes `s` is essentially the observation from Chapter 02 (joint angles, velocities, uprightness, commanded speed — the several dozen sensor readings the robot reads). "State" and "observation" are close enough to treat as the same thing here.
- **`V(s)`** — the number the forecaster outputs: *given that the robot is in situation `s` right now, how much return do we expect it to collect from here to the end of the episode, on average?*

Where does the forecaster's skill come from? It is trained by hindsight. After every rollout, we know what return each situation *actually* led to. We nudge `V` so that next time it sees a similar situation, its forecast is closer to what really happened. Over thousands of rollouts, `V` becomes a genuinely good predictor of "how good is this spot." It is learned by trial and error, exactly like the policy — just predicting a number instead of choosing an action.

> **Insight: the value function is a baseline, not a goal.**
> The robot is never rewarded for making `V` high. `V` does not change what the robot tries to do. Its entire purpose is to be a *yardstick* — a running estimate of "normal," so the algorithm can tell whether any particular action did better or worse than normal. It is the precise, learned version of the soup-taster's sense of "what my soup usually scores." Without it, "higher than expected" was a hand-wave; with it, "expected" is a number the algorithm can compute.

### Advantage — actual return minus the forecast

**Plain language first.** Now we can finally answer the question the whole rule hinges on: *was this particular action better or worse than what we'd normally expect from that spot?* You answer it by comparing two numbers — what actually happened, against what the forecaster predicted would happen. If the robot was in a situation the forecaster scored at 30, took an action, and the episode went on to earn 38, then that action did **+8 better than expected**. If it earned only 25, the action did **−5 worse than expected**. That gap — actual minus forecast — is the **advantage**.

**Now the formalism.** The advantage is written `A`, and in plain-text house style:

```
advantage  A  =  (actual return from here)  −  V(s)
```

Symbol by symbol:

- **`A`** — short for *advantage*. One number per action. Positive = that action did better than the situation's usual; negative = worse; near zero = typical.
- **`actual return from here`** — the return the robot *really* went on to collect after taking the action, read straight from the rollout (Chapter 02: we recorded every reward, so we can sum them up).
- **`V(s)`** — the forecaster's prediction for that same starting situation `s`: what we *expected* before the action was taken.
- **`−`** — the subtraction that turns two raw numbers ("got 38," "expected 30") into one comparison ("+8, better than usual").

And here is the payoff — the advantage is *exactly* the `(return earned − expected return)` factor that Chapter 03's policy-gradient nudge needed but could only wave at:

```
nudge to θ   ∝   A   ×   (direction in θ that made the taken action more likely)
```

Read that against the Chapter 03 version and notice nothing changed *except* that the vague "return earned − expected return" is now a concrete, computed `A`. The rule is identical: when `A` is positive (the action beat its situation's forecast), step θ to make that action more likely; when `A` is negative, step the other way; when `A` ≈ 0, barely move. The advantage just gives the rule a reliable measuring stick instead of a guess.

> **Insight: why subtract a baseline at all?**
> Subtracting `V(s)` does not change *which way* the rule pushes on average — it changes how *noisy* the push is. Suppose every action in some easy situation earns a return around 40. Without a baseline, every one of those actions looks "good" (return 40, well above zero), so the rule keeps cranking up *all* of them — a wash, plus a lot of noise. With the baseline, the forecaster already expected ~40, so most of those actions score an advantage near zero and are left alone; only the ones that genuinely beat or missed 40 move the dials. Subtracting "what we expected anyway" strips out the part of the return that was never about the action's quality, leaving a much cleaner signal. Cleaner signal means the algorithm can take *bigger* steps safely — which is exactly the goal we set at the top of this chapter.

---

## PPO's fix — only trust an update so far

We now have a clean measuring stick (the advantage). But a clean direction does not by itself solve the steering problem from the top of the chapter: even pointed the right way, *stepping too far* can still over-correct into a fluke. PPO's distinctive contribution is the mechanism that caps the step. This is the idea the algorithm is named for — *proximal*, meaning "stay nearby": **don't let the new policy stray too far from the one that collected the data.**

### The idea in plain words

Picture the update as a knob labeled "how much more likely should we make this action." The advantage says which way to turn it and roughly how hard. PPO adds a hard stop on the knob: *you may increase this action's probability, but only up to a fixed limit — say, 20% more likely than the old policy made it. Past that limit, turning the knob harder buys you nothing.* Same on the downside: you may decrease an action's probability, but only down to a floor — say, 20% less likely — and no further in one update.

Why a hard stop? Because the advantage was measured under the *old* policy — the one that actually ran in the simulator and collected the rollout. The moment you change the policy, that measurement starts going stale; the further you move, the less the old advantage tells you about the new policy. PPO's bet is: trust the advantage for a *modest* move, and refuse to be talked into a big one by a single noisy batch. It is the good driver's firm-but-bounded correction, made into a rule. The "trust-region" name comes from exactly this: there is a *region* around the old policy you're allowed to move within, and PPO won't let the update leave it.

### The idea as a compact expression

PPO implements that hard stop with an operation called **clipping** — literally, "if a number tries to go past a boundary, clamp it to the boundary." Here is the core of PPO's objective in plain-text house style. Do not panic at it; we read every piece immediately after.

```
objective  =  min( r · A ,  clip(r, 1−ε, 1+ε) · A )

   where   r  =  (new policy's probability of the action)
                 ─────────────────────────────────────────
                 (old policy's probability of the action)
```

Symbol by symbol, in words:

- **`A`** — the **advantage** we just built: how much better than expected this action was. The whole objective is this number, scaled. Positive `A` = an action worth encouraging; negative `A` = one worth discouraging.
- **`r`** — the **probability ratio**: how much *more or less* likely the **new** (being-updated) policy makes this action compared to the **old** policy that collected the rollout. `r = 1` means "no change, the new policy agrees with the old." `r = 1.2` means "the new policy now makes this action 20% more likely." `r = 0.8` means "20% less likely." `r` is the precise measure of *how far the update has moved* for this action.
- **`ε`** — *epsilon*, a small fixed number that sets the size of the hard stop. The standard default is **0.2**. It defines the boundary: the update may push `r` up to `1 + ε = 1.2` or down to `1 − ε = 0.8`, and no further.
- **`clip(r, 1−ε, 1+ε)`** — the clamp. It takes `r` and, if it has wandered outside the band `[0.8, 1.2]`, snaps it back to the nearest edge. Inside the band, it leaves `r` untouched. This is the literal hard stop on the knob.
- **`min( … , … )`** — "take the smaller of the two." This is the clever part. One term (`r · A`) is the unclipped reward for moving; the other (`clip(r, …) · A`) is the version with the hard stop applied. Taking the **minimum** of the two means: *the policy is never rewarded for moving past the boundary.* Once `r` exits the band in the helpful direction, the clipped term stops growing, the `min` selects it, and there is no further gradient — no incentive to step further. The update gets all the benefit of moving *to* the boundary and none of the temptation to leap past it.

Put back into plain English, the whole expression says: **"Improve the actions with positive advantage and suppress the ones with negative advantage — but only let the policy's probabilities move so far from where they started. Beyond that limit, stop pushing."** That single guardrail is what lets PPO take steps far bolder than raw policy gradient would dare, *without* the fishtailing. It is the governor on the engine Chapter 03 built — and it is the part rsl_rl runs for us.

> **Insight: clipping is humility about a stale measurement.**
> The deep reason clipping works is subtle and worth sitting with. The advantage `A` was computed from data the *old* policy generated. It is only trustworthy *near* the old policy. Clipping is the algorithm admitting, "I measured this advantage under the old behavior; I don't trust it far from there, so I'll act on it only within a small radius and re-measure before going further." That is also why each batch of data is squeezed for only a handful of small update passes and then thrown away — which is exactly the next idea.

---

## On-policy — why PPO learns only from fresh data

**Plain language first.** Here is a question that trips up every beginner: the robot collected a great rollout an hour ago, full of useful experience — why not keep reusing it to keep learning, the way you might re-read a good textbook? The answer is the same staleness from the clipping insight, pushed one step further. The instant you update the policy, the old rollout describes the behavior of a robot that *no longer exists*. Its advantages were measured against a forecaster and a policy that have both since moved. Learning from it now would be like steering your car using a photo of the road taken a mile back — the information was true once, but acting on it now drives you off course.

So PPO insists on **on-policy** learning: it only ever learns from rollouts collected by the *current* policy. Run the current policy, collect a fresh rollout, do a few small clipped updates, then **throw the rollout away** and collect a brand-new one with the now-slightly-different policy. Old data is not archived or reused; it is discarded after each round. This is the precise meaning of the line you met back in Chapter 02 — *"Nothing is saved from previous rollouts"* — and it is why training spends so much of its time simply collecting experience.

You can see the same logic threaded through everything in this chapter:

- The **advantage** is only valid for the policy that produced it.
- **Clipping** limits how far you move *before* that validity runs out.
- **On-policy** is the rule that says: once you've moved, the old data is spent — go get fresh data.

All three are the same fact wearing three hats: *a measurement of how good an action was is only trustworthy near the policy that made the measurement.* PPO is, end to end, an exercise in respecting that fact while still making progress as fast as possible.

> **Insight: on-policy is the price of trustworthy advantages, and it is expensive.**
> Discarding hard-won experience after a few update passes feels wasteful, and it *is* — on-policy methods like PPO need enormous amounts of fresh simulation, which is the entire reason this project runs **2048 environments in parallel** (Chapter 01). The trade is the same one that runs through the whole series: you pay in sample efficiency to buy stability. PPO chooses stability every time. Given how easily the raw rule fishtails into the ditch, that is usually the right call — and it is why PPO, not the textbook policy gradient, is the default workhorse in robotics RL.

---

## Worked example: the walking curve, climbing smoothly

Theory is cheap; here is PPO actually working, on the same baseline G1 walker from Chapters 01–03.

Recall the setup: 2048 environments in parallel, a policy network of roughly 200,000 weights, a value-function network forecasting alongside it, episodes capped at 1000 timesteps, and a reward that pays for matching the commanded speed, staying upright, and moving cleanly. Training ran for about **2050 iterations** — that is 2050 rounds of *collect a fresh rollout → compute advantages with the forecaster → take a few small clipped updates → discard the rollout → repeat*. Every one of those rounds is the full PPO loop you just learned, executed by rsl_rl.

The thing to look at is the *shape* of the reward curve over those iterations — and the headline is in one word: **smooth.**

- **Early on,** return per episode is very low (the robot is mostly falling), but it turns upward and climbs *steadily*, not in violent jumps. That steadiness is clipping doing its job: even when an early rollout contains a lucky robot that stayed up by accident, clipping refuses to let the update lunge toward that fluke. The curve rises without spiking and crashing.
- **Through the middle,** the climb is brisk and *monotone* — it mostly goes up, rarely lurching backward. A raw, unclipped policy gradient at an aggressive learning rate would show exactly the fishtailing we warned about: the curve would leap up, over-correct, collapse, recover, collapse again. PPO's curve does not do that. Each step was bold enough to make fast progress and bounded enough never to over-correct into the ditch.
- **Late,** around iteration 2050, the curve flattens near a return of roughly **50**, with episodes routinely reaching **995–1000** of the 1000 timesteps. Most actions are now "typical," advantages hover near zero, and the clipped updates barely move θ. Learning has settled.

That smooth, stable, mostly-monotone climb from face-plants to a full-length walk **is what PPO working looks like.** When you reach Chapter 05 and start reading these curves yourself, a clean upward sweep like this is the signature of a healthy run; a curve that spikes and collapses is the signature of the instability PPO was built to prevent. The reason the baseline curve looks as well-behaved as it does is precisely the advantage-plus-clipping machinery of this chapter, running quietly inside rsl_rl on every one of those 2050 iterations.

> **Insight: PPO is invisible when it works.**
> Notice that nothing in the *result* — a robot walking — tells you PPO was involved. The algorithm's fingerprint is not in the gait; it is in the *shape of the learning curve*. PPO's whole contribution is making the path from random to skilled **smooth and reliable** instead of a gamble. That is why the next chapter is about reading these curves: the curve is where the optimizer's behavior becomes visible, and learning to read it is learning to see whether the machinery of this chapter is doing its job.

---

## What you now understand

- The **raw policy-gradient rule is unstable**: like over-correcting a steering wheel, a step taken too far over-corrects toward a noisy fluke and can wreck a policy that was improving. PPO's entire purpose is to take steps that are *as bold as possible without overshooting*.
- The **value function `V(s)`** is a second, learned network — a forecaster that predicts the expected total future return from a situation `s`. It is a *baseline*, a yardstick for "normal," never a goal the robot chases.
- The **advantage `A = (actual return) − V(s)`** measures whether an action did better or worse than that situation's forecast. It is the concrete, computed version of Chapter 03's hand-wavy "higher-than-expected return," and subtracting the baseline strips noise out of the learning signal so bigger steps are safe.
- **PPO** caps how far each update may move via **clipping**: the probability ratio `r` (new-policy likelihood ÷ old-policy likelihood) is held inside a band `[1−ε, 1+ε]` (typically ε = 0.2, the standard default), and the `min(...)` construction ensures the policy is *never rewarded for moving past that boundary*. This is the **trust-region** idea — stay near the policy that collected the data.
- PPO is **on-policy**: it learns only from rollouts the *current* policy collected, does a few small clipped updates, then **discards** that data and gathers fresh experience — because an advantage is only trustworthy near the policy that produced it. The price is needing huge amounts of simulation (hence 2048 parallel environments); the payoff is stability.
- **PPO is machinery this repo *uses*, not something we wrote** — it ships inside the **rsl_rl** library under `mjlab`. We configured it and supplied the rewards; the clipping, the advantage estimation, and the value-function training are all rsl_rl's doing.
- Concretely, PPO's signature is the **smooth, mostly-monotone climb** of the baseline walker's reward curve over ~2050 iterations to a return near 50 — a clean upward sweep instead of the spike-and-collapse a raw, unclipped update would produce.

You now understand not just *how* a policy improves (Chapter 03) but *how it improves safely* (this chapter) — the difference between an idea and the workhorse algorithm a real project trains with. Everything from here on is downstream of a PPO run: the curves it produces, the gaits it discovers, and the ways the reward you wrote can lead it somewhere you never intended. The next step is to learn to *read* a training run — to look at the reward and episode-length curves PPO produces and tell, at a glance, whether learning is healthy, stalled, or quietly lying to you.

Continue to [Chapter 05 — Reading the Training](05-reading-the-training.md).
