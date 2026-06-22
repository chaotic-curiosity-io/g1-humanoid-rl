# Chapter 03 — Trial and Error Made Precise

*Chapter 02 ended on a question. We now know what the policy is (the neural network brain, with adjustable weights θ), what the reward is (the per-timestep score), and what the return is (the total score over an episode). The thing training wants is also clear: find the weights θ that earn the most return. But knowing the goal is not the same as knowing how to reach it. This chapter is about the "how" — the single idea that turns a pile of falling robots into a walking one. It is the first genuinely theoretical chapter in the series, and the most important. Take it slowly.*

---

## The problem, stated plainly

At the start of training, the policy's weights θ are random numbers. The robot reads an observation, the random network spits out 23 random-ish joint angles, and the robot crumples. It does this 2048 times in parallel, and every copy crumples.

Now: how do you improve from there?

You cannot just *tell* the robot the right joint angles — if you knew those, you would have hand-programmed the gait and skipped all of this (Chapter 01). You cannot label the experience "correct" or "incorrect," because nobody knows what correct looks like until the robot finds it. All you have is the score. The robot tried *something*, and the environment handed back a number.

So the only information available is: *here is what the robot did, and here is how much return it earned for doing it.* The entire art of reinforcement learning is squeezing improvement out of exactly that — and nothing more. The name for squeezing improvement out of that is **trial-and-error learning**: try things, notice which tries earned more return, and shift your future behavior toward those.

That is the whole idea in one sentence. The rest of this chapter makes it precise, gently, one layer at a time.

---

## The analogy: adjusting a recipe after each taste

Imagine you are inventing a soup, with no recipe to follow.

You have a row of dials in front of you — one for salt, one for heat, one for cooking time, one for how much you blend it, a few dozen more. Each dial is currently set to a random position. You turn the dials to *some* setting, cook the soup, and taste it. The taste is a single judgment: a number from 1 to 10. That number is the only feedback you get. Nobody tells you "add more salt" — you just get the score.

How do you improve?

You cook many batches, each with the dials set slightly differently, and you pay attention. The batches that scored an 8 had the salt dial a little higher than average. The batches that scored a 3 had the heat cranked up. You do not *know* the perfect recipe, but a pattern is emerging: *nudge the salt dial up a touch, nudge the heat dial down a touch.* So you do — a small nudge, not a wild swing, because one good batch might have been luck. Then you cook again, taste again, and nudge again.

Repeat that loop a few thousand times and a real recipe emerges from pure tasting. You never reasoned about chemistry. You just kept nudging the dials toward the settings that, on average, tasted better.

The policy learns walking in *exactly* this way. Hold onto this mapping, because every piece of it has a precise counterpart:

| In the soup analogy | In the robot |
|---|---|
| The row of dials | The policy's weights, **θ** — roughly 200,000 of them |
| A dial setting | One specific value of θ (one "version" of the brain) |
| Cooking one batch | Running one episode — letting the policy act until the robot falls or times out |
| The taste score (1–10) | The **return** — total reward earned over the episode |
| "Nudge the salt up, the heat down" | The **policy gradient** (defined below) |
| One round of nudging the dials | One **gradient step** (defined below) |

The robot has more dials than you have soup dials, and it tastes thousands of batches at once instead of one. But the loop is the same: try, score, nudge toward what scored higher, repeat.

> **Insight: there is no teacher, only a taster.**
> Nobody ever tells the policy the right action — not once, not ever. The environment is a taster, not a cook: it scores what the policy already did, but it never demonstrates what it should have done. Every other supervised-learning setup you may have heard of (label this photo "cat") hands the learner the right answer. Reinforcement learning never does. This is why it needs *so much* experience — billions of timesteps — to work: it is reverse-engineering "what should I do" out of nothing but "here is what that try was worth."

---

## From "nudge the dials" to the precise idea

Now we sharpen the analogy into the actual rule. We will do it in plain words first, then write the one expression that captures it, then re-read that expression word by word. No symbol will appear without being named in English.

### Step 1 — actions have probabilities, not certainties

One subtlety we glossed over: the policy does not output a single fixed action for a given observation. It outputs a *probability distribution* over actions — a slightly random spread. For the same observation, it might lean toward "left knee to 14°" but occasionally try 11° or 17° instead. This built-in randomness is not a flaw; it is the *trying* part of trial and error. Without it, the policy would do the exact same thing forever and never discover anything better. The randomness is how it samples new dial settings to taste.

So at every timestep the policy is really answering: *given what I see, how likely should each possible action be?* Learning, then, is not about flipping actions from "wrong" to "right." It is about adjusting those likelihoods — making some actions a little more probable and others a little less.

### Step 2 — the rule, in one sentence

Here is the heart of the entire chapter. After collecting a rollout (a batch of episodes, from Chapter 02) and seeing how much return each one earned, the learning rule is:

> **Increase the probability of the actions that led to higher-than-expected return. Decrease the probability of the actions that led to lower-than-expected return. Leave roughly average actions roughly alone.**

That is it. That single rule, applied over and over, is what carries the robot from random twitching to a smooth walk. Read it once more and notice the phrase *higher-than-expected*: the comparison is against what the policy normally earns, not against zero. An action is "good" only relative to the policy's own current baseline. (Building a precise estimate of "expected" is the job of the *value function*, which is the subject of Chapter 04. For this chapter, "higher than usual / lower than usual" is enough.)

### Step 3 — the same rule, written as an expression

The family of algorithms that implement this rule is called **policy gradient**. The word *gradient* is borrowed from calculus, but you do not need calculus to read it. A gradient is simply *the direction to nudge the dials that increases the score fastest* — the multi-dial generalization of "salt up, heat down." Here is the policy-gradient idea, written in the plain-text house style of this series:

```
nudge to θ   ∝   (return earned − expected return)  ×  (direction in θ that made the taken action more likely)
```

Every symbol, in words:

- **`nudge to θ`** — the change we apply to the policy's weights this round. θ (theta) is the same bag of ~200,000 weights from Chapter 02; nudging it is literally editing the robot's brain a tiny bit.
- **`∝`** — "is proportional to." The size of the nudge scales with the right-hand side; we also multiply by a small *learning rate* (a tiny fixed number, like 0.001) so each nudge stays gentle. This is the "small nudge, not a wild swing" from the soup.
- **`return earned − expected return`** — how much *better than usual* this episode turned out. Positive when the try beat the policy's normal performance, negative when it fell short, near zero when it was typical. This number sets both the **sign** (push the action's probability up or down) and the **strength** (a big surprise gets a big nudge) of the update.
- **`direction in θ that made the taken action more likely`** — the gradient proper: of all the ways we could tweak the 200,000 weights, this is the combination that would make the policy *more inclined* to choose the action it actually chose. The neural network can compute this direction exactly, for free, as a by-product of having produced the action.

Multiply those two together and you get the rule in Step 2 made mechanical: when an action beat expectations, the first factor is positive, so we step θ *along* the direction that makes that action more likely — its probability goes up. When an action fell short, the first factor is negative, so we step the *opposite* way — its probability goes down. Average that nudge over all 49,152 transitions in the rollout (Chapter 02's rollout size: 24 steps × 2048 environments), and you have one honest, denoised estimate of which way to nudge every dial.

### Step 4 — one gradient step, defined

Applying that averaged nudge to θ exactly once is a single **gradient step** (also called an optimization step, or just an *update*). It is the atomic unit of learning — the robot equivalent of cooking one round of batches and adjusting the dials once. Training is nothing but gradient step, after gradient step, after gradient step:

```
collect a rollout  →  compute the nudge  →  apply one gradient step to θ  →  repeat
```

The baseline G1 walker ran this loop for roughly **2050 iterations** — that is 2050 gradient steps. Each step moved θ by a hair. Two thousand hairs later, the random brain had become a walking one.

> **Insight: a gradient step is small on purpose, and that is the whole safety story.**
> Why nudge gently instead of leaping straight to whatever scored highest? Because the score from any single rollout is noisy — a robot might earn high return partly by luck (a fortunate stumble that happened to stay upright). If you slammed the dials all the way toward one lucky batch, you would overfit to that fluke and likely *destroy* a policy that was otherwise improving. Small steps average the luck out: over many rollouts, the genuine signal ("salt up") accumulates while the random noise ("this one batch got lucky") cancels. The price of small steps is that learning is slow and needs enormous amounts of experience. The reward is that it is *stable*. Chapter 04 (PPO) is, at its core, an elaborate mechanism for making these steps as large as possible **without** ever leaping so far that a lucky fluke wrecks the policy.

---

## The loop, as a diagram

Here is the trial-and-error loop drawn out — the same shape as the soup loop, now in the robot's terms. This is the engine room of the whole project:

```
        ┌──────────────────────────────────────────────────────────┐
        │                                                          │
        ▼                                                          │
  ┌───────────┐     run the current      ┌──────────────────────┐  │
  │  policy   │ ───  policy in 2048   ──► │   rollout: a batch   │  │
  │  weights  │      environments         │   of episodes, each  │  │
  │    θ      │      until they fall      │   with its RETURN     │  │
  └───────────┘      or time out          └──────────┬───────────┘  │
        ▲                                            │              │
        │                                            ▼              │
        │                                  ┌──────────────────────┐ │
        │  apply ONE gradient step:        │  compare each try's   │ │
        │  nudge θ so good actions          │  return to the        │ │
        └──── become more probable,    ◄─── │  policy's expected    │─┘
              bad actions less          ​    │  return  (better?     │
                                             │  worse? typical?)     │
                                             └──────────────────────┘

   try  ──►  score each try  ──►  nudge toward what paid off  ──►  try again
```

Read the bottom line as the one-sentence summary of reinforcement learning: **try, score each try, nudge toward what paid off, try again.** Everything technical in this series — every reward term, every algorithm, every plot — is machinery bolted onto that loop.

---

## Worked example: the walking policy's return going up

Theory is cheap. Here is the loop actually running, on the baseline G1 walker from this project.

Recall the setup from Chapters 01–02: 2048 environments in parallel, the policy a neural network of ~200,000 weights, an episode capped at 1000 timesteps (20 simulated seconds), and a reward that pays for matching the commanded speed, staying upright, and moving cleanly. Training ran the gradient-step loop above for about 2050 iterations.

Watch what trial-and-error learning does to the return over those iterations:

- **Iteration 0 (random θ).** The policy outputs near-random joint angles. The robot falls in a fraction of a second — episodes end after 10 to 30 timesteps. Return per episode is very low. There is almost no signal yet, just 2048 different ways of falling over. But even here, the rule bites: among 2048 falls, the handful that happened to stay upright a few steps longer earned slightly more return than expected, and the gradient step nudged θ a hair toward whatever those robots did.

- **Early iterations (the return starts climbing).** Those tiny nudges compound. Because staying upright is worth reward at *every* timestep, an action that keeps the robot standing one step longer earns more total return — so the rule relentlessly increases the probability of "don't fall" actions. Episode length climbs from tens of steps toward hundreds. The robot is not walking yet; it is mostly learning to *not immediately collapse*. But the return curve has unmistakably turned upward.

- **Middle iterations (walking emerges).** Now that the robot survives long enough to actually go somewhere, the `track_lin_vel` part of the reward (matching the commanded speed, from Chapter 02) starts to dominate the gradient. Actions that produce forward progress at the right speed beat expectations and get amplified; flailing-in-place actions fall short and get suppressed. A gait — a repeating step pattern nobody designed — emerges purely because stepping forward earns more return than not. The return rises steeply.

- **Late iterations (the curve flattens).** By around iteration 2050, the policy is walking smoothly at the commanded speed, episodes routinely reach **995–1000** of the 1000 timesteps (the robot survives the full attempt), and the return per episode has climbed to roughly **50** and stopped rising much. Further nudges produce smaller and smaller gains: the dials are near a good setting, so most tries are now "typical," the *return − expected* factor hovers near zero, and the gradient step barely moves θ. Learning has converged.

That entire arc — from 10-step face-plants to 1000-step smooth walking — was produced by *one rule applied 2050 times*: nudge θ toward the actions that earned higher-than-expected return. No gait was choreographed. No joint angle was ever specified by a human. The walk is the fixed point that trial and error settled into.

> **Insight: the reward never mentions "walking."**
> Look back at the reward from Chapter 02 — it pays for speed, uprightness, and smoothness, term by term. It never says "walk." Walking is simply the strategy the policy discovered for collecting those rewards, because among all the things a two-legged body can do, a coordinated gait is what maximizes "go the commanded speed while staying upright and smooth." The gait fell out of the score-chasing. This is the most important and most dangerous fact in the whole field: the policy optimizes the *reward you wrote*, not the behavior you *pictured*. When those two come apart — when a policy finds a high-return behavior you never intended — you get the diving robot and the lying cartwheel scorer that later chapters are built around. Hold that thought; it is the spine of everything from Chapter 06 onward.

---

## A caveat, kept honest

The picture above is the *idea* of policy gradient, and it is true — but the raw version described here is rarely used as-is, because it is fragile. Two problems lurk:

1. **The "expected return" baseline is hand-wavy.** We kept saying "higher than expected" without saying how the algorithm *knows* what to expect. Estimating that baseline well is what separates a policy that learns from one that thrashes. The tool for it — the **value function** — is Chapter 04.
2. **Gentle steps are slow; bold steps are unstable.** We have a genuine tension: small nudges waste experience, big nudges risk overshooting into a fluke and wrecking the policy. The algorithm this project actually uses, **PPO**, is precisely the engineering compromise that lets each step be as bold as possible *without* ever leaping into the ditch.

So this chapter gives you the engine; Chapter 04 gives you the governor and the steering that make the engine usable in practice. Everything in Chapter 04 is built directly on the rule you just learned — it refines this rule, it does not replace it.

---

## What you now understand

- **Trial-and-error learning** is the only thing available when there is no teacher: try actions, observe how much return each try earned, and shift future behavior toward the tries that paid off. The environment scores what you did; it never demonstrates what you should have done.
- The policy outputs a *spread* of possible actions, not one fixed action — and that built-in randomness is the "trying" that lets it discover anything new.
- The **policy-gradient** rule, in one sentence: increase the probability of actions that earned **higher-than-expected** return, decrease the probability of actions that earned lower, leave typical actions alone. The size and sign of each nudge come from how far the return beat (or missed) the policy's usual performance.
- A **gradient step** (optimization step / update) is one application of that averaged nudge to the weights θ — the atomic unit of learning. Training is just thousands of these in a row (about 2050 for the baseline walker).
- Steps are deliberately *small* so that luck averages out and a single fortunate rollout cannot wreck the policy — the trade is slow-but-stable learning, which is why so much experience is needed.
- Concretely, that one rule applied ~2050 times carried the G1 from 10-step collapses to a smooth, full-length 1000-step walk earning a return near 50 — with no gait ever programmed, only rewarded.

You now hold the core idea of how a policy improves. But you also saw its two soft spots: the vague "expected return" baseline, and the small-step-versus-big-step tension. The next chapter introduces the algorithm that fixes both — the one this project actually trains with.

Continue to [Chapter 04 — PPO for Novices](04-ppo-for-novices.md).
