# Chapter 02 — The Vocabulary

*Chapter 01 gave you the three big ideas: a simulator that stands in for a real robot, parallel environments that multiply experience, and a learning approach that substitutes a goal for hand-coded joint angles. Now comes the vocabulary — the eight precise terms that let us talk about the practice loop without hand-waving. By the end of this chapter, phrases like "the policy collects a rollout and the return improves" will mean something concrete to you.*

---

## Why vocabulary matters here

In casual speech, "the robot practices" is good enough. But once you want to understand *why* learning sometimes stalls, *why* a reward that seems reasonable produces bizarre behavior, or *why* one experiment differs from another, you need sharper tools. The eight terms in this chapter are the units of measurement for the rest of the series.

We will define each one in the same order the computation actually happens: first, set up the world the robot lives in; then describe what the robot reads from that world; then what it outputs; then what brain decides the output; then how that brain is scored; then what one attempt looks like; then a batch of attempts; then the single number that training is ultimately chasing.

The running example throughout is the G1 walking experiment — the same one whose results appear in Chapters 05–10.

---

## Environment

The **environment** is the simulated world the robot inhabits. It is the thing that surrounds the robot, receives what the robot does, and sends back what the robot can observe and how well it did.

Think of the environment as the game board: it sets the physical rules (gravity, friction, the shape of the floor), tracks the state of every joint and the position of the body in space, and runs the clock that ends an attempt.

In the G1 walking experiment, the environment is a flat virtual plane — no hills, no obstacles — with the G1 standing at the center. At every fraction of a second, the environment does three things in sequence: it reads the robot's joint-angle commands, advances the physics by one small time step, and reports back what happened.

> **Insight: the environment is the teacher, but it does not explain anything.**
> The environment does not say "you fell because your left knee buckled." It only reports numbers. The learning algorithm must infer what went wrong by comparing those numbers across thousands of tries. The richer and more honest those numbers are, the better the algorithm can learn — which is exactly why reward design is the most consequential engineering decision in this whole process (a theme that runs through Chapters 06–14).

---

## Observation

An **observation** is the snapshot of numbers the robot reads from the environment at one instant in time.

If the environment is the game board, the observation is the robot's senses — everything it is allowed to know about the state of the world right now.

For the G1 walker, one observation is a vector of 47 numbers — here is what they capture:

| Group | What it measures | Approximate count |
|---|---|---|
| Joint angles | Where each motor is currently pointing | 23 |
| Joint velocities | How fast each motor is moving | 23 |
| Body orientation | Which way gravity is pulling relative to the body frame | ~3 |
| Commanded velocity | The forward speed the robot has been asked to reach | 1+ |

The robot does not get to see its absolute position in the room, or which direction it is facing in world coordinates. It only reads local, body-relative signals. This is a design choice: observations that are relative to the robot's own body generalize better across orientations, and it mirrors what a real robot's onboard sensors would actually measure.

Every 20 milliseconds (50 times per second), a fresh observation arrives. The robot must decide what to do based only on this snapshot.

---

## Action

An **action** is the set of numbers the robot sends back to the environment after reading an observation.

In the G1 experiment, the action is a vector of **23 numbers** — one target joint angle for each of the 23 motors in the robot's body (knees, hips, ankles, shoulders, elbows, wrists). The environment takes these target angles and instructs each motor to move toward them, applying realistic motor forces and physical constraints.

The action does not command acceleration, force, or torque directly. It commands a target position. The physical simulation works out the forces needed to try to reach that position within the next time step. This matters: the robot cannot teleport a joint to its target — momentum and joint limits mean the actual angle lags behind the command, just as real motors do.

The action-observation loop runs at 50 Hz: observation in, action out, physics advances, new observation in — 50 times every simulated second.

> **Insight: the action space is intentionally low-level.**
> Nobody gave the robot a "take a step" command. The 23-dimensional action is literally a list of where each joint should point. Making that list add up to a walk is the entire problem. This is why 2048 parallel environments and billions of timesteps are necessary: discovering that certain sequences of joint-angle vectors produce locomotion, from a cold start with random numbers, is not a simple search problem.

---

## Policy

The **policy** is the function — implemented as a neural network — that maps each observation to each action. It is the robot's brain.

Formally, the policy takes in the current observation and outputs an action:

```
policy π_θ : observation oₜ → action aₜ
```

Read that as: "the policy π with weights θ, given observation oₜ at time t, produces action aₜ." The subscript θ (theta) just names the internal numbers (weights) of the neural network. When training adjusts the policy, it is adjusting θ.

In the G1 experiment, the policy is a fully connected neural network with three hidden layers of 512, 256, and 128 units (narrowing as it goes deeper) — roughly 200,000 adjustable weights in total. Before training, all weights are random, so the policy outputs random joint angles and the robot falls immediately. Training's job is to find values of θ that make the policy output good actions.

The policy has no memory of past timesteps (it is called *memoryless* or *Markovian*: the output depends only on the current observation, not the history). Walking turns out to be learnable under this constraint — the current body velocity and joint state carry enough information about what came just before. (One nuance worth knowing is coming: in practice the policy adds a small amount of randomness to its outputs so it can explore different actions and discover better ones — Chapter 03 explains why.)

> **Insight: the policy is what training is improving.**
> Every data point collected during practice, every reward signal, every learning update — all of it exists to improve θ. The observation space, action space, and environment are fixed. Only θ changes. When someone says "the robot is learning," what they mean precisely is "θ is being adjusted."

---

## Reward

The **reward** is a single scalar number that the environment emits after each action, indicating how well the robot did in that one timestep.

This is the score. It is the only feedback signal the learning algorithm has. There are no labels, no demonstration of correct behavior, no commentary about what went wrong. Just a number: higher is better.

In the G1 walking experiment, the reward at each timestep is a sum of several components:

```
r_t  =  w₁ · track_lin_vel      # Did the robot match the commanded speed?
       + w₂ · upright            # Is the torso roughly vertical?
       + w₃ · air_time           # Did feet spend time off the ground (a proxy for dynamic walking)?
       − w₄ · action_rate_l2     # Penalty for jerky, high-energy joint changes
       + ...                     # (a few more terms, similarly weighted)
```

Each term is a carefully chosen proxy for part of what "walking well" means. The weights (w₁, w₂, …) control the relative importance of each component.

The key word is *proxy*. No set of scalar reward terms fully captures what a good walking robot is. They are approximations. When a proxy is wrong — or when a policy discovers a high-scoring behavior that the proxy didn't anticipate — the robot may learn something other than what the designer intended. That gap between the proxy and the true goal is the source of almost every surprise in this series.

---

## Episode

An **episode** is one complete run of the robot, from the moment it resets to the moment the run ends.

Episodes end in exactly two ways:

1. **Time-out.** The robot survived long enough — in the G1 experiment, 1000 timesteps, which is 20 simulated seconds. This counts as success.
2. **Early termination.** The robot's torso hit the ground (its height dropped below a threshold). The attempt is cut short at that moment.

The **episode length** — how many timesteps before one of these endings — is a direct measure of progress. At the very start of training, the episode length might be 10–30 timesteps: the robot falls almost immediately. By the end of training, episodes routinely reach 995–1000 timesteps: the robot is staying upright for the full attempt.

Each of the 2048 parallel environments runs its own independent episode. When one finishes, that environment resets and starts a new episode immediately, without waiting for the others.

---

## Rollout

A **rollout** is a batch of experience collected by running the current policy for a fixed number of timesteps across all parallel environments.

Every time the learning algorithm runs the policy to gather data, it is collecting a rollout. A rollout contains, for each timestep in each environment, four things:

- The observation the robot saw: `oₜ`
- The action the policy chose: `aₜ`
- The reward it received: `rₜ`
- Whether the episode ended: a boolean "done" flag

In the G1 experiment, one rollout covers **24 timesteps × 2048 environments = 49,152 transitions**. That is the unit of experience the learning algorithm gets to work with before updating the policy.

After each rollout, the policy is updated, the rollout is discarded, and a new one is collected. Nothing is saved from previous rollouts — the learning algorithm only ever uses experience from the *current* policy. (Why this matters becomes clear in Chapter 04, when we discuss what "on-policy" means.)

---

## Return

The **return** is the total reward accumulated over an episode.

If a rollout is a batch of experience, the return is the answer to the question the learning algorithm ultimately cares about: "over an entire episode, how much reward did this robot earn?"

The simplest version is just the sum of all per-timestep rewards:

```
return = r₀ + r₁ + r₂ + … + rₜ
```

In practice, we use a *discounted* return: rewards earned further in the future count slightly less than rewards earned now. The discount factor is written γ (gamma) and is typically set close to 1 (for example, γ = 0.99):

```
return = r₀ + γ·r₁ + γ²·r₂ + γ³·r₃ + …
```

Read each term as: the reward at the k-th step in the future is multiplied by γᵏ, which shrinks toward zero as k grows. A reward 100 steps away contributes γ¹⁰⁰ ≈ 0.366 of what it would at γ = 1.00 — present rewards are weighted more heavily than distant ones.

Why discount? Two reasons. Practically, it prevents returns from growing unbounded in long episodes. Conceptually, it captures the idea that a reward you earn sooner is more certain than one you might earn in a future that may not arrive (the episode might end early).

> **Insight: what the learning algorithm is actually optimizing.**
> The policy does not directly maximize any one reward term. It does not "try to stay upright." It maximizes the *expected discounted return* — the total accumulated score it expects to earn, on average, starting from the current moment. That is the goal in its most precise form. Everything else — the environment, the observation, the action, the reward — exists in service of this one number. Understanding that the target is *expected future return, not current score* is the key to understanding why some reward designs lead to surprising long-horizon strategies.

---

## Putting the eight terms together

Here is the full loop, one cycle:

```
Environment resets  →  initial observation o₀
Policy reads o₀     →  outputs action a₀
Environment applies a₀, advances physics,
  emits reward r₀, emits next observation o₁
Policy reads o₁     →  outputs action a₁
...
Episode ends (time-out or fall)
```

Repeat that loop across 2048 environments for 24 steps each, and you have one rollout. Sum the rewards across the full episode (with discounting), and you have the return. Average the return across many episodes, and you have the learning curve metric you will see plotted in Chapter 05.

The thing training is trying to do is simple to say: find the policy weights θ that maximize the expected discounted return. Everything in Chapters 03 and 04 is about the mechanics of how that search actually works.

---

## What you now understand

- The **environment** is the simulated world that surrounds the robot — it receives actions, enforces physics, and emits observations and rewards.
- An **observation** is the snapshot of sensor numbers the robot reads each timestep (joint angles, velocities, commanded speed — 47 numbers for the G1 walker).
- An **action** is the vector of joint-angle targets the robot outputs (23 numbers, one per motor).
- The **policy** is the neural network that maps observations to actions; its adjustable weights θ are exactly what training improves.
- A **reward** is a scalar score emitted after each action — a proxy, always imperfect, for what the designer wants the robot to do.
- An **episode** is one complete run from reset to time-out or fall; episode length is a direct measure of survival skill.
- A **rollout** is a batch of transitions (observation, action, reward, done) collected from all parallel environments across some number of steps — the data the learning algorithm works with.
- The **return** is the discounted sum of rewards over an episode: the single number the policy is ultimately trying to maximize.

These eight terms are the grammar of everything that follows. The next chapter asks the question that comes after: now that we know what the policy, the reward, and the return are — how does the policy actually improve?

Continue to [Chapter 03 — Trial and Error Made Precise](03-trial-and-error.md).
