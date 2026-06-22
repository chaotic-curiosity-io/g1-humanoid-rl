---
layout: default
title: Teaching a Humanoid to Move — a zero-background RL curriculum
---

# Teaching a Humanoid to Move — a zero-background RL curriculum

A human-shaped robot taught itself to walk inside a computer — with no rules
about how to move, no programmer specifying joint angles, no motion-capture
suit. Just a score, millions of practice attempts, and a physics simulator
running faster than real time. Then it taught itself to run, cartwheel,
backflip, and get up off the floor.

This is a fifteen-chapter curriculum that takes you through that whole arc
from the beginning, with **zero background in robotics or machine learning
assumed**. Every term is defined the first time it appears. Every result is
shown on video. The failures stay failures — they carry as much of the lesson
as the wins.

## Start here

**[Chapter 1 — The Big Picture](reports/01-the-big-picture.md)** is the
on-ramp. Read the chapters in order: each one builds on the concepts the
previous one introduced, and the series is designed as one cohesive arc, not
a collection of standalone reports.

---

## The seven parts

### Part I — What is this even?
*Chapters 1–2*

The first two chapters answer the questions a newcomer brings before anything
else can land: what kind of project is this, and what are the eight words you
need before you can read further?

Start: [Chapter 1 — The Big Picture](reports/01-the-big-picture.md)

---

### Part II — How learning actually works
*Chapters 3–5*

Three chapters on the engine under the hood: why trial and error can produce a
skill from nothing, how the algorithm (PPO) turns that into a practical
training loop, and how to read the numbers a training run produces.

Start: [Chapter 3 — Trial and Error Made Precise](reports/03-trial-and-error.md)

---

### Part III — First real skill: walking
*Chapters 6–8*

The concrete experiment — watch the robot go from random collapse to smooth
walking, verify the result is reproducible, then run your first controlled
experiment by pulling one lever.

Start: [Chapter 6 — Watching It Walk](reports/06-watching-it-walk.md)

---

### Part IV — Reward engineering
*Chapters 9–11*

Three chapters on what happens when the score measures a *proxy* for the
behavior you want rather than the behavior itself — including a robot that
earned maximum points by diving face-first into the ground.

Start: [Chapter 9 — The Running Dive](reports/09-the-running-dive.md)

---

### Part V — Imitation
*Chapters 12–13*

A completely different training paradigm: instead of writing score terms, show
the robot a recording of a cartwheel (or a backflip) and reward it for
matching each frame. Easier said than done.

Start: [Chapter 12 — Imitation and the Cartwheel](reports/12-imitation-and-the-cartwheel.md)

---

### Part VI — Building a task from nothing
*Chapter 14*

No reference motion, no velocity command — just a score you design from
scratch for a behavior the robot has never seen: getting up off the floor from
any fallen pose. Four reward iterations before it actually works.

Start: [Chapter 14 — Building Get-Up from Scratch](reports/14-building-get-up-from-scratch.md)

---

### Part VII — Synthesis
*Chapter 15*

The capstone names the single principle every previous chapter was circling
and shows why it is a discipline, not a trick.

Start: [Chapter 15 — Reward Engineering as Craft](reports/15-reward-engineering-as-craft.md)

---

## Appendices

- [Methods and techniques reference](reports/methods-reference.md) — reward
  terms, terminations, the retargeting pipeline, curricula, recording gotchas,
  helper scripts, and ops notes: one lookup surface for the full toolkit.
- [Cartwheel journey](cartwheel-journey.md) — the full iteration-by-iteration
  log of the cartwheel training campaign, for readers who want the unabridged
  story behind Chapter 12.

---

*A [Chaotic Curiosity](https://chaoticcuriosity.io) project. Built on
[mjlab](https://github.com/mujocolab/mjlab) (an Isaac-Lab-style RL API on
MuJoCo-Warp), trained on an NVIDIA DGX Spark. The full method, all the plots, and
the training runs behind these reports live in the
[repository](https://github.com/chaotic-curiosity-io/g1-humanoid-rl).*
