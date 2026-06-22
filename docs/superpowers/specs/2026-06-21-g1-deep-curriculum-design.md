# Design: G1 Deep Curriculum — zero-to-everything, single end-to-end series

**Date:** 2026-06-21
**Branch:** `g1-deep-curriculum`
**Status:** Approved (verbal), pending implementation plan
**Supersedes:** the topical-reports structure from
`2026-06-19-g1-skills-curriculum-master.md` (those reports become source
material; the trained results they document are reused, not re-run)

## Goal

Replace the current "hands-on, zero-background walkthrough of the walking policy"
(and the topical multi-track reports that grew around it) with a **single,
cohesive, end-to-end curriculum** that takes a reader **from zero experience to a
deep understanding of everything this repo did** — no jargon, method, or insight
left unturned.

A true beginner should be able to start at chapter 1 knowing nothing and finish
the final chapter understanding, at a deep level: what reinforcement learning is
and how it actually works (including the policy-gradient/PPO machinery this repo
runs), every skill this project trained (walking, running, spin/backward gaits,
cartwheel, backflip, get-up), every reward hack we hit and why, and the real code
we wrote (the get-up task, the backflip landing reward).

## Audience & voice

Zero robotics/ML background, same as the existing series — but the *ceiling* is
much higher. Every term is defined the first time it appears; concrete analogies
lead; the reader climbs every rung in order. The difference from the current
reports: we go all the way down to the real machinery (RL theory, real reward
code, the reward-shape math) — but never skip a rung to get there.

**Handling technical depth for a novice — intuition first, formalism second,
always both:**
1. Plain-language version with an analogy.
2. The precise version (a formula or code block) introduced gently and
   **immediately re-explained in words**, every symbol named.
3. A worked example drawn from *this repo* so it is never abstract.

Equations and code appear, but always sandwiched in plain language — optional
depth for the curious, never a wall the novice hits.

## Spine: a concept ladder (zero → deep)

Ordered by **idea difficulty**, not chronology. Each chapter builds strictly on
the prior ones; the project's real work (and real code) supplies the worked
examples that teach each rung. Full depth means the RL-theory spine (Part II)
comes up front, before the worked examples that rely on it.

### The chapter map — 15 chapters across 7 parts

**Part I — What is this even? (zero assumptions)**
1. `01-the-big-picture` — a robot that *learns* to move vs. one that's programmed;
   the simulator; why thousands of robots train in parallel.
2. `02-the-vocabulary` — observation, action, policy, reward, episode,
   environment, rollout — each defined once, concretely.

**Part II — How learning actually works (the RL-theory spine)**
3. `03-trial-and-error` — the policy-gradient idea: nudge toward what paid off.
4. `04-ppo-for-novices` — what PPO's objective optimizes, why clipping makes it
   stable, advantage estimation — the actual machinery this repo runs, derived
   gently for a beginner.
5. `05-reading-the-training` — reward curves, episode length, convergence; what
   the numbers do and don't tell you.

**Part III — First real skill: walking** *(absorbs 01, 02, 03)*
6. `06-watching-it-walk` — flailing → walking, frame by frame; the
   velocity-tracking task and its reward terms.
7. `07-proving-its-real` — reproducing the benchmark; what every reward term
   actually does.
8. `08-turning-the-knobs` — one-change experiments; *higher reward ≠ better
   robot*; the curriculum-clobber gotcha.

**Part IV — Reward engineering, the throughline** *(absorbs running-and-flight,
more-gaits, reward-hacking-gallery)*
9. `09-the-running-dive` — rewarding "air time" → a reward-hacked face-plant
   (proxy gaming, shown live).
10. `10-more-gaits-and-commands` — spin, backward; the command system and
    curriculum mechanics in depth.
11. `11-the-reward-hacking-gallery` — a taxonomy of failure modes (proxy gaming,
    silent compensation, metric lying).

**Part V — Imitation: copying a reference** *(absorbs imitation-cartwheel,
imitation-backflip)*
12. `12-imitation-and-the-cartwheel` — retargeting pipeline, the `exp(-error/std)`
    reward, termination thresholds, the scorer that lied (links
    `cartwheel-journey.md`).
13. `13-the-backflip-in-three-attempts` — threshold diagnosis, and the landing
    reward **we wrote** (the real `landing_feet_upright` code walked through).

**Part VI — Building a task from nothing** *(absorbs getting-up)*
14. `14-building-get-up-from-scratch` — designing an initial-state distribution,
    the new mjlab task **we built** (real reset/reward code), the four-iteration
    reward-shaping journey.

**Part VII — Synthesis**
15. `15-reward-engineering-as-craft` — the meta-lessons; the
    [methods reference](../reports/methods-reference.md) as the consolidated
    toolkit; where to go next.

## "No insight left unturned" — three recurring devices

Woven through every chapter:
- **Real artifacts inline:** the exact training commands, the actual reward
  functions we wrote (e.g. `landing_feet_upright`, the get-up reset), the real
  plots and clips (all 56 assets get a home).
- **Insight callouts:** a recurring highlighted box surfacing the deep "why" at
  each step (the curriculum-clobber, why `exp(-error/std)` is a good reward, why
  ending the episode on success backfired).
- **"What you now understand" + forward hook** closing each chapter: a few bullets
  consolidating the new concepts and a one-line bridge to the next rung.

**Honesty preserved:** the failures stay failures (the dive, the back-landing, the
stillness hack, the scorer that lied). The curriculum's thesis — *RL is reward
engineering; the metric can lie; watch the robot* — is carried by those honest
results, so none get sanded into clean successes.

## Migration mechanics, files & assets

**This is a pure authoring/synthesis project — no Spark, no training, no GPU.**
Every clip, plot, command, and result already exists; we restructure and deepen.

**New file set** in `docs/reports/`:
- `01-the-big-picture.md` … `15-reward-engineering-as-craft.md` (15 chapters).
- `README.md` — rewritten as the curriculum table of contents (Parts I–VII).
- `index.md` (Pages landing) — rewritten to open onto the new series.

**Retired** (absorbed → removed; preserved in git history): the 10 topical
reports — `00-primer`, `01-watching-it-learn`, `02-reproducing-the-benchmark`,
`03-turning-the-knobs`, `running-and-flight`, `more-gaits`, `imitation-cartwheel`,
`imitation-backflip`, `getting-up`, `reward-hacking-gallery`.

**Kept as-is** (linked from the new series as deep-dive appendices):
- `methods-reference.md` (Part VII links it).
- `docs/cartwheel-journey.md` (ch.12 links it).
- the reproducibility code `recovery-task/`, `backflip-v3/` (ch.13–14 link the
  real source).

**Assets:** all 56 existing clips/plots get a home in the relevant chapter —
coverage tracked so none are orphaned. A few *new* static diagrams may help the
theory chapters (the train→rollout→update loop, the policy-gradient nudge) —
simple SVG/text-diagram figures authored locally, no rendering needed; flagged
when added.

**Collision-safe sequencing:** new `01/02/03…` would briefly collide with old
`01/02/03…`, so the plan writes new files under final names, removes retired ones,
rewires `index.md`/`README.md`, then runs a link-and-asset audit — the site is
never left half-broken.

## Build approach

- A shared **concept ledger** authored first: the ordered list of every term/idea
  and the chapter that introduces it — the spine's correctness check (nothing used
  before it is defined). Every chapter declares "assumes" (prior concepts) and
  "introduces" (new ones).
- Chapters drafted in dependency order, each pulling from its specific source
  reports + assets, to the depth/voice spec above.
- Execute with subagents (fresh author per chapter + a reviewer per chapter
  against the ledger), then a whole-series **continuity pass**.
- **Continuity pass:** read start to finish as a novice would — verify the ladder
  holds (no forward references), every asset is placed, all links resolve, no term
  is undefined on first use, the honest failures survived intact.

## Success criteria

1. A true beginner can start at chapter 1 knowing nothing and finish chapter 15
   understanding everything this repo did at a deep level — including the
   policy-gradient/PPO machinery and the actual reward code we wrote.
2. Every concept, method, and insight from the 10 retired reports is preserved or
   deepened — nothing lost in the migration.
3. All 56 clips/plots are placed; the real artifacts (commands, reward functions,
   new task code) appear inline where relevant.
4. The honest failures (dive, back-landing, stillness hack, scorer-that-lied)
   remain failures and carry the thesis.
5. Zero draft banners; all internal links resolve; `index.md` + syllabus point
   only at the new series; methods-reference + cartwheel-journey survive as linked
   appendices.
6. Published live, in sync with origin.

## Out of scope

- Any new training or experiments (pure synthesis).
- Altering the actual results.
- RL theory beyond what's needed to understand *this repo's* work — PPO is derived
  conceptually for a novice, not a survey of alternative algorithms.

## Open questions

None outstanding — the four design sections were approved as presented. Granularity
(15 chapters / 7 parts) and depth (full, incl. RL theory) are settled; any
chapter split/merge can be handled during planning without reopening the design.
