# G1 Deep Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the topical multi-track reports with a single zero-to-deep concept-ladder curriculum (15 chapters / 7 parts) that takes a beginner from no knowledge to a deep understanding of everything this repo did — including the RL-theory spine and the real reward code we wrote.

**Architecture:** Pure documentation synthesis — no training, no GPU, no Spark. A concept ledger is authored first as the ladder's correctness backbone (every term mapped to the chapter that introduces it). Fifteen chapters are then written in dependency order, each absorbing specific retired reports and re-homing specific existing assets, to a strict intuition-then-formalism voice. Finally the syllabus + landing page are rewired, the 10 retired reports removed, and a whole-series continuity pass verifies the ladder holds end-to-end.

**Tech Stack:** Markdown (GitHub-flavored, served by GitHub Pages with the Cayman Jekyll theme from `docs/`). Source spec: `docs/superpowers/specs/2026-06-21-g1-deep-curriculum-design.md`.

## Global Constraints

- **Pure synthesis — NO training, NO GPU, NO Spark.** Every clip, plot, command, and result already exists in the repo or the retired reports. Do not invent results, numbers, or experiments. Do not run `mjlab.scripts.*`.
- **Repo is PUBLIC** — everything committed is world-readable. No secrets, no infra details (the old CLAUDE.md was deliberately untracked for this reason; do not reintroduce host IPs/usernames/container names into published files).
- **Audience/voice:** reader with *zero* robotics/ML background. Every term defined the first time it appears. Concrete analogies lead.
- **Depth ceiling = full, including RL theory.** Math and code blocks are allowed and expected, but always via the **intuition-then-formalism sandwich**: (1) plain-language + analogy, (2) the formula/code introduced gently and immediately re-explained in words with every symbol named, (3) a worked example from *this repo*. Never a wall a novice hits.
- **Concept-ladder rule (the make-or-break property):** no chapter may use a term/concept before the chapter that introduces it (per the concept ledger). Every chapter respects its declared `assumes` and `introduces`.
- **Three recurring devices in every chapter:** (a) real artifacts inline (exact commands, the real reward code, the real plots/clips); (b) **Insight callouts** — a recurring highlighted box surfacing the deep "why"; (c) a **"What you now understand"** recap + one-line forward hook at the end.
- **Honesty:** the failures stay failures (the air-time dive, the back-landing backflip, the get-up stillness hack, the cartwheel scorer-that-lied). They carry the thesis; do not sand them into clean successes.
- **Asset reuse:** embed existing assets from `docs/reports/assets/` via the same `<video controls autoplay loop muted playsinline ...>` pattern the current reports use for MP4s, and `![alt](assets/...)` for PNGs. All 56 assets must be placed (Task 20 audits this).
- **Video embed pattern (copy exactly, swap filenames):**
  ```html
  <video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/POSTER.png">
    <source src="assets/CLIP.mp4" type="video/mp4">
    Your browser doesn't support embedded video — <a href="assets/CLIP.mp4">download the clip</a> instead.
  </video>
  ```
- **Relative links:** chapters link siblings by bare filename (`12-imitation-and-the-cartwheel.md`); link the methods reference as `methods-reference.md`; link the cartwheel journey as `../cartwheel-journey.md`; link repo code as `../../recovery-task/` and `../../backflip-v3/`.
- **Staging discipline:** stage only the files each task creates/modifies, by explicit path. Never `git add -A`. The untracked `.superpowers/` scratch and the untracked local `CLAUDE.md` (gitignored) must never be staged.
- **Branch:** `g1-deep-curriculum` (already created; spec already committed there).
- **Commit footer** (every commit):
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01D6dhn7JiNfx8tpFbitRmgN
  ```
- **Verification, not pytest:** these are docs. Each task's "test" is: required sections present, concept-ledger compliance (no undefined-on-first-use, no forward references), assigned assets embedded and paths exist, links resolve, recap+hook present, honest results intact.

---

## File Structure

**Build artifact (scratch, not published):**
- `.superpowers/sdd/concept-ledger.md` — the ordered concept→chapter map + the asset→chapter map. Authored in Task 1; read by every chapter author.

**Created (`docs/reports/`):** the 15 chapters —
`01-the-big-picture.md`, `02-the-vocabulary.md`, `03-trial-and-error.md`,
`04-ppo-for-novices.md`, `05-reading-the-training.md`, `06-watching-it-walk.md`,
`07-proving-its-real.md`, `08-turning-the-knobs.md`, `09-the-running-dive.md`,
`10-more-gaits-and-commands.md`, `11-the-reward-hacking-gallery.md`,
`12-imitation-and-the-cartwheel.md`, `13-the-backflip-in-three-attempts.md`,
`14-building-get-up-from-scratch.md`, `15-reward-engineering-as-craft.md`.

**Modified:** `docs/reports/README.md` (→ curriculum TOC), `docs/index.md` (→ landing for the new series).

**Removed (Task 19):** `docs/reports/00-primer.md`, `01-watching-it-learn.md`, `02-reproducing-the-benchmark.md`, `03-turning-the-knobs.md`, `running-and-flight.md`, `more-gaits.md`, `imitation-cartwheel.md`, `imitation-backflip.md`, `getting-up.md`, `reward-hacking-gallery.md`.

**Kept (appendices, linked):** `docs/reports/methods-reference.md`, `docs/cartwheel-journey.md`, `recovery-task/`, `backflip-v3/`.

**Authoring cycle for every chapter task (the "steps"):**
1. Read the concept-ledger entry (assumes/introduces + assigned assets) and the source report(s) this chapter absorbs.
2. Draft the chapter to the voice/depth spec + the three devices, embedding the assigned assets.
3. Verify: ledger compliance (no undefined term, no forward reference), assets embedded with existing paths, links resolve, recap+hook present, honest results intact, no draft banner.
4. Commit (only the new file).

---

### Task 1: Concept ledger + asset map

**Files:**
- Create: `.superpowers/sdd/concept-ledger.md`

**Produces:** the ordered concept→chapter map and the asset→chapter map that every chapter task consumes.

- [ ] **Step 1: Inventory the assets.** List `docs/reports/assets/` and bucket all 56 files by the work they depict (walking/progression, running s1_*, gaits gait_*, reward-hack s4_*, cartwheel_*, backflip s2_*, get-up s3_*). Record each file → its target chapter. Note `control_scalars.csv` is source data (link it from ch07; not a media embed); the other 55 are `.mp4`/`.png` media to embed.
- [ ] **Step 2: Author the concept ledger** — an ordered table: every term/concept the curriculum teaches, in first-appearance order, with the chapter that `introduces` it. Cover at minimum: simulator, parallel envs, observation, action, policy, reward, episode, environment, rollout, return, trial-and-error, policy gradient, gradient/optimization step, advantage, value function, PPO, clipping, on-policy, reward curve, episode-length metric, convergence, velocity/command tracking, reward term, reward weight, emergent gait, curriculum, curriculum-clobber, termination, reward hacking/proxy gaming, silent compensation, metric-lying, motion imitation, reference motion, retargeting, tracking reward `exp(-error/std)`, termination threshold, initial-state distribution, reward shaping iteration, gated reward.
- [ ] **Step 3: Map concepts → the 7-part chapter structure** (per the spec's chapter map), assigning each concept's `introduces` chapter and listing each chapter's `assumes` (all concepts introduced in earlier chapters it relies on).
- [ ] **Step 4: Verify** every concept has exactly one `introduces` chapter; every chapter's `assumes` references only earlier chapters; every one of the 56 assets is assigned to exactly one chapter. (This file is scratch — do NOT commit it; it lives under `.superpowers/`.)

---

### Task 2: Chapter 01 — The big picture

**Files:**
- Create: `docs/reports/01-the-big-picture.md`
**Absorbs:** `00-primer.md` (the framing half). **Assets:** none required (conceptual); may reference a still if the ledger assigns one.
**Introduces:** robot-that-learns-vs-programmed, simulator, parallel environments. **Assumes:** nothing.

- [ ] **Step 1: Read** `00-primer.md` and the spec's Part I description.
- [ ] **Step 2: Write** the chapter: what this project is (a humanoid that *learns* to move by practicing in simulation, not hand-programmed joints); what a physics simulator is and why we use one; why thousands of robots train in parallel (more experience per unit time). Plain language, analogies, zero assumed terms. End with the "What you now understand" recap + a forward hook to the vocabulary chapter.
- [ ] **Step 3: Verify** no undefined jargon; recap+hook present; no draft banner; reads as chapter 1 of a series (welcoming, zero assumptions).
- [ ] **Step 4: Commit** (`git add docs/reports/01-the-big-picture.md`; message `Curriculum ch01: the big picture`).

---

### Task 3: Chapter 02 — The vocabulary

**Files:**
- Create: `docs/reports/02-the-vocabulary.md`
**Absorbs:** `00-primer.md` (the vocabulary half). **Assets:** per ledger (likely a labelled still if available).
**Introduces:** observation, action, policy, reward, episode, environment, rollout, return. **Assumes:** ch01 concepts.

- [ ] **Step 1: Read** `00-primer.md` and ch01.
- [ ] **Step 2: Write** each core term defined once, concretely, in dependency order: environment → observation → action → policy (the brain mapping observations→actions) → reward (the score signal) → episode → rollout → return (sum of rewards). Use the G1 walking setting as the running concrete example. Recap + hook to Part II ("now: how does the policy actually improve?").
- [ ] **Step 3: Verify** every listed term defined exactly once on first use; uses only ch01 concepts before defining its own; recap+hook present.
- [ ] **Step 4: Commit** (message `Curriculum ch02: the vocabulary`).

---

### Task 4: Chapter 03 — Trial and error made precise (policy gradient)

**Files:**
- Create: `docs/reports/03-trial-and-error.md`
**Assets:** may add one new text/SVG diagram of the "nudge toward what paid off" idea (author locally; flag it). **Introduces:** trial-and-error learning, the policy-gradient idea, a gradient/optimization step. **Assumes:** ch01–02.

- [ ] **Step 1: Read** ch02 (for the policy/reward/return definitions it builds on).
- [ ] **Step 2: Write** using the intuition-then-formalism sandwich: (1) analogy — adjusting a recipe after each taste; the policy is a set of dials, and we nudge the dials toward settings that earned more return; (2) the precise idea — increase the probability of actions that led to higher-than-expected return, decrease it for lower; name "policy gradient" and explain it in words (no heavy derivation — the *idea*, gently); (3) worked example — the walking policy's return going up over training. Optionally a simple diagram. Recap + hook to PPO.
- [ ] **Step 3: Verify** the sandwich is intact (intuition before any formal statement; every symbol/term re-explained in words); a novice could follow it; recap+hook present; any new diagram file embedded with a valid path.
- [ ] **Step 4: Commit** (message `Curriculum ch03: trial and error (policy gradient)`).

---

### Task 5: Chapter 04 — PPO for novices

**Files:**
- Create: `docs/reports/04-ppo-for-novices.md`
**Assets:** optional new diagram of the train→rollout→update loop. **Introduces:** on-policy learning, value function, advantage, PPO objective, clipping/trust-region stability. **Assumes:** ch01–03.

- [ ] **Step 1: Read** ch03 (policy gradient) — PPO is the stable, practical version of that idea.
- [ ] **Step 2: Write** with the sandwich, in this order: (1) the problem with naive policy gradient — too-big updates destabilize learning (analogy: over-correcting the steering); (2) advantage — "was this action better than expected?" needs a baseline → the value function (a learned guess of expected return); (3) PPO's fix — only trust an update so far; the clipping idea that caps how much the policy changes per step, keeping learning stable; state the clipped-objective idea in words (and a compact formula, immediately re-explained, every symbol named); (4) "on-policy" — why PPO learns from its own fresh rollouts. Worked example: point at the real walking reward curve's smooth climb as PPO working. Note clearly that PPO is machinery this repo *uses* (via mjlab/rsl_rl), not something we authored. Recap + hook to "reading the training."
- [ ] **Step 3: Verify** the hardest chapter still obeys the sandwich (no formula without plain-language lead-in and symbol-by-symbol re-explanation); advantage/value/clipping each introduced before use; the "we use, didn't write PPO" caveat present; recap+hook present.
- [ ] **Step 4: Commit** (message `Curriculum ch04: PPO for novices`).

---

### Task 6: Chapter 05 — Reading the training

**Files:**
- Create: `docs/reports/05-reading-the-training.md`
**Absorbs:** the "how to read a reward curve" content from `01-watching-it-learn.md`. **Assets:** the baseline reward + episode-length plots (`learn_baseline_reward.png`, `learn_baseline_episode_length.png` per ledger). **Introduces:** reward curve, mean-reward, episode-length metric, convergence/plateau, the "metric ≠ behavior" caution (seed). **Assumes:** ch01–04.

- [ ] **Step 1: Read** `01-watching-it-learn.md` (curve-reading section) + ch04.
- [ ] **Step 2: Write** how to read training output: the reward curve (S-shape, climb, plateau), episode length (and why it tracks reward early), what "convergence" looks like, and the first seed of the thesis — a number going up doesn't prove the *behavior* is good (foreshadow reward hacking). Embed the baseline plots. Recap + hook to Part III ("let's watch a real one learn to walk").
- [ ] **Step 3: Verify** plots embedded with valid paths; only ch01–04 concepts used before introducing its own; the "metric ≠ behavior" seed present; recap+hook present.
- [ ] **Step 4: Commit** (message `Curriculum ch05: reading the training`).

---

### Task 7: Chapter 06 — Watching it walk

**Files:**
- Create: `docs/reports/06-watching-it-walk.md`
**Absorbs:** `01-watching-it-learn.md`. **Assets:** the progression clips + stills (`chase.mp4`, `side.mp4`, `grid.mp4`, `still_early.png`, `still_mid.png`, `still_final.png` per ledger). **Introduces:** velocity/command tracking task, reward term, reward weight, emergent gait. **Assumes:** ch01–05.

- [ ] **Step 1: Read** `01-watching-it-learn.md` + the progression assets.
- [ ] **Step 2: Write** the flailing→walking story frame by frame (embed the progression clips + the three stills labelled by iteration); introduce the velocity-tracking task (command "go this fast this way") and that the gait *emerges* from reward terms rather than being choreographed. Recap + hook to reproducing it.
- [ ] **Step 3: Verify** progression clips + stills embedded; emergent-gait idea clear; uses only ch01–05 concepts before its own; recap+hook present.
- [ ] **Step 4: Commit** (message `Curriculum ch06: watching it walk`).

---

### Task 8: Chapter 07 — Proving it's real

**Files:**
- Create: `docs/reports/07-proving-its-real.md`
**Absorbs:** `02-reproducing-the-benchmark.md`. **Assets:** `repro_Train_mean_reward.png`, `repro_Train_mean_episode_length.png`, and the per-term plots (`control_term_*.png`) per ledger. **Introduces:** reproducibility/seed, the reward-term breakdown (track_lin_vel, upright, air_time, action_rate, etc.), termination. **Assumes:** ch01–06.

- [ ] **Step 1: Read** `02-reproducing-the-benchmark.md` + assets.
- [ ] **Step 2: Write** the reproduce-from-scratch story (same curve = real, not luck; the exact train command), then unpack what each reward term actually rewards (embed the per-term plots) and what terminations do. Recap + hook to "turning the knobs."
- [ ] **Step 3: Verify** the real train command shown; reward terms explained + plots embedded; reproducibility argument intact; recap+hook present.
- [ ] **Step 4: Commit** (message `Curriculum ch07: proving it's real`).

---

### Task 9: Chapter 08 — Turning the knobs

**Files:**
- Create: `docs/reports/08-turning-the-knobs.md`
**Absorbs:** `03-turning-the-knobs.md`. **Assets:** `ab_control.mp4`, `ab_tweak.mp4`, `ab_control_still.png`, `ab_tweak_still.png`, `ab_Train_mean_reward.png`, `ab_speed_tracking_error.png` per ledger. **Introduces:** one-change experiment discipline, *higher reward ≠ better robot*, the command-velocity range + the curriculum-clobber. **Assumes:** ch01–07.

- [ ] **Step 1: Read** `03-turning-the-knobs.md` + assets.
- [ ] **Step 2: Write** the slow-vs-full-range A/B (embed both clips + stills + plots); the change-exactly-one-thing discipline; the first full statement of *higher reward ≠ better robot*; and the curriculum-clobber gotcha (must override the command AND all three curriculum stages, `=`-syntax). **Insight callout** on the clobber. Recap + hook to Part IV (reward engineering).
- [ ] **Step 3: Verify** A/B assets embedded; the curriculum-clobber explained with the real override pattern; the higher-reward≠better-robot thesis stated; recap+hook present.
- [ ] **Step 4: Commit** (message `Curriculum ch08: turning the knobs`).

---

### Task 10: Chapter 09 — The running dive (proxy gaming, live)

**Files:**
- Create: `docs/reports/09-the-running-dive.md`
**Absorbs:** `running-and-flight.md`. **Assets:** `s1_walker_side/chase.mp4` + still, `s1_dive_side/chase.mp4` + still, `s1_shuffle_side.mp4` + still, `s1_reward.png`, `s1_air_time.png` per ledger. **Introduces:** reward hacking, proxy gaming, flight phase. **Assumes:** ch01–08.

- [ ] **Step 1: Read** `running-and-flight.md` + assets.
- [ ] **Step 2: Write** the three-policy story (walker/dive/shuffle): rewarding `air_time` to force a flight phase reward-hacks into a face-down dive that scores *highest* (~60 vs walker ~51) while looking worst; embed the clips + the reward + air-time plots (note the air-time spike at the moment the hack is discovered). Define reward hacking / proxy gaming here as the first named instance. **Insight callout** on proxy-vs-outcome. Recap + hook to gaits/commands.
- [ ] **Step 3: Verify** all three policies' clips + plots embedded; the dive stays an honest failure; reward-hacking/proxy-gaming defined; recap+hook present.
- [ ] **Step 4: Commit** (message `Curriculum ch09: the running dive`).

---

### Task 11: Chapter 10 — More gaits and the command system

**Files:**
- Create: `docs/reports/10-more-gaits-and-commands.md`
**Absorbs:** `more-gaits.md`. **Assets:** `gait_spin_side/chase.mp4` + still, `gait_backward_side.mp4` + still per ledger. **Introduces:** the command/twist system in depth, command-vs-curriculum interaction, "same recipe, unequal results." **Assumes:** ch01–09.

- [ ] **Step 1: Read** `more-gaits.md` + assets.
- [ ] **Step 2: Write** spin-in-place (clean first-try, reward ~76) and backward-walk (honest partial, reward ~39 — tracks the −1 m/s command less cleanly); the command system (lin_vel_x/y, ang_vel_z) and how the curriculum re-randomizes it (callback to the clobber); the lesson that "just a command change" doesn't guarantee equal quality. Embed clips. Recap + hook to the gallery.
- [ ] **Step 3: Verify** both gait clips embedded; backward stays an honest partial; command system explained; recap+hook present.
- [ ] **Step 4: Commit** (message `Curriculum ch10: more gaits and the command system`).

---

### Task 12: Chapter 11 — The reward-hacking gallery

**Files:**
- Create: `docs/reports/11-the-reward-hacking-gallery.md`
**Absorbs:** `reward-hacking-gallery.md`. **Assets:** reuse `s1_dive_side.mp4` (proxy gaming), `s4_noupright_side.mp4` + still (silent compensation); the cartwheel-scorer exhibit is narrative (its clip lands in ch12). **Introduces:** the taxonomy — proxy gaming (recap), silent compensation, metric lying. **Assumes:** ch01–10.

- [ ] **Step 1: Read** `reward-hacking-gallery.md` + assets.
- [ ] **Step 2: Write** the three-failure-mode taxonomy: (1) proxy gaming (the dive, recap from ch09); (2) silent compensation (remove the `upright` reward → robot doesn't collapse, the fall-termination quietly holds posture, it just under-moves; embed `s4_noupright`); (3) metric lying (the cartwheel roll-angle scorer fooled by crash-rolls — narrative, forward-link the full story to ch12). This is the throughline chapter: name the pattern that recurs across the whole project. **Insight callout** distinguishing the three. Recap + hook to Part V.
- [ ] **Step 3: Verify** all three modes covered with their evidence; the no-upright clip embedded; metric-lying forward-links to ch12 (no broken claim); recap+hook present.
- [ ] **Step 4: Commit** (message `Curriculum ch11: the reward-hacking gallery`).

---

### Task 13: Chapter 12 — Imitation and the cartwheel

**Files:**
- Create: `docs/reports/12-imitation-and-the-cartwheel.md`
**Absorbs:** `imitation-cartwheel.md` (links `../cartwheel-journey.md`). **Assets:** `cartwheel_side.mp4`, `cartwheel_chase.mp4`, `cartwheel_still.png` per ledger. **Introduces:** motion imitation, reference motion, retargeting pipeline, the tracking reward `exp(-error/std)`, termination threshold, the scorer-that-lied (payoff of ch11's metric-lying). **Assumes:** ch01–11.

- [ ] **Step 1: Read** `imitation-cartwheel.md`, `../cartwheel-journey.md` + assets.
- [ ] **Step 2: Write** the paradigm shift — instead of shaping a reward from scratch, give the robot a *reference motion* to match each step; the retargeting pipeline (`.pkl`→CSV→`.npz`); the tracking reward — introduce `exp(-error/std)` via the sandwich (analogy: full marks for a perfect match, smoothly less as you drift; the formula, every symbol named; why it's a good shape); termination thresholds (0.25→0.5 m for aerial motion). Then the cartwheel success (embed clips) and the iterB scorer-lied story (the payoff of ch11). Link the deep log `../cartwheel-journey.md`. **Insight callout** on why `exp(-error/std)` is a good reward. Recap + hook to the backflip.
- [ ] **Step 3: Verify** the `exp(-error/std)` sandwich intact; cartwheel clips embedded; retargeting pipeline shown; scorer-lied story resolved here; journey link resolves; recap+hook present.
- [ ] **Step 4: Commit** (message `Curriculum ch12: imitation and the cartwheel`).

---

### Task 14: Chapter 13 — The backflip in three attempts

**Files:**
- Create: `docs/reports/13-the-backflip-in-three-attempts.md`
**Absorbs:** `imitation-backflip.md` (links `../../backflip-v3/`). **Assets:** `s2_v1_grounded_side.mp4`, `s2_v2_attempt_side/chase.mp4`, `s2_v2_still.png`, `s2_anchor_err.png`, `s2_ee_term.png`, `s2_v3_landed_side/chase.mp4`, `s2_v3_landed_still.png` per ledger. **Introduces:** gated reward, the landing reward we wrote, threshold-diagnosis-by-metric. **Assumes:** ch01–12.

- [ ] **Step 1: Read** `imitation-backflip.md`, `backflip-v3/README.md`, `backflip-v3/landing_feet_upright.py` + assets.
- [ ] **Step 2: Write** the three-attempt arc: v1 tight thresholds → never leaves the ground; v2 loosened → airborne but lands on its back; v3 + a **landing reward we wrote** → lands on its feet. Walk through the real `landing_feet_upright` code (the gating to the last 40% of the motion — introduce "gated reward"), why it's cartwheel-safe (default weight 0), and the metric tells (landing reward climbing + anchor error dropping 0.63→0.43). Embed v1/v2/v3 clips + the metric plots. Link `../../backflip-v3/`. Keep the honest nuance (recovering crouch, not a crisp stick). **Insight callout** on gated rewards. Recap + hook to building a task from scratch.
- [ ] **Step 3: Verify** the real landing-reward code walked through; all three attempts' clips + the metric plots embedded; gated-reward introduced; honest-nuance intact; code link resolves; recap+hook present.
- [ ] **Step 4: Commit** (message `Curriculum ch13: the backflip in three attempts`).

---

### Task 15: Chapter 14 — Building get-up from scratch

**Files:**
- Create: `docs/reports/14-building-get-up-from-scratch.md`
**Absorbs:** `getting-up.md` (links `../../recovery-task/`). **Assets:** `s3_getup_side/chase.mp4`, `s3_getup_still.png`, `s3_crouch_side.mp4`, `s3_reward.png`, `s3_standup.png` per ledger. **Introduces:** initial-state distribution, building a new task (no reference, no command), the four-iteration reward-shaping journey, success-termination pitfall. **Assumes:** ch01–13.

- [ ] **Step 1: Read** `getting-up.md`, `recovery-task/README.md`, `recovery-task/mdp/events.py` + `rewards.py` + `recovery-task/config/g1/env_cfgs.py` + assets.
- [ ] **Step 2: Write** the hardest paradigm — no reference, no command, behavior is 100% the reward you write. Cover: designing the initial-state distribution (the four genuinely-fallen reset poses; the zero-action probe that proved it non-trivial); the new mjlab task we built (the real reset/reward code, lightly walked through); the four-attempt shaping journey (stillness hack → success-termination trap → stable-crouch local optimum → full stand) — each a distinct, real reward failure fixed. Embed the get-up clips (v4 full stand + v3 crouch contrast) + the reward/height plots. Link `../../recovery-task/`. **Insight callout** on "in a from-scratch task, the initial-state distribution and success definition are as load-bearing as the reward." Recap + hook to synthesis.
- [ ] **Step 3: Verify** all four attempts documented as distinct failures→fix; real task code referenced; clips + plots embedded; code link resolves; recap+hook present.
- [ ] **Step 4: Commit** (message `Curriculum ch14: building get-up from scratch`).

---

### Task 16: Chapter 15 — Reward engineering as the craft (synthesis)

**Files:**
- Create: `docs/reports/15-reward-engineering-as-craft.md`
**Assets:** none required (synthesis). **Introduces:** the consolidated thesis; pointers onward. **Assumes:** ch01–14.

- [ ] **Step 1: Read** all prior chapters' recaps + `methods-reference.md`.
- [ ] **Step 2: Write** the capstone: pull the throughline together — RL is reward engineering; the robot does exactly what you reward; the metric can lie; the only real verdict is watching the robot. Summarize the arc (emergent gaits → reward hacks → imitation → from-scratch tasks) and what each taught. Point to `methods-reference.md` as the consolidated toolkit and the repo code (`../../recovery-task/`, `../../backflip-v3/`) for the curious. End with "where to go next" (the deferred Tier specs). Final "What you now understand — everything." No forward hook (it's the end); instead a closing.
- [ ] **Step 3: Verify** the thesis is fully articulated; methods-reference + code links resolve; no new undefined concepts; reads as a satisfying capstone.
- [ ] **Step 4: Commit** (message `Curriculum ch15: reward engineering as the craft`).

---

### Task 17: Rewrite the syllabus README

**Files:**
- Modify: `docs/reports/README.md`

- [ ] **Step 1: Read** the current `README.md` + all 15 new chapter titles.
- [ ] **Step 2: Rewrite** it as the curriculum table of contents: a short intro (zero-background, read in order), then the 7 Parts each listing its chapters with a one-line hook, then an "Appendices" section linking `methods-reference.md` and `../cartwheel-journey.md`. Remove the old paradigm-track tables. Preserve the "Media / how produced" note (updated: clips/plots embedded throughout; source on the Spark).
- [ ] **Step 3: Verify** all 15 chapter links resolve to created files; the 7-part structure matches the spec; appendix links resolve; no link to a to-be-removed report.
- [ ] **Step 4: Commit** (message `Curriculum: rewrite README as the new TOC`).

---

### Task 18: Rewrite the Pages landing page

**Files:**
- Modify: `docs/index.md`

- [ ] **Step 1: Read** the current `docs/index.md`.
- [ ] **Step 2: Rewrite** it to open onto the new series: the same inviting zero-background hook, then "Start at [Chapter 1](reports/01-the-big-picture.md) and read in order," then a brief Parts overview linking the part entry points, and the appendix links. Keep the Chaotic Curiosity / mjlab / repo footer.
- [ ] **Step 3: Verify** links point only at new chapters + appendices (no link to a retired report); the "read in order" framing is clear; footer intact.
- [ ] **Step 4: Commit** (message `Curriculum: rewrite Pages landing for the new series`).

---

### Task 19: Retire the 10 topical reports + fix kept-file references

**Files:**
- Remove: `docs/reports/00-primer.md`, `01-watching-it-learn.md`, `02-reproducing-the-benchmark.md`, `03-turning-the-knobs.md`, `running-and-flight.md`, `more-gaits.md`, `imitation-cartwheel.md`, `imitation-backflip.md`, `getting-up.md`, `reward-hacking-gallery.md`
- Modify (only if they link a removed report): `docs/reports/methods-reference.md`

- [ ] **Step 1: Grep** the kept files for links to any of the 10 retired reports: `git grep -nE "00-primer|01-watching-it-learn|02-reproducing|03-turning|running-and-flight|more-gaits|imitation-cartwheel|imitation-backflip|getting-up|reward-hacking-gallery" -- 'docs/' ':!docs/superpowers'`. (The new chapters and README should already point at new files; this catches `methods-reference.md` or stragglers.)
- [ ] **Step 2: Repoint** any such links to the corresponding new chapter (e.g. a `03-turning-the-knobs.md` link → `08-turning-the-knobs.md`); leave `docs/superpowers/` historical docs untouched (they record the prior project faithfully).
- [ ] **Step 3: Remove** the 10 retired report files with `git rm`.
- [ ] **Step 4: Verify** no kept/published file (outside `docs/superpowers/`) links to a removed file; the 10 files are gone; the 15 chapters + README + methods-reference + index remain.
- [ ] **Step 5: Commit** (message `Curriculum: retire the 10 topical reports (absorbed into the new series)`).

---

### Task 20: Whole-series continuity pass + asset audit

**Files:**
- Modify: any chapter needing a continuity fix (by exact path).

- [ ] **Step 1: Concept-ladder check.** Read all 15 chapters in order against `.superpowers/sdd/concept-ledger.md`: confirm no term is used before its `introduces` chapter (no forward references), and every chapter introduces what the ledger assigns it. Note any violation.
- [ ] **Step 2: Asset audit.** Of the 56 files in `docs/reports/assets/`, the 55 media files (`.mp4`/`.png`) must each be embedded in exactly one chapter (grep each filename across `docs/reports/*.md`); the one data file `control_scalars.csv` is the source data behind ch07's per-term plots — it is *linked* (or simply noted as the underlying data) from ch07, not embedded as media, and is not an orphan. List any genuinely orphaned media file; place orphans in the right chapter.
- [ ] **Step 3: Link + jargon audit.** Confirm every intra-repo relative link in the 15 chapters + README + index resolves to an existing file; confirm no draft banners remain; spot-check that no term appears undefined on first use.
- [ ] **Step 4: Fix** any issue found inline (smallest change; respect the ledger).
- [ ] **Step 5: Commit** (message `Curriculum: whole-series continuity pass (ladder, assets, links)`).

---

## After this plan

Publishing (merge `g1-deep-curriculum` → `main` and push) is handled by the **finishing-a-development-branch** skill after the continuity pass passes — not a plan task. That delivers success criterion 6 (published live, in sync with origin).

---

## Self-Review

**1. Spec coverage** — every spec element maps to a task: concept ledger + asset map → Task 1; the 15 chapters (Parts I–VII) → Tasks 2–16; the depth/voice sandwich + three devices → embedded in every chapter task's Step 2 + Global Constraints; supersede mechanics (rewrite README/index, retire 10, keep appendices) → Tasks 17–19; the continuity pass + 56-asset coverage → Task 20; publish → finishing-a-development-branch (noted). The "absorbs" mapping covers all 10 retired reports (00→ch1–2, 01→ch5–6, 02→ch7, 03→ch8, running→ch9, more-gaits→ch10, gallery→ch11, cartwheel→ch12, backflip→ch13, getup→ch14).

**2. Placeholder scan** — no TODO/TBD/"fill in later" in any task's instructions. Chapter tasks intentionally don't pre-write the prose (the prose IS the deliverable), but each gives the author the exact source files, assigned assets, assumes/introduces concepts, content beats, and required insight callouts — enough to write without re-deriving. The concept-ledger and asset-map specifics are produced in Task 1 (the one dependency every chapter consumes), not left vague.

**3. Type/name consistency** — chapter filenames are identical between the File Structure block, each task header, the README/index link tasks, and the retire-list (new `01-the-big-picture`…`15-reward-engineering-as-craft`; retired `00-primer`…`reward-hacking-gallery`). The `assumes`/`introduces` chains are monotonic (each chapter assumes only lower-numbered chapters). Asset filenames referenced match the existing `docs/reports/assets/` inventory used by the current reports.
