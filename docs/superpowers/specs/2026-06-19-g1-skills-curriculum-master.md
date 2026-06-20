# Design: G1 Skills Curriculum — a multi-paradigm learning program

**Date:** 2026-06-19
**Branch:** `g1-skills-curriculum`
**Status:** Approved (verbal), pending implementation plan
**Supersedes/extends:** `2026-06-18-g1-walking-learning-arc-design.md` (the walking
arc becomes the Foundations + Locomotion track of this larger program)

## Goal

Extend the existing zero-background learning series (`docs/reports/00`–`03`) into a
**multi-paradigm curriculum** that onboards a newcomer to *everything* this repo can
do: locomotion shaping, motion imitation, from-scratch task rewards, and reward
engineering. Same reader as the walking series — **no robotics or ML background** —
same voice: plain language, concrete analogies over notation, every term defined on
first use, each report standalone and re-runnable.

The reader should be able to enter the syllabus, pick any track, and come away able to
explain that paradigm in their own words: what the robot is rewarded for, what
behavior *emerges*, and what they could change to explore further.

This is the **"spec all, train a spine"** path (chosen over a full train-everything
campaign or a corpus-only effort): write specs for *all* candidate tasks, but actually
train only a curated **spine** of four representative tasks now — one per paradigm —
as the onboarding backbone. The rest remain ready-to-run specs we promote to full runs
later. This maximizes understanding-per-GPU-hour at bounded reboot risk (the same
philosophy that drove the walking arc's "deep-on-one-task" choice), while still
covering every paradigm and honoring "however long it takes" as an ongoing campaign
rather than a single sprint.

## Audience & voice

Identical bar to the walking series. No undefined jargon; the first time a term appears
(policy, reward, episode, PPO, environment, termination, motion tracking, reward
hacking) it is explained in plain language. Concrete analogies. Each report standalone
and re-runnable by the reader from this checkout + the Spark.

## Program shape — two layers

1. **Master syllabus spec** — *this document*. Defines the tier taxonomy, the spine,
   the corpus/Track-A layout, the methods reference, ops/sequencing, and program-level
   success criteria. The index of record.
2. **Per-task specs** — one per candidate idea, in the proven walking-arc template,
   grouped by tier (below). Two depths:
   - **Full specs** for the four spine tasks (the trained tier leads): Goal →
     Audience → staged-arc table with costs → concrete runs with exact commands → the
     experiment → artifacts → ops/safety → success criteria → open questions. (Tier 4
     has no trained lead, so it gets only a compact spec — see below.)
   - **Compact-but-complete specs** for the deferred long tail: goal, env + reward
     terms to touch, new-code-needed flag, the train/record commands, cost estimate,
     and success criteria — enough to promote to a full run later without re-designing.
     Bundled one file per tier (a section per task) to keep the spec directory
     manageable.

### Tier taxonomy (the full candidate set — "spec all")

| Tier | Family | Per-task cost | New code | Spine pick |
|---|---|---|---|---|
| **1** | Gait tweaks on the existing velocity env — running⭐, crouched "Groucho", tiptoe, energy-efficiency, spin-in-place, prescribed-gait (hop/march), backward/sideways | ~1 h GPU each | none → light (a contact-schedule reward term) | **Running / flight phase** |
| **2** | Acrobatics on the existing tracking env — backflip⭐, spinkick/martial-arts, jump, dance | retarget + ~hours each | none (retargeter + pipeline exist) | **Backflip** |
| **3** | From-scratch task rewards — get-up⭐, push-recovery, single-leg balance | new task + reward manager + ~hours | **yes** | **Get-up / fall recovery** |
| **4** | Object / whole-body — reach-to-target, kick-a-ball, carry | MJCF scene edits + new rewards + ~hours | **yes, most** | *(spec-only — deferred)* |
| **×** | Reward-hacking gallery (cross-cutting) | cheap, fast | light | **Reward-hacking gallery** |

⭐ = trained spine. Tier 4 is **spec-only** in this program (highest research risk).

## The four spine campaigns

Each is one trained backbone task → one report. Items depending on mjlab internals not
visible from this checkout are marked **⚠ verify on Spark** and collected in "Open
questions" below.

### S1 · Running / flight phase — *Locomotion track* (cheapest)

- **Env:** `Mjlab-Velocity-Flat-Unitree-G1` (existing). **New code:** none expected.
- **Method:** raise commanded `lin_vel_x` to a high range **and override all three
  `command_vel` curriculum stages** (the curriculum-clobber gotcha — stage-0 fires at
  every reset and resets the range; tuples use `=` syntax). Bump the `feet_air_time`
  reward weight to permit a true flight phase.
- **Experiment:** existing `model_2050.pt` walker (control, already on disk) **vs** a
  fresh fast runner (B). ~1–1.5 h GPU for the fresh run.
- **Report shows:** emergence of an aerial phase (both feet off ground); a
  cadence-vs-commanded-speed plot; side/chase clips.
- **Success:** a *visible* flight phase and measurably different cadence/step-length,
  attributable to the velocity-range + air-time change.
- **⚠ verify:** exact `feet_air_time` term name/path; the three curriculum-stage
  override keys; whether high speed needs episode-length or terrain adjustment.

### S2 · Backflip — *Imitation track* (longest run)

- **Env:** `Mjlab-Tracking-Flat-Unitree-G1` (existing). **New code:** none — the
  `smpl_backflip_to_g1.py` retargeter and the tracking pipeline already exist.
- **Method:** retarget a **single, feasible** backflip reference → `motion.npz`; train
  with `--env.commands.motion.motion-file <npz>`. Apply every cartwheel lesson: one
  feasible reference (no accidental double-flip), **0.5 m anchor/ee thresholds**, and
  **render at the same threshold or with terminations off**.
- **Experiment:** likely iterative references (iterA/B/C, as the cartwheel needed).
  Training is multi-hour (cartwheel ran ~11 h — the spine's heaviest run).
- **Report shows:** multi-camera clips at matched thresholds; the score-spoofing caveat
  front and centre (`score_cartwheel.py` is roll-angle-only; visual review required).
- **Success:** a *frame-by-frame-confirmed* completed flip (takeoff → full rotation →
  landing), not a scorer number.
- **⚠ verify:** the backflip reference's feasibility/duration; thresholds that let a
  full flip complete.

### S3 · Get-up / fall recovery — *From-scratch-tasks track* (most new code)

- **Env:** **NEW** — e.g. `Mjlab-Recovery-Flat-Unitree-G1`, cloned from the flat env.
  **New code: yes — the real engineering of the program.**
- **Method:**
  - **Initial state:** spawn the robot in randomized *fallen* poses (prone / supine /
    crumpled) instead of standing.
  - **Reward:** base height rising toward standing + torso uprightness + (once up)
    stillness/stability; penalize wild joint velocity/torque. **Terminate on
    success-hold or timeout — *not* on falling** (falling is the start state, not
    failure).
  - **Command:** none (remove the twist entirely).
- **Cost:** a code-writing + container-test sub-step, then ~hours training with
  iterative reward shaping (the task most prone to reward-hacking → natural cross-link
  to S4).
- **Report shows:** the flail→stand emergence; an honest log of the reward-shaping
  iterations, including hacks it found and how we closed them.
- **Success:** from a randomized fallen pose, reliably reaches and *holds* a stable
  stand; the strategy is visible; the shaping journey is documented.
- **⚠ verify:** how mjlab defines task/reward managers and initial-state randomization
  (read an existing task config in-container); whether any upstream recovery task
  exists to fork; cleanest way to spawn fallen poses.

### S4 · Reward-hacking gallery — *Reward-engineering track* (cheapest, cross-cutting)

- **Env:** reuse existing velocity + tracking envs. **New code:** small naive reward
  terms or CLI weight overrides.
- **Method:** a set of deliberately-naive rewards, each a short, killable run that
  documents the cheat:
  - reward raw forward base velocity (no uprightness) → diving/lunging faceplant
  - reward base height only → jump-and-collapse
  - reward "distance from start" → getting launched / exploiting contacts
  - the **existing** roll-angle scorer fooled by crash-rolls (specimen already in hand)
- **Cost:** cheap — several short runs, killed once the hack is visibly reproduced.
- **Report shows:** per specimen — the naive reward → what you'd *expect* → what
  actually happened (clip) → the fix. The capstone: "reward design *is* the job."
- **Success:** ≥3 reproducible, visually-clear hacks documented with before/after.
- **⚠ verify:** which hacks induce cleanest on existing envs vs need a custom term.

### Cost gradient & risk sequencing

Deliberate order: prove the pipeline cheaply before committing to the expensive/risky
parts. S1 (hours, no code) and S4 (cheap, tiny code) come first; S3 (code + hours) and
S2 (long run) come after. If something is structurally broken, it surfaces in the cheap
runs — not 11 hours into a backflip.

## Track-A corpus layout (no moved files, no broken Pages URLs)

Existing `00`–`03` stay exactly where they are (they *are* the Foundations + Locomotion
spine of the reading order). New reports get **descriptive names**; `README.md` is
upgraded into the **syllabus** that carries the track grouping — tracks without the
false linear-ramp that renaming to `04/05/06…` would imply.

```
docs/reports/
  README.md                        ← upgraded to SYLLABUS: 4 tracks + per-track reading
                                     order + ready-to-run (deferred-spec) links
  00-primer.md                     Foundations               (existing)
  01-watching-it-learn.md          Locomotion                (existing)
  02-reproducing-the-benchmark.md  Locomotion                (existing)
  03-turning-the-knobs.md          Locomotion                (existing)
  running-and-flight.md            Locomotion — NEW          (S1)
  imitation-cartwheel.md           Imitation — NEW narrative entry; links to cartwheel-journey.md
  imitation-backflip.md            Imitation — NEW           (S2)
  getting-up.md                    From-scratch tasks — NEW  (S3)
  reward-hacking-gallery.md        Reward engineering — NEW  (S4)
  methods-reference.md             Reference handbook — NEW
  assets/                          existing + new per-report media
```

`cartwheel-journey.md` is preserved as the deep iteration log; `imitation-cartwheel.md`
is the narrative entry point that links to it.

### The tracks — a Foundations primer + four paradigm tracks

Foundations is the shared keystone (the vocabulary every track builds on); the four
*paradigm* tracks below are the four ways this repo teaches a robot to do something,
and "four paradigms" throughout this document refers to them (not to Foundations).

- **Foundations** — `00-primer` (the keystone vocabulary).
- **Locomotion** — `01`, `02`, `03`, `running-and-flight`. The "shape a reward from
  scratch; the gait is emergent" paradigm.
- **Imitation** — `imitation-cartwheel`, `imitation-backflip`. The "track a reference
  each step" paradigm.
- **From-scratch tasks** — `getting-up`. The "no reference, no command: behavior is
  100% the reward terms you wrote" paradigm.
- **Reward engineering** (cross-cutting) — `reward-hacking-gallery` + `methods-reference`.

## Methods & Techniques reference — contents

Consolidates the toolkit currently scattered across `CLAUDE.md`, `cartwheel-journey.md`,
and the helper scripts, in the same accessible voice but organized for lookup:

- **Reward-term catalog** — velocity (`track_lin/ang_vel`, `feet_air_time`,
  `base_height`, `upright`, `action_rate`, `joint_torques`); tracking (`anchor_pos`,
  `ee_body_pos`).
- **Terminations & thresholds** — the 0.25 → 0.5 m aerial lesson; *render at the
  training threshold*.
- **The retargeting pipeline** — MimicKit `.pkl` → CSV → `motion.npz` → train.
- **Curricula** — the velocity-range curriculum-clobber; the terrain curriculum.
- **Domain randomization** — note what is/isn't randomized.
- **Recording gotchas** — offscreen renderer mutations, shadow-map acne, debug-viz
  lines, holding a fixed command while recording.
- **Helper-script index** — `record_policy`, `record_learning_progression`,
  `watch_learning`, `play_motion_npz`, `plot_training_curves`, `score_cartwheel`,
  `smpl_backflip_to_g1`.
- **Ops/safety** — host quiesce, SIGINT to the python child (not the wrapping shell),
  `free -h` headroom, the bracket-trick for watcher loops.

## Spec files (`docs/superpowers/specs/`, dated 2026-06-19)

- `2026-06-19-g1-skills-curriculum-master.md` — this document.
- `2026-06-19-spine-running-flight.md` — full (S1).
- `2026-06-19-spine-backflip.md` — full (S2).
- `2026-06-19-spine-getup-recovery.md` — full (S3).
- `2026-06-19-spine-reward-hacking-gallery.md` — full (S4).
- `2026-06-19-tier1-gait-tweaks-compact.md` — compact (crouched, tiptoe, efficiency,
  spin, prescribed-gait, backward/sideways).
- `2026-06-19-tier2-acrobatics-compact.md` — compact (spinkick, jump, dance).
- `2026-06-19-tier3-tasks-compact.md` — compact (push-recovery, single-leg balance).
- `2026-06-19-tier4-objects-compact.md` — compact, spec-only (reach, kick-ball, carry).

Nine spec files total: "spec all," kept manageable by bundling the deferred long tail.

## Campaign sequencing (single-GPU, sequential)

1. **S1 Running** — cheap, no new code → proves the velocity-tweak loop fast.
2. **S4 Reward-hacking gallery** — cheap, several short killable runs; some specimens
   reuse S1/existing.
3. **S3 Get-up** — code-writing sub-step → container test → train (the engineering
   gate).
4. **S2 Backflip** — last; longest GPU run, on a known-good pipeline.

Each run bracketed by the CLAUDE.md **host-quiesce procedure**: stop co-tenant
containers (`open-webui`, `compose-arangodb-1`, `ollama-compose`), confirm `free -h`
headroom (~110 GiB for a ~13 GiB workload), keep `num_envs` bounded; `sudo`-gated
`swapoff` / `systemctl stop comfyui.service` are interactive-only and noted as such.
Runs are sequential (single GPU, one training job at a time).

**Checkpoint between tasks:** review artifacts → write that report → commit → start the
next. The syllabus `README.md` and the methods reference are drafted early (mostly
consolidation) and refined as runs land. Each report is written right after its run,
while context is fresh.

## Success criteria (program-level)

1. A no-background reader can enter the syllabus, pick any track, and explain that
   paradigm (what's rewarded, what emerges, what to tweak) — the existing bar, now
   across **four paradigms**.
2. **All candidate ideas have a spec** (full for the spine, compact for the rest)
   executable later without re-designing.
3. The **four spine tasks are trained to completion**, each with a published report +
   committed assets in the established style.
4. The methods reference gives a newcomer **one lookup surface** for the toolkit.
5. **Every trained behavior is visually verified, not score-asserted** — the cartwheel
   lesson, encoded as a program-level gate.

## Out of scope

- Tier-4 object-interaction *training* (spec-only this program; high research risk).
- Sim-to-real / hardware deployment.
- Quadruped (Go1) breadth.
- AMP / adversarial motion priors (noted as a future Imitation-track follow-up).

## Open questions to resolve during planning

Consolidated from the per-spine **⚠ verify** items; each resolved by reading a config or
running a probe inside `mjlab-dev` before the corresponding run:

1. **S3 (biggest):** mjlab's task/reward-manager API and initial-state randomization —
   read an existing task config; determine the cleanest way to spawn randomized fallen
   poses; check whether an upstream recovery task exists to fork.
2. **S1:** exact `feet_air_time` reward term name + the three `command_vel`
   curriculum-stage override keys; whether high speed needs episode-length/terrain
   changes.
3. **S2:** backflip reference feasibility/duration; thresholds that let a full flip
   complete and render uncut.
4. **S4:** cleanest reward-hack induction per specimen (existing-env CLI override vs a
   small custom term).
5. **Plotting:** metric export path (W&B API vs local tensorboard) — already solved in
   the walking arc; reuse that decision.
