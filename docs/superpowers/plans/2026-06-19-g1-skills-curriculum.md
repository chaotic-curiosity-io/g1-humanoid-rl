# G1 Skills Curriculum — Specs & Corpus Scaffolding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the complete spec set (nine files) and the Track-A corpus scaffolding for the multi-paradigm G1 skills curriculum, then run read-only Spark probes to resolve the open questions — everything knowable *without* burning GPU — so the four training campaigns can be planned just-in-time as follow-ons.

**Architecture:** Pure documentation + one read-only remote-probe task. We extend the existing `docs/reports/` series (00–03) into four paradigm tracks indexed by an upgraded syllabus `README.md`, write per-task specs at two depths (full for the four spine tasks, compact-bundled for the deferred long tail), consolidate the scattered toolkit into a methods reference, and create structured report drafts to be populated after each training run. No training runs and no mjlab source edits happen in this plan.

**Tech Stack:** Markdown (GitHub-flavored, served by GitHub Pages from `docs/`); git; `ssh spark` for read-only probes only. The master spec being implemented: `docs/superpowers/specs/2026-06-19-g1-skills-curriculum-master.md`.

## Global Constraints

- **Repo is PUBLIC** — everything committed here is world-readable.
- **No GPU, no training, no mjlab source edits in this plan.** The only Spark contact is the **read-only** probes in Task 12 (reading `params/*.yaml`, listing files, `--help` output). No `mjlab.scripts.train`.
- **Audience/voice (all reports):** reader with *no robotics or ML background*; define every term on first use; concrete analogies over notation; each report standalone and re-runnable.
- **Spec template** (full specs) = the walking-arc shape: Goal → Audience & voice → staged-arc table *with cost estimates* → concrete runs *with exact commands* → the experiment → artifacts & retrieval → ops & safety → success criteria → open questions.
- **Visual-verification gate** (recorded in specs, enforced in the training plans): every trained behavior is *visually verified, not score-asserted*.
- **Spec/plan locations & dates:** specs in `docs/superpowers/specs/` dated `2026-06-19`; this plan in `docs/superpowers/plans/`.
- **Do NOT stage the pre-existing working-tree `CLAUDE.md` modification.** Stage only files each task creates/modifies, by explicit path.
- **`.gitignore`:** never commit `*.mp4` except `docs/reports/assets/*.mp4`; never commit checkpoints/caches.
- **Branch:** `g1-skills-curriculum` (already created; master spec already committed there).
- **Commit footer** (every commit):
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01D6dhn7JiNfx8tpFbitRmgN
  ```

---

## File Structure

**Specs created (`docs/superpowers/specs/`):**
- `2026-06-19-spine-running-flight.md` — full spec, S1
- `2026-06-19-spine-backflip.md` — full spec, S2
- `2026-06-19-spine-getup-recovery.md` — full spec, S3
- `2026-06-19-spine-reward-hacking-gallery.md` — full spec, S4
- `2026-06-19-tier1-gait-tweaks-compact.md` — compact bundle (crouched, tiptoe, efficiency, spin, prescribed-gait, backward/sideways)
- `2026-06-19-tier2-acrobatics-compact.md` — compact bundle (spinkick, jump, dance)
- `2026-06-19-tier3-tasks-compact.md` — compact bundle (push-recovery, single-leg balance)
- `2026-06-19-tier4-objects-compact.md` — compact bundle, spec-only (reach, kick-ball, carry)

**Corpus created/modified (`docs/reports/`):**
- `methods-reference.md` — new reference handbook
- `README.md` — **modify**: upgrade into the tracked syllabus
- `running-and-flight.md`, `imitation-cartwheel.md`, `imitation-backflip.md`, `getting-up.md`, `reward-hacking-gallery.md` — new report drafts (structured skeleton + "populated after the S_n run" banner)

**Probe output:** findings appended to the **open-questions** section of the relevant spec files (Task 12).

**Verification note:** these are documentation deliverables, so each task's "test" is a *verification step* — the file exists, its required sections are present, it is internally consistent with the master spec, and its intra-repo links resolve. There is no pytest in this plan.

---

### Task 1: Full spec — S1 Running / flight phase

**Files:**
- Create: `docs/superpowers/specs/2026-06-19-spine-running-flight.md`

**Draws from:** master spec §"S1 · Running / flight phase" and §"Methods…"; existing `docs/reports/03-turning-the-knobs.md` (the velocity-range A/B it extends); CLAUDE.md "Velocity-range overrides — the curriculum clobber".
**Feeds into:** the S1 training plan (follow-on); linked from `running-and-flight.md` and the syllabus.

- [ ] **Step 1: Write the spec** using the full template. Required sections, each populated from the sources above (no placeholders):
  - **Goal** — train a fast G1 that develops a true flight phase; contrast with the existing walker.
  - **Audience & voice** — one paragraph, zero-background bar.
  - **Staged-arc table** — Replay existing walker (`model_2050.pt`, seconds) | Fresh fast run (~1–1.5 h GPU) | Record + cadence plot (minutes).
  - **Concrete runs** — exact `ssh spark "docker exec mjlab-dev bash -lc '...'"` command for `mjlab.scripts.train Mjlab-Velocity-Flat-Unitree-G1` with the high `lin-vel-x` range **and the three `command-vel` curriculum-stage overrides** (copy the `=`-syntax pattern from CLAUDE.md verbatim), plus a `feet_air_time` weight bump; and the `record_policy.py` command for the A/B clips.
  - **The experiment** — control = `model_2050.pt`; B = fast runner; one changed family of knobs.
  - **Artifacts & retrieval** — clips → `docs/reports/assets/`; cadence-vs-speed PNG via `plot_training_curves.py`/telemetry.
  - **Ops & safety** — the host-quiesce bracket (copy from CLAUDE.md).
  - **Success criteria** — visible flight phase; measurably different cadence/step-length; attributable to the changed knobs.
  - **Open questions** — exact `feet_air_time` term name; the three curriculum-stage override keys; whether high speed needs episode-length/terrain change. (These are resolved in Task 12.)
- [ ] **Step 2: Verify** — file opens; all nine template sections present; the curriculum-override snippet matches CLAUDE.md's `=` syntax; no `TODO`/`TBD` outside the Open-questions section.
- [ ] **Step 3: Commit**
  ```bash
  git add docs/superpowers/specs/2026-06-19-spine-running-flight.md
  git commit -m "Add full spec: S1 running / flight phase

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01D6dhn7JiNfx8tpFbitRmgN"
  ```

---

### Task 2: Full spec — S2 Backflip

**Files:**
- Create: `docs/superpowers/specs/2026-06-19-spine-backflip.md`

**Draws from:** master spec §"S2 · Backflip"; `docs/cartwheel-journey.md` (every hard-won lesson); CLAUDE.md "Cartwheel pipeline" + "Tracking terminations are a two-edged knife"; `scripts/smpl_backflip_to_g1.py`, `scripts/score_cartwheel.py`.
**Feeds into:** the S2 training plan; linked from `imitation-backflip.md` and the syllabus.

- [ ] **Step 1: Write the spec** using the full template:
  - **Goal** — a frame-confirmed single backflip via motion tracking, reusing the existing retargeter.
  - **Audience & voice** — zero-background bar.
  - **Staged-arc table** — Retarget reference (minutes) | Train tracking (~hours, the heaviest spine run) | Record at matched threshold + visual review.
  - **Concrete runs** — exact commands: `smpl_backflip_to_g1.py` → `csv_to_npz` → `mjlab.scripts.train Mjlab-Tracking-Flat-Unitree-G1 --env.commands.motion.motion-file <npz>`; `record_policy.py --termination-threshold 0.5` (or `--disable-terminations`) for uncut render.
  - **The experiment** — likely iterative references (iterA/B/C), single feasible flip, 0.5 m thresholds.
  - **Artifacts & retrieval** — multi-camera clips → `docs/reports/assets/`.
  - **Ops & safety** — host-quiesce bracket; multi-hour run handling (SIGINT to the python child).
  - **Success criteria** — *frame-by-frame-confirmed* takeoff→rotation→landing; **score must not be trusted alone** (state the `score_cartwheel.py` spoof explicitly).
  - **Open questions** — reference feasibility/duration; thresholds that let a full flip complete and render uncut.
- [ ] **Step 2: Verify** — all sections present; the visual-verification gate is stated as a success criterion (not advice); render command uses a threshold ≥ training threshold or `--disable-terminations`.
- [ ] **Step 3: Commit** (same footer; message `Add full spec: S2 backflip`).

---

### Task 3: Full spec — S3 Get-up / fall recovery

**Files:**
- Create: `docs/superpowers/specs/2026-06-19-spine-getup-recovery.md`

**Draws from:** master spec §"S3 · Get-up / fall recovery"; CLAUDE.md "Editing mjlab source" + "Pre-commit checks (when touching mjlab/)" + the velocity task description.
**Feeds into:** the S3 training plan (the only follow-on with real new-code + pytest/pyright cycles); linked from `getting-up.md` and the syllabus.

- [ ] **Step 1: Write the spec** using the full template, with an explicit **new-code sub-stage**:
  - **Goal** — a from-scratch task (no reference, no command) where the robot learns to stand up from a randomized fallen pose; the behavior is 100% the reward terms.
  - **Audience & voice** — zero-background bar.
  - **Staged-arc table** — Write the new task (`Mjlab-Recovery-Flat-Unitree-G1`) + reward manager + fallen-pose init (code + container test) | Smoke-train probe | Full train (~hours, iterative shaping) | Record + review.
  - **Concrete runs** — `docker cp`/in-container edit path (from CLAUDE.md); the `ruff format && ruff check --fix && pyright` + `pytest` pre-commit commands; the smoke-train then full-train commands.
  - **The experiment** — reward = base-height-rising + uprightness + (once up) stillness; terminate on success-hold/timeout, **not on falling**; no twist command.
  - **Artifacts & retrieval** — flail→stand clips; an honest shaping-iteration log (cross-link to S4).
  - **Ops & safety** — host-quiesce; the mjlab-edit ownership rules (no `sudo chown`).
  - **Success criteria** — reliably reaches and *holds* a stable stand from randomized fallen poses; strategy visible; shaping journey documented.
  - **Open questions** — mjlab task/reward-manager API + initial-state randomization; whether an upstream recovery task exists to fork; cleanest fallen-pose spawn. (Largest unknown; Task 12 probes it.)
- [ ] **Step 2: Verify** — all sections present; the new-code sub-stage names the exact pre-commit commands from CLAUDE.md; terminations explicitly exclude "falling".
- [ ] **Step 3: Commit** (footer; message `Add full spec: S3 get-up / fall recovery`).

---

### Task 4: Full spec — S4 Reward-hacking gallery

**Files:**
- Create: `docs/superpowers/specs/2026-06-19-spine-reward-hacking-gallery.md`

**Draws from:** master spec §"S4 · Reward-hacking gallery"; CLAUDE.md "Quantitative scorers lie" + `scripts/score_cartwheel.py`; existing `docs/reports/03-turning-the-knobs.md` ("higher reward ≠ better robot").
**Feeds into:** the S4 training plan; linked from `reward-hacking-gallery.md` and the syllabus.

- [ ] **Step 1: Write the spec** using the full template:
  - **Goal** — a curated gallery of deliberately-naive rewards, each documenting a cheat and its fix; the capstone that teaches "reward design *is* the job".
  - **Audience & voice** — zero-background bar.
  - **Staged-arc table** — per specimen: induce (short killable run) | capture clip | document fix.
  - **Concrete runs** — for each of ≥3 specimens: the naive reward (CLI weight override or small term) on the existing velocity/tracking env; the `record_policy.py` capture command; include the *already-in-hand* `score_cartwheel.py`-fooled-by-crash-roll specimen.
  - **The experiment** — per specimen: naive reward → expected → actual (clip) → fix.
  - **Artifacts & retrieval** — before/after clips → `docs/reports/assets/`.
  - **Ops & safety** — host-quiesce (light; runs are short/killable).
  - **Success criteria** — ≥3 reproducible, visually-clear hacks with before/after.
  - **Open questions** — cleanest induction per specimen (existing-env override vs custom term).
- [ ] **Step 2: Verify** — all sections present; ≥3 concrete specimens with exact reward changes; the crash-roll specimen referenced.
- [ ] **Step 3: Commit** (footer; message `Add full spec: S4 reward-hacking gallery`).

---

### Task 5: Compact bundle — Tier 1 deferred gait tweaks

**Files:**
- Create: `docs/superpowers/specs/2026-06-19-tier1-gait-tweaks-compact.md`

**Draws from:** master spec tier table + the original idea list; CLAUDE.md velocity overrides.
**Feeds into:** future promotion to full runs; linked from the syllabus "ready-to-run".

- [ ] **Step 1: Write the bundle** — a short intro, then one **compact spec section per task**: crouched "Groucho", tiptoe, energy-efficiency, spin-in-place, prescribed-gait (hop/march), backward/sideways. Each section = **Goal** (1–2 sentences) · **Env + reward terms to touch** · **New-code flag** · **Train/record commands** (the velocity command + curriculum-stage override pattern, parameterized) · **Cost estimate** · **Success criteria**. The prescribed-gait section flags "light new code: a contact-schedule reward term".
- [ ] **Step 2: Verify** — six sections, each with all six compact fields; no field left as "TODO".
- [ ] **Step 3: Commit** (footer; message `Add compact specs: Tier-1 deferred gait tweaks`).

---

### Task 6: Compact bundle — Tier 2 deferred acrobatics

**Files:**
- Create: `docs/superpowers/specs/2026-06-19-tier2-acrobatics-compact.md`

**Draws from:** master spec; `docs/cartwheel-journey.md`; the cartwheel pipeline in CLAUDE.md.
**Feeds into:** future promotion; syllabus "ready-to-run".

- [ ] **Step 1: Write the bundle** — one compact section each for: spinkick/martial-arts, jump (note "can be tracking *or* a task reward"), dance. Same six compact fields as Task 5. Each reuses the retargeting pipeline; note reference-sourcing per task (spinkick example already in the pipeline; dance from SMPL mocap).
- [ ] **Step 2: Verify** — three sections, six fields each; pipeline reuse stated.
- [ ] **Step 3: Commit** (footer; message `Add compact specs: Tier-2 deferred acrobatics`).

---

### Task 7: Compact bundle — Tier 3 deferred from-scratch tasks

**Files:**
- Create: `docs/superpowers/specs/2026-06-19-tier3-tasks-compact.md`

**Draws from:** master spec; the S3 get-up full spec (Task 3) as the template these mirror.
**Feeds into:** future promotion; syllabus "ready-to-run".

- [ ] **Step 1: Write the bundle** — one compact section each for: push-recovery (note "needs external-force application"), single-leg / flamingo balance. Same six compact fields; **New-code flag = yes** for both (new task + reward manager), cross-referencing the get-up spec's approach.
- [ ] **Step 2: Verify** — two sections, six fields each; new-code flag set; get-up cross-reference present.
- [ ] **Step 3: Commit** (footer; message `Add compact specs: Tier-3 deferred from-scratch tasks`).

---

### Task 8: Compact bundle — Tier 4 object / whole-body (spec-only)

**Files:**
- Create: `docs/superpowers/specs/2026-06-19-tier4-objects-compact.md`

**Draws from:** master spec §"Out of scope" (Tier-4 training is deferred) + the idea list.
**Feeds into:** syllabus "ready-to-run" (clearly labelled highest-risk, spec-only).

- [ ] **Step 1: Write the bundle** — one compact section each for: reach-to-target, kick-a-ball, carry. Same six compact fields; each flags **New-code = yes, most** (MJCF scene edits to add a free body + new rewards) and **highest research risk**. A header states Tier 4 is spec-only in this program.
- [ ] **Step 2: Verify** — three sections, six fields each; spec-only/highest-risk header present.
- [ ] **Step 3: Commit** (footer; message `Add compact specs: Tier-4 object/whole-body (spec-only)`).

---

### Task 9: Methods & Techniques reference

**Files:**
- Create: `docs/reports/methods-reference.md`

**Draws from:** CLAUDE.md (reward/termination/curriculum/recording/ops sections), `docs/cartwheel-journey.md`, the `scripts/` helper docstrings.
**Feeds into:** linked from the syllabus (Reward-engineering track) and cross-linked by every report.

- [ ] **Step 1: Write the handbook** organized for lookup, in the accessible voice, with these sections (each populated from the sources, no placeholders): Reward-term catalog (velocity + tracking terms) · Terminations & thresholds (the 0.25→0.5 m lesson; render-at-threshold) · Retargeting pipeline (`.pkl`→CSV→`npz`→train) · Curricula (the velocity curriculum-clobber; terrain) · Domain randomization · Recording gotchas (renderer mutations, shadow acne, debug-viz lines, fixed command) · Helper-script index (all of `scripts/`) · Ops/safety (host-quiesce, SIGINT-to-child, the bracket-trick).
- [ ] **Step 2: Verify** — all eight sections present; each helper script in `scripts/` appears in the index; terminal/term names match CLAUDE.md.
- [ ] **Step 3: Commit** (footer; message `Add methods & techniques reference handbook`).

---

### Task 10: Upgrade README into the tracked syllabus

**Files:**
- Modify: `docs/reports/README.md`

**Draws from:** master spec §"Track-A corpus layout" + §"The tracks"; existing README content (preserve the "Media"/"How these were produced" notes).
**Feeds into:** the published Pages landing for the series; links every report + spec.

- [ ] **Step 1: Rewrite README** as the syllabus: a short intro, then the **five groupings** — Foundations (`00`), Locomotion (`01`,`02`,`03`,`running-and-flight`), Imitation (`imitation-cartwheel`→links `cartwheel-journey.md`, `imitation-backflip`), From-scratch tasks (`getting-up`), Reward engineering (`reward-hacking-gallery`, `methods-reference`). Each entry: title + one-line hook + link. Add a **"Ready-to-run (specs, not yet trained)"** section linking the four compact tier bundles. Mark not-yet-populated reports `(draft — populated after the S_n run)`. Preserve the existing Media/`.gitignore`-exception note.
- [ ] **Step 2: Verify** — every linked file path exists (the report drafts from Task 11 and specs from Tasks 1–8) or is clearly marked draft; the four paradigm tracks are distinguished from the Foundations primer (consistency with the master spec's count).
- [ ] **Step 3: Commit** (footer; message `Upgrade reports README into tracked syllabus`).

---

### Task 11: Create the new report drafts (skeletons)

**Files:**
- Create: `docs/reports/running-and-flight.md`, `docs/reports/imitation-cartwheel.md`, `docs/reports/imitation-backflip.md`, `docs/reports/getting-up.md`, `docs/reports/reward-hacking-gallery.md`

**Draws from:** the structure of existing `01`/`03` reports (the proven report shape); the four full spine specs (Tasks 1–4) for each report's planned content; `cartwheel-journey.md` for `imitation-cartwheel.md`.
**Feeds into:** populated after each training campaign; linked from the syllabus.

- [ ] **Step 1: Create each draft** with a visible top banner — `> **Draft.** This report is populated after the S_n training run; see [the spec](../superpowers/specs/...) for the plan.` — followed by the agreed section skeleton (plain-language intro → the exact command → results placeholder for plots/clips → "tweak this to explore"). `imitation-cartwheel.md` additionally summarizes and links `cartwheel-journey.md`. Banners keep half-finished Pages honest until the runs land.
- [ ] **Step 2: Verify** — all five files exist; each has the draft banner + skeleton headings; each links its spec; no empty report is presented as finished.
- [ ] **Step 3: Commit** (footer; message `Add report drafts (skeletons) for the five new reports`).

---

### Task 12: Read-only Spark probes — resolve open questions

**Files:**
- Modify: the **Open questions** section of `2026-06-19-spine-running-flight.md`, `2026-06-19-spine-getup-recovery.md`, `2026-06-19-spine-backflip.md`, `2026-06-19-spine-reward-hacking-gallery.md` (append findings).

**Draws from:** the `⚠ verify on Spark` items consolidated in the master spec's "Open questions".
**Feeds into:** the four follow-on training plans (turns unknowns into knowns).

**This task touches the Spark but is strictly READ-ONLY — no training, no source edits.** If the Spark or `mjlab-dev` is unreachable, record that and leave the open-questions sections as-is (the training plans can probe at execution time).

- [ ] **Step 1: Probe S1 unknowns** — read the baseline run's config and confirm reward/curriculum keys:
  ```bash
  ssh spark "docker exec mjlab-dev bash -lc 'sed -n 1,200p /workspace/mjlab/logs/rsl_rl/g1_velocity/2026-04-17_18-46-23/params/env.yaml'"
  ```
  Expected: YAML showing the reward terms (locate the air-time term's exact name) and the `command_vel` curriculum stages (confirm the three override keys). Record the exact names.
- [ ] **Step 2: Probe S3 unknowns** — inspect mjlab's task/reward-manager structure and look for any existing recovery task:
  ```bash
  ssh spark "docker exec mjlab-dev bash -lc 'ls /workspace/mjlab/src/mjlab/tasks && grep -ril \"recover\\|getup\\|get_up\\|stand_up\" /workspace/mjlab/src/mjlab || true'"
  ```
  Expected: the task package layout + whether a recovery task exists to fork. Record findings.
- [ ] **Step 3: Probe S2/S4 helpers** — confirm the record/scoring helpers and tracking motion-file flag exist as the specs assume:
  ```bash
  ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && python scripts/record_policy.py --help 2>&1 | sed -n 1,60p'"
  ```
  Expected: the `--termination-threshold` / `--disable-terminations` flags present (S2/S4). Record confirmation.
- [ ] **Step 4: Append findings** to each spec's Open-questions section — replace each resolved `⚠ verify` line with the concrete value found (or "probe deferred — Spark unreachable"). Keep unresolved items clearly marked.
- [ ] **Step 5: Commit**
  ```bash
  git add docs/superpowers/specs/2026-06-19-spine-*.md
  git commit -m "Resolve open questions from read-only Spark probes

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01D6dhn7JiNfx8tpFbitRmgN"
  ```

---

## After this plan — the four training campaigns (follow-on plans)

Once Task 12 turns the unknowns into knowns, write one just-in-time plan per spine task, in execution order, each its own `docs/superpowers/plans/` file with real test cycles appropriate to its work (pytest/pyright for S3's new mjlab code; visual-verification + artifact checks for the runs):

1. **S1 Running** → train, record A/B, populate `running-and-flight.md`.
2. **S4 Reward-hacking gallery** → induce ≥3 hacks, populate `reward-hacking-gallery.md`.
3. **S3 Get-up** → write+test the new task, train with shaping, populate `getting-up.md`.
4. **S2 Backflip** → retarget, train (longest), populate `imitation-backflip.md`.

Each follow-on run is bracketed by the host-quiesce procedure and ends with its report committed before the next begins.

---

## Self-Review

**1. Spec coverage** — every master-spec deliverable maps to a task here: nine spec files → Tasks 1–8; methods reference → Task 9; syllabus → Task 10; report drafts → Task 11; open-questions resolution → Task 12. The *trained policies + populated reports* (success criteria 3) are explicitly deferred to the named follow-on plans (training cannot happen in a no-GPU plan) — this is a sequencing decision, not a gap.

**2. Placeholder scan** — the report *drafts* (Task 11) are intentionally skeletal but each carries a visible Draft banner and full section headings + spec link (not silent stubs); the spec Open-questions sections carry `⚠ verify` items by design (resolved in Task 12), mirroring the walking spec's own "open questions" pattern. No `TODO`/"fill in later" appears in any task's *instructions*.

**3. Type/name consistency** — spec filenames are identical between the File Structure block, each task, and the syllabus links (`2026-06-19-spine-*`, `2026-06-19-tier{1..4}-*-compact`); report filenames (`running-and-flight`, `imitation-cartwheel`, `imitation-backflip`, `getting-up`, `reward-hacking-gallery`) are identical across Tasks 10, 11, and the follow-on list; the `S1/S2/S3/S4` labels match the master spec throughout.
