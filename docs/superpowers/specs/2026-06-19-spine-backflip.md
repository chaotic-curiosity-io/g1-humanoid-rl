# Spec: S2 — Backflip (Motion Imitation)

**Date:** 2026-06-19
**Branch:** `g1-skills-curriculum`
**Status:** Ready for training campaign
**Track:** Imitation (spine task 2 of 4)
**Feeds into:** `docs/reports/imitation-backflip.md`; linked from the syllabus `docs/reports/README.md`

---

## Goal

Train a Unitree G1 humanoid to perform a single complete backflip — takeoff, full rotation, landing — by teaching it to imitate a pre-recorded reference motion. The result is verified frame by frame in a recorded video, not by a numerical score.

**What is motion imitation?** Instead of writing a reward that says "do a backflip" in abstract terms, we give the training system a reference: a frame-by-frame recording of a human backflip that has been mathematically mapped ("retargeted") onto the G1's body. The policy is then rewarded for making the robot's joints match the reference as closely as possible at each timestep. If the robot drifts too far from the reference, the simulation resets. Over thousands of iterations, the robot learns to follow the reference faithfully enough to complete the full motion.

**What is a "policy"?** A neural network that reads the robot's sensor data every 20 ms and outputs motor commands. Training is the process of adjusting that network's millions of numbers so the robot's behavior matches what we want.

**Why a backflip?** It is the natural companion to the cartwheel (already trained in this repo), and the hardest single-skill test of the imitation pipeline: it requires a genuine airborne phase, full backward rotation, and a stable landing — each of which is an independent learning challenge. The cartwheel journey revealed every hard-won lesson about this pipeline; the backflip applies them all from the start.

This is the spine task for the **Imitation track** of the skills curriculum. It is placed last in the execution order because it is the longest GPU run and depends on a proven pipeline (S1, S4, and S3 confirm the infrastructure before we commit to multiple training hours here).

---

## Audience & voice

This report is written for someone with no background in robotics or machine learning. Every term is defined on first use.

- **Policy:** the neural network that controls the robot — its "brain."
- **Reward:** a number the training algorithm uses as feedback. Higher reward means the robot did something we wanted.
- **Episode:** one uninterrupted simulation run. The robot starts at frame 0 of the reference motion, the policy runs, and eventually the simulation resets (because the robot drifted too far, time ran out, or the motion completed).
- **Termination / termination threshold:** a rule that cuts an episode short when the robot's pose drifts more than a set distance from the reference. At 0.25 m it is very strict; at 0.5 m it is lenient enough for an aerial motion where limbs sweep widely.
- **Retargeting:** the process of mathematically re-expressing a human motion (stored as SMPL body-model pose data) in terms of the G1 robot's joints and proportions.
- **NPZ / motion file:** a compressed NumPy file (`.npz`) that stores the retargeted reference motion — the target the policy tries to match each frame.
- **Flight phase / inversion:** the part of the backflip where the robot is fully airborne and its body has rotated past horizontal. In a completed backflip, the robot goes upside-down; in a crash-roll, it tumbles on the ground instead.

Concrete analogies are preferred over mathematical notation. Each report is standalone: a reader who picks it up without reading the rest of the series can follow the reasoning.

---

## Staged-arc table

| Stage | What runs on the Spark | Estimated cost | Output |
|---|---|---|---|
| **Retarget reference** | Run `smpl_backflip_to_g1.py` to convert MimicKit's backflip `.pkl` → G1 `.pkl`; then `pkl_to_csv.py` to produce a `.csv`; then `mjlab.scripts.csv_to_npz` to produce `backflip.npz` (confirm at run time whether `csv_to_npz` ingests the gmr `.pkl` directly or requires the `pkl_to_csv` CSV step); visually inspect duration and single-flip integrity via `play_motion_npz.py` | ~5–15 minutes (CPU, no GPU) | `backflip.npz` reference motion; confirmed single flip, feasible duration |
| **Train (iterA — initial attempt)** | `mjlab.scripts.train Mjlab-Tracking-Flat-Unitree-G1` with the retargeted `backflip.npz`, 0.5 m termination thresholds, 4096 envs, 20000 iterations | ~8–12 h GPU (the spine's heaviest run; the cartwheel took ~11 h) | Checkpoint `model_19999.pt`; W&B run; `params/env.yaml` confirming thresholds |
| **Record + visual review** | `record_policy.py --termination-threshold 0.5` (matching training) or `--disable-terminations`; frame-by-frame review of the rendered video | ~10 minutes (CPU-runnable) | Multi-camera MP4s; pass/fail visual verdict |
| **Iterate if needed (iterB, iterC…)** | Adjust reference duration/quality or threshold and retrain — expected from cartwheel precedent | Additional ~hours per iter | Improved checkpoint; updated visual verdict |

The recording step is CPU-runnable and can overlap with other work on the GPU.

---

## Concrete runs

All commands are issued from the Windows host and execute inside `mjlab-dev` on the Spark.

### Step 1 — Retarget the backflip reference

`smpl_backflip_to_g1.py` takes MimicKit's SMPL backflip `.pkl`, performs forward kinematics using MimicKit's shipped `smpl.xml` MJCF skeleton, and calls GMR's retargeter to produce a G1-compatible `.pkl`. It bypasses SMPL-X body-model weights entirely, which avoids the license and dependency issues that made automatic retargeters unreliable for the cartwheel's aerial phase.

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && python scripts/smpl_backflip_to_g1.py'"
```

This writes the retargeted pkl to `/workspace/pose-pipeline/outputs/gmr_pkl/smpl_backflip_to_g1.pkl`. Then convert it to the NPZ format that mjlab's tracking task expects:

```bash
# Confirm at run time whether csv_to_npz ingests the gmr .pkl directly or requires the pkl_to_csv CSV step.
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && \
  python scripts/pkl_to_csv.py \
  /workspace/pose-pipeline/outputs/gmr_pkl/smpl_backflip_to_g1.pkl \
  /workspace/pose-pipeline/outputs/gmr_pkl/smpl_backflip_to_g1.csv'"

ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && \
  python -m mjlab.scripts.csv_to_npz \
  /workspace/pose-pipeline/outputs/gmr_pkl/smpl_backflip_to_g1.csv \
  /workspace/pose-pipeline/motions/backflip.npz'"
```

Before training, inspect the reference visually to confirm it represents a single backflip of feasible duration (not accidentally two flips or a truncated motion — the exact failure mode that derailed cartwheel iter B):

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/play_motion_npz.py \
  --motion /workspace/pose-pipeline/motions/backflip.npz \
  --output /workspace/clips/s2_reference_preview.mp4'"
```

Pull the preview clip and watch it before proceeding. The clip should show: a single continuous backward rotation from standing through inversion to landing. If the clip shows two flips, or stops mid-motion, fix the reference first.

```bash
scp spark:/workspace/clips/s2_reference_preview.mp4 docs/reports/assets/
```

### Step 2 — Train the tracking policy (iterA)

The key lessons from the cartwheel:

- **0.5 m termination thresholds**, not the default 0.25 m. At 0.25 m the policy never sees a completed flip — legs and arms sweep through a wide arc during the airborne phase, and a tight threshold cuts every attempt before the landing half. Raising to 0.5 m gives the policy enough tolerance to experience the full motion and learn from it.
- **Single, feasible reference** only. The cartwheel iter B used a reference that accidentally contained two back-to-back cartwheels (a `--duration` bug in `pkl_to_csv`). The policy could not finish even one. Confirm the reference is a single flip before starting.
- **From scratch** (no warm-start from the cartwheel checkpoint). The backflip is a different motion with a different airborne trajectory; warm-starting from cartwheel would bias the initialization wrongly.

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file /workspace/pose-pipeline/motions/backflip.npz \
  --env.terminations.anchor-pos.threshold 0.5 \
  --env.terminations.ee-body-pos.threshold 0.5 \
  --agent.num-envs 4096 \
  --agent.max-iterations 20000 \
  --agent.seed 42'"
```

After the run completes, verify the thresholds landed:

```bash
ssh spark "docker exec mjlab-dev bash -lc 'grep -A3 threshold \
  /workspace/mjlab/logs/rsl_rl/g1_tracking/<timestamp>/params/env.yaml'"
```

Replace `<timestamp>` with the actual run directory printed by the trainer at startup. Both `anchor_pos` and `ee_body_pos` must show `0.5`; if they still show `0.25`, the threshold override did not land and the run must be repeated with corrected flag syntax.

### Step 3 — Record at matched threshold (or with terminations off)

**Prerequisite:** `record_policy.py` is not present in the container. Copy it before recording:

```bash
scp scripts/record_policy.py spark:/tmp/
ssh spark "docker cp /tmp/record_policy.py mjlab-dev:/workspace/scripts/record_policy.py"
```

**Critical:** the render threshold must match or relax the training threshold. If training used 0.5 m but the render uses the default 0.25 m, every backflip attempt is cut short mid-flip on screen — the exact bug that made iter B's video look like repeated failures even though the policy was making genuine progress. Use `--termination-threshold 0.5` to match training exactly, or `--disable-terminations 1` to run the policy continuously and see its full behavior.

```bash
# Match training threshold (recommended first pass):
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task-id Mjlab-Tracking-Flat-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_tracking/<timestamp>/model_19999.pt \
  --motion-file /workspace/pose-pipeline/motions/backflip.npz \
  --termination-threshold 0.5 \
  --output-dir /workspace/clips/s2_iterA_side'"

# Alternatively — disable terminations to see continuous policy behavior:
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task-id Mjlab-Tracking-Flat-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_tracking/<timestamp>/model_19999.pt \
  --motion-file /workspace/pose-pipeline/motions/backflip.npz \
  --disable-terminations 1 \
  --output-dir /workspace/clips/s2_iterA_noterminations'"
```

Record multiple camera angles for the final confirmed policy (chase, side, front, top, grid):

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task-id Mjlab-Tracking-Flat-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_tracking/<timestamp>/model_19999.pt \
  --motion-file /workspace/pose-pipeline/motions/backflip.npz \
  --disable-terminations 1 \
  --cameras chase side front top grid \
  --output-dir /workspace/clips/s2_final'"
```

Pull all clips to the Windows box:

```bash
scp spark:/workspace/clips/s2_*.mp4 docs/reports/assets/
```

### Step 4 — Score (for logging only) and frame-by-frame review

Run `score_cartwheel.py` to get a numerical completion count, but treat the number as a starting point for visual review — not a verdict:

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && \
  MUJOCO_GL=egl python scripts/record_policy.py \
    --task-id Mjlab-Tracking-Flat-Unitree-G1 \
    --checkpoint-file logs/rsl_rl/g1_tracking/<timestamp>/model_19999.pt \
    --motion-file /workspace/pose-pipeline/motions/backflip.npz \
    --disable-terminations 1 \
    --dump-telemetry /workspace/clips/s2_telemetry.npz \
    --output-dir /workspace/clips/s2_for_scoring && \
  python scripts/score_cartwheel.py --telemetry /workspace/clips/s2_telemetry.npz'"
```

Then open the side-view clip on the Windows box and step through it frame by frame (any video player with frame-step: VLC `,`/`.` keys, or Windows Video Player frame-forward). Look for:

1. **Takeoff** — the robot leaves the ground (no foot contact).
2. **Inversion** — the body rotates past horizontal; the head goes below the hips.
3. **Landing** — feet re-contact the ground; the robot stands upright (not face-planted).

These three phases must all be visible in at least one episode of the 40-second video for a confirmed backflip. The scorer alone cannot confirm this — see Success criteria.

---

## The experiment

**Method:** motion imitation (also called "motion tracking"). The policy is rewarded each step for how closely the robot's joint angles, body positions, and velocities match the corresponding frame of the backflip reference. There is no separate hand-crafted "do a backflip" reward — the behavior emerges from tracking the reference accurately enough to complete the full motion.

**Reference:** MimicKit's shipped `smpl_backflip.pkl`, retargeted to G1 via `smpl_backflip_to_g1.py`. MimicKit ships hand-curated retargets for the G1 that outperform automated retargeters for acrobatic motions (the cartwheel journey showed that SMPL-X automatic retargeters degrade during aerial phases, where body-model fits are noisiest). Using MimicKit's own assets for forward kinematics avoids the license and dependency issues of full SMPL-X body model weights.

**Iteration strategy:** The cartwheel required three iterations (iterA, B, C) before converging on a working policy. Expect the backflip to follow a similar arc:

- **IterA:** train from scratch with a clean single-flip reference and 0.5 m thresholds. Review the rendered video. If the policy shows genuine airborne phases (even imperfect ones), continue to iterB with a warm start and more iterations. If the policy flops or face-plants, inspect whether the reference is feasible (check `play_motion_npz.py` output) and whether thresholds need further adjustment.
- **IterB and beyond:** adjust the reference duration, threshold, or (rarely) reward weights based on what the video shows. Never trust the score alone; always review the video before declaring an iteration successful or failed.

**The cartwheel lessons applied here from the start:**

| Lesson | What went wrong | How we apply it here |
|---|---|---|
| Single feasible reference | Cartwheel iterB used a double-cartwheel reference (2× duration due to `--duration` bug); policy could not finish even one | Visually inspect `play_motion_npz.py` output before training; confirm single flip of feasible duration (~2–4 s) |
| 0.5 m termination thresholds | Cartwheel iterA: default 0.25 m cut every attempt mid-flip; policy never experienced a completed rotation | Set `anchor-pos` and `ee-body-pos` thresholds to 0.5 m from the start |
| Render at training threshold | Cartwheel iterB: render used default 0.25 m; every attempt was cut on screen mid-flip, hiding genuine policy progress | Use `--termination-threshold 0.5` or `--disable-terminations` in every render |
| Scorer is spoofable | Cartwheel iterB: scorer counted crash-roll-through-180° as "inversion"; 95 % score on a policy that face-planted | Visual frame-by-frame review is the only valid verdict; scorer is informational only |

---

## Artifacts & retrieval

| Artifact | Location on Spark | Committed to repo |
|---|---|---|
| Reference preview clip | `/workspace/clips/s2_reference_preview.mp4` | `docs/reports/assets/s2_reference_preview.mp4` |
| Final multi-camera clips (chase, side, front, top, grid) | `/workspace/clips/s2_final_*.mp4` | `docs/reports/assets/s2_final_*.mp4` |
| Scoring telemetry | `/workspace/clips/s2_telemetry.npz` | NOT committed — stays on Spark |
| Final policy checkpoint | `logs/rsl_rl/g1_tracking/<timestamp>/model_19999.pt` | NOT committed — stays on Spark |
| W&B training run | `donb-chaotic-curiosity` entity | Link embedded in `imitation-backflip.md` |
| Retargeted reference NPZ | `/workspace/pose-pipeline/motions/backflip.npz` | NOT committed — stays on Spark |

MP4 files are committed under `docs/reports/assets/` only — `.gitignore` contains the deliberate exception `!docs/reports/assets/*.mp4`. Do not commit MP4s anywhere else. Do not commit checkpoints or `.npz` motion files.

---

## Ops & safety

The Spark's unified memory is shared by co-tenant containers (`open-webui`, `compose-arangodb-1`, `ollama-compose`, `comfyui`). Under memory contention the system enters a swap-death-spiral and hard-reboots — not a clean CUDA out-of-memory error. The S2 training run is the spine's heaviest (~8–12 h), making the host-quiesce bracket especially important here.

**Before the training run (on the Spark):**

```sh
docker stop open-webui compose-arangodb-1 ollama-compose
# Confirm headroom: the workload needs ~13 GiB; aim for ~110 GiB free
free -h
# swapoff -a and systemctl stop comfyui.service need an interactive sudo session
# if memory is tight, open an interactive session first: ssh spark then sudo swapoff -a
```

**During the run:**

Training is launched as a single foreground command (or backgrounded explicitly). To stop it cleanly, send SIGINT directly to the Python child — the `bash -c '... > log'` wrapper does not propagate SIGINT on the first try:

```sh
docker exec mjlab-dev bash -lc "kill -INT \$(pgrep -f mjlab.scripts.train)"
```

Never use a bare `pgrep -f` or `grep` of the train command in a counting or watcher loop — it matches its own command line and loops forever or miscounts. Use the bracket trick instead: `ps -eo cmd | grep "[p]ython -u -m mjlab.scripts.train"`.

**For a multi-hour run from the Windows host**, launch with a background redirect so it survives SSH disconnect:

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && nohup python -m mjlab.scripts.train \
  Mjlab-Tracking-Flat-Unitree-G1 \
  ... \
  > /workspace/logs/s2_train.log 2>&1 &'"
```

Monitor progress periodically:

```bash
ssh spark "tail -f /workspace/logs/s2_train.log"
```

**After the run (always, even on failure):**

```sh
docker start open-webui compose-arangodb-1 ollama-compose
```

**Viser:** training prints a `http://localhost:8080` URL from inside the container; translate to `:8081` on the host (port 8080 is owned by open-webui when running, but open-webui is stopped during training).

**Recording:** the offscreen renderer runs on CPU, so recording can happen while the GPU is busy with a subsequent training iteration or a different job. Run `record_policy.py` without `MUJOCO_GL=egl` only if EGL is unavailable; with EGL it is significantly faster.

---

## Success criteria

The following criteria must **all** be met before the S2 run is declared successful and `imitation-backflip.md` is published.

1. **Frame-by-frame-confirmed completed backflip.** In the rendered video (recorded with `--termination-threshold 0.5` or `--disable-terminations`), at least one episode visually shows all three phases: takeoff (no foot contact with ground), inversion (head below hips, body past horizontal), and landing (feet re-contact ground, robot stands upright). This is confirmed by stepping through the clip frame by frame — not by reading a score.

2. **`score_cartwheel.py` alone is never sufficient.** The scorer is roll-angle-only: it counts any episode where the pelvis passes |roll| > 150° as "inverted" and the next standing frame as "recovered." A crash-roll — where the robot tumbles face-first on the ground, rolling through 180° before coming to rest — satisfies both conditions. A score of "8/9 completions" is therefore compatible with zero actual backflips (this is what happened in cartwheel iterB at the 95 % completion rate claim). The score is logged for the report but the visual gate is the decisive criterion.

3. **Render threshold matches or relaxes training threshold.** The render must use `--termination-threshold 0.5` (matching training) or `--disable-terminations`. Rendering with the default 0.25 m threshold on a policy trained at 0.5 m cuts every attempt mid-flip on screen, making genuine progress invisible. This is verified by checking the `record_policy.py` command used for the final clips.

4. **Reference integrity confirmed.** The `play_motion_npz.py` preview shows a single backflip of feasible duration (approximately 2–4 seconds of motion content, not two back-to-back flips and not a truncated clip). This is checked before training begins and documented in the iteration log.

5. **Report written and published.** `docs/reports/imitation-backflip.md` is populated with the final multi-camera clips, the visual verification statement, the iteration log (including any failed iterations and what they revealed), and a plain-language explanation, committed to `main` and visible on GitHub Pages.

---

## Open questions

The following items were targeted for resolution by the read-only Spark probe in Task 12 of the implementation plan.

**Probe run: 2026-06-19 — `cat scripts/record_policy.py` (local checkout) + `python -m mjlab.scripts.csv_to_npz --help` (container)**

1. **Reference feasibility and duration — deferred (requires retargeting first).** This item cannot be resolved without running `smpl_backflip_to_g1.py` and `csv_to_npz` to produce `backflip.npz`, then inspecting it with `play_motion_npz.py`. That step requires GPU and is part of the S2 training campaign itself. Confirm the reference is a single feasible flip before proceeding to training — the double-cartwheel bug remains a live risk.

2. **Exact termination-threshold flag names — PARTIALLY RESOLVED.** The `record_policy.py` script (read from this checkout) uses `--termination-threshold` and `--disable-terminations` exactly as specified. However, the termination keys it overrides are hardcoded as `anchor_pos` and `ee_body_pos` (not CLI-configurable): `env_cfg.terminations["anchor_pos"].params["threshold"]`. The *train* command flag `--env.terminations.anchor-pos.threshold` needs verification against the tracking task's `params/env.yaml` from the cartwheel run — **⚠ still unresolved:** inspect `logs/rsl_rl/g1_tracking/2026-04-19_20-54-47_cartwheel-iterC-single/params/env.yaml` for the exact CLI path for the termination threshold overrides.

3. **Whether 0.5 m is the right threshold for a backflip — deferred.** Cannot be resolved without iterA video review. Use 0.5 m to start (matching the cartwheel precedent); adjust for iterB based on what the video shows. Document the reason if changed.

4. **`record_policy.py` flag names — RESOLVED WITH IMPORTANT CORRECTIONS.** The script in this local checkout (`scripts/record_policy.py`) was read directly. Key findings:

   - **Exact flag names confirmed:** `--disable-terminations` (int, 0 or 1), `--termination-threshold` (float), `--dump-telemetry` (path to `.npz`). These match the spec's usage.
   - **The script is tracking-only.** It is hardcoded to `Mjlab-Tracking-Flat-Unitree-G1` via `load_env_cfg(args.task_id)` — a `--task-id` flag exists but defaults to the tracking task. **It cannot be used as written for the velocity task (S1, S4).** The S1 and S4 record commands in their specs need a different recording approach.
   - **`--no-shadows`, `--no-reflections`, `--no-debug-viz` do NOT exist.** Shadows are disabled by the script unconditionally in code (`env_cfg.viewer.enable_shadows = False`). There is no CLI flag for this. The spec commands using these flags are wrong and must be removed.
   - **`--motion-file` is required.** The script requires `--motion-file` (the tracking motion NPZ) as a positional/required argument.
   - **`--output-dir` (not `--output`).** The output flag is `--output-dir`, not `--output`.
   - **`--checkpoint-file` (not `--checkpoint`).** The checkpoint flag is `--checkpoint-file`.
   - **The script is NOT copied to the Spark.** `/workspace/scripts/` on the Spark contains only `plot_training_curves.py` and `watch_learning.py`. Before any S2 recording can happen, `record_policy.py` must be copied to the Spark: `scp scripts/record_policy.py spark:/tmp/ && ssh spark "docker cp /tmp/record_policy.py mjlab-dev:/workspace/scripts/record_policy.py"`.

5. **`csv_to_npz` ingestion format — RESOLVED.** The probe confirmed: `csv_to_npz` takes `--input-file` (a CSV, **not a `.pkl`**) and `--output-name`. The full pipeline requires the intermediate `pkl_to_csv.py` step — you cannot skip it and feed a `.pkl` directly to `csv_to_npz`. This is already reflected in the Concrete runs section, which is correct as written.
