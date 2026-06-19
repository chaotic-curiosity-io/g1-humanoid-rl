# G1 Walking Learning-Arc Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce four beginner-friendly reports that walk a no-robotics-background reader through how a Unitree G1 learns to walk, backed by real artifacts: a learning-progression video, a reproduced training benchmark, and a one-knob A/B experiment.

**Architecture:** All compute runs on the DGX Spark (`ssh spark` → `docker exec mjlab-dev …`) over Tailscale; this Windows repo is the control + authoring surface. Existing assets (42 baseline checkpoints, their tensorboard log) are reused for the "watch it learn" half; two fresh ~1500-iter `Velocity-Flat` runs (a control + a velocity-range tweak) supply the "reproduce + experiment" half. A small committed helper turns tensorboard scalars into PNG plots. Videos stay on the Spark (gitignored, pulled for viewing); plots + stills + reports are committed.

**Tech Stack:** mjlab (MuJoCo-Warp, RSL-RL, tyro CLI), Python in the `mjlab-dev` NGC PyTorch container, tensorboard `EventAccumulator` + matplotlib for plots, ffmpeg for stills, ssh/scp over Tailscale.

## Global Constraints

- Run everything via `ssh spark 'docker exec mjlab-dev bash -lc "cd /workspace/mjlab && …"'`. Use `python -m mjlab.scripts.<name>` (NOT `uv run`).
- Training env count: **2048** (`--env.scene.num-envs 2048`); never exceed without checking `free -h` first — unified memory → OOM = hard reboot, not clean CUDA error.
- Fresh training runs MUST be bracketed by host quiesce/restore (see Task 4/Task 5). Restore co-tenants + `swapon -a` even on failure.
- Stop training with SIGINT to the **python child**: `kill -INT $(pgrep -f mjlab.scripts.train)` — not the wrapping shell.
- Baseline run dir (reuse, never overwrite): `logs/rsl_rl/g1_velocity/2026-04-17_18-46-23/` (paths relative to `/workspace/mjlab`).
- Baseline reference curve (for reproducibility claims): iter 500→3.30, 900→21.30, 1400→31.47, 2050→50.51 reward; ep length 173→962→925→995.
- Both experiments share `experiment_name=g1_velocity`; disambiguate via `--agent.run-name`. Tag fresh runs `arc-control` and `arc-tweak-slow`.
- Tweak = single knob (forward-speed range `lin_vel_x`). **The Flat task runs a `command_vel` curriculum whose stage-0 fires at step 0** (confirmed in Task 1), so a plain range override is clobbered back to `(-1,1)` on the first reset. The tweak MUST override the command range AND all three curriculum stages, using tyro's `=` tuple syntax (space-separated values are rejected):
  `"--env.commands.twist.ranges.lin-vel-x=(-0.5, 0.5)"` + `"--env.curriculum.command-vel.params.velocity-stages.{0,1,2}.lin-vel-x=(-0.5, 0.5)"`.
  **Control needs NO range flags** — its curriculum stays at stage 0 = `(-1.0, 1.0)` for all iters <5000, so a 1500-iter control run reproduces the baseline faithfully. Both keep seed 42; only `lin_vel_x` differs.
- Headline metric tags: `Train/mean_reward`, `Train/mean_episode_length`. Reward-term tags are `Episode_Reward/*`; terminations `Episode_Termination/{fell_over,time_out}`.
- Commit cadence: one commit per task on branch `g1-walking-learning-arc`. Commit messages end with the two trailers used in the repo (Co-Authored-By + Claude-Session).
- Reports written for a reader with **zero** robotics/ML background: define every term on first use; commands shown so they can re-run; each ends with a "tweak this to explore" section.
- Generated artifacts on Spark live under `~/robotic-simulation/out/arc/`. Committed assets live under `docs/reports/assets/` (PNG/CSV commit; MP4 is gitignored).

## File Structure

- `scripts/plot_training_curves.py` — NEW. tensorboard-scalars → CSV + PNG curves; overlays N runs. One responsibility: turn logged scalars into plottable artifacts.
- `docs/reports/00-primer.md` — NEW. Zero-assumption concepts.
- `docs/reports/01-watching-it-learn.md` — NEW. Progression + reading a reward curve.
- `docs/reports/02-reproducing-the-benchmark.md` — NEW. Fresh control run vs baseline.
- `docs/reports/03-turning-the-knobs.md` — NEW. The velocity-range A/B + explore menu.
- `docs/reports/assets/` — NEW dir. PNG plots + frame stills (committed); MP4s (pulled, gitignored).
- `setup-notes.md` — MODIFY. Append a Session-2 block (canonical "what we did" record).
- `README.md` — MODIFY. Add a pointer to `docs/reports/`.

---

### Task 1: De-risk runs on the Spark (smoke tests, no host changes)

Proves the exact commands work and the tweak override actually sticks, before spending the quiesce window. A 64-env / 3-iter smoke is tiny and safe to run alongside the co-tenants.

**Files:**
- Modify: `setup-notes.md` (append a short "Session 2 — pre-run recon" note)

**Interfaces:**
- Produces: confirmed-working invocations consumed by Tasks 3–5; the effective-range proof for the tweak.

- [ ] **Step 1: Confirm the progression script loads its CLI**

Run:
```bash
ssh spark 'docker exec mjlab-dev bash -lc "cd /workspace/mjlab && python -m mjlab.scripts.record_learning_progression --help 2>&1 | head -30"'
```
Expected: tyro help listing `--run-dir`, `--num-checkpoints`, `--cameras`, `--command-lin-vel-x` (no traceback).

- [ ] **Step 2: Smoke-train the CONTROL recipe (3 iters, 64 envs)**

Run:
```bash
ssh spark 'docker exec mjlab-dev bash -lc "cd /workspace/mjlab && MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python -u -m mjlab.scripts.train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 64 --agent.max-iterations 3 --agent.run-name arc-smoke-control 2>&1 | tail -25"'
```
Expected: prints `[INFO] Training with: device=cuda:0, seed=42`, runs 3 iterations, exits 0. A new dir `logs/rsl_rl/g1_velocity/<ts>/` with `model_0.pt` appears.

- [ ] **Step 3: Smoke-train the TWEAK override and PROVE the range stuck**

Run:
```bash
ssh spark 'docker exec mjlab-dev bash -lc "cd /workspace/mjlab && MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python -u -m mjlab.scripts.train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 64 --agent.max-iterations 3 --agent.run-name arc-smoke-tweak --env.commands.twist.ranges.lin-vel-x -0.5 0.5 2>&1 | grep -iE \"lin_vel_x|error|traceback\" | head"'
```
Then verify the effective range in the just-written run's `params/env.yaml`:
```bash
ssh spark 'docker exec mjlab-dev bash -lc "cd /workspace/mjlab && d=\$(ls -dt logs/rsl_rl/g1_velocity/*/ | head -1); echo \$d; grep -A3 \"lin_vel_x\" \$d/params/env.yaml | head"'
```
Expected: the dumped `commands.twist.ranges.lin_vel_x` is `(-0.5, 0.5)`, confirming the override is honored and not clobbered by a curriculum. **If it shows `(-1.0, 1.0)`, STOP** and resolve (curriculum clobber) before the real runs.

- [ ] **Step 4: Clean up smoke run dirs**

Run:
```bash
ssh spark 'docker exec mjlab-dev bash -lc "cd /workspace/mjlab && for d in \$(ls -dt logs/rsl_rl/g1_velocity/*/); do grep -q \"arc-smoke\" \$d/params/agent.yaml 2>/dev/null && echo rm \$d && rm -rf \$d; done"'
```
Expected: smoke dirs removed; the `2026-04-17_18-46-23` baseline untouched (verify it still lists 42 `.pt`).

- [ ] **Step 5: Record findings + commit**

Append to `setup-notes.md` a "Session 2 — pre-run recon" block stating: override path `--env.commands.twist.ranges.lin-vel-x` works and sticks (curriculum off for Flat); rate ~1.25 s/iter; container has matplotlib/pandas; 39 tensorboard scalar tags incl. `Train/mean_reward`.
```bash
git add setup-notes.md && git commit -m "$(printf 'Session 2 recon: confirm velocity-range override + tooling on Spark\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_012rLA3NvtjLP18n2bDJKxa1')"
```

---

### Task 2: Plot helper — tensorboard scalars → CSV + PNG curves

One reusable, committed tool. Runs inside the container (where tensorboard + matplotlib live). Supports overlaying multiple runs for the reproducibility and A/B plots.

**Files:**
- Create: `scripts/plot_training_curves.py`
- Verify against: `logs/rsl_rl/g1_velocity/2026-04-17_18-46-23/` (baseline tfevents)

**Interfaces:**
- Produces: CLI `python scripts/plot_training_curves.py --run [name=]<dir> [--run …] --tags <t…> --out <dir>`; writes `<out>/<name>_scalars.csv` and `<out>/<tag_with_slashes_as_underscores>.png`. Consumed by Tasks 6.

- [ ] **Step 1: Write the helper**

Create `scripts/plot_training_curves.py`:
```python
#!/usr/bin/env python3
"""Extract RSL-RL tensorboard scalars to CSV and render PNG curve plots.

Runs inside the mjlab-dev container (needs tensorboard + matplotlib, both present).
Pass --run multiple times (optionally name=dir) to overlay runs on each tag.

Examples
--------
python scripts/plot_training_curves.py --run logs/rsl_rl/g1_velocity/<ctrl> --out out/arc/ctrl
python scripts/plot_training_curves.py \
    --run baseline=logs/rsl_rl/g1_velocity/2026-04-17_18-46-23 \
    --run control=logs/rsl_rl/g1_velocity/<ctrl> \
    --tags Train/mean_reward Train/mean_episode_length --out out/arc/repro
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_event_file(run_dir: Path) -> Path:
    hits = sorted(glob.glob(str(run_dir / "events.out.tfevents*")))
    if not hits:
        raise FileNotFoundError(f"no tfevents in {run_dir}")
    return Path(hits[-1])


def load_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    ea = EventAccumulator(str(find_event_file(run_dir)), size_guidance={"scalars": 0})
    ea.Reload()
    return {
        tag: [(s.step, s.value) for s in ea.Scalars(tag)]
        for tag in ea.Tags().get("scalars", [])
    }


def write_csv(scalars: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "step", "value"])
        for tag, series in scalars.items():
            for step, val in series:
                w.writerow([tag, step, val])


def plot_tag(runs: dict[str, dict], tag: str, out_png: Path) -> bool:
    plt.figure(figsize=(8, 5))
    plotted = False
    for name, scalars in runs.items():
        series = scalars.get(tag)
        if not series:
            continue
        xs = [s for s, _ in series]
        ys = [v for _, v in series]
        plt.plot(xs, ys, label=name, linewidth=2)
        plotted = True
    if not plotted:
        plt.close()
        return False
    plt.xlabel("training iteration")
    plt.ylabel(tag)
    plt.title(tag)
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()
    return True


def parse_run(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, p = spec.split("=", 1)
        return name, Path(p)
    return Path(spec).name, Path(spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="run dir, optionally name=dir; repeatable to overlay")
    ap.add_argument("--tags", nargs="*",
                    default=["Train/mean_reward", "Train/mean_episode_length"])
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    runs = {name: load_scalars(p) for name, p in map(parse_run, args.run)}
    out = Path(args.out)
    for name, scalars in runs.items():
        write_csv(scalars, out / f"{name}_scalars.csv")
    made = [str(out / f"{t.replace('/', '_')}.png")
            for t in args.tags if plot_tag(runs, t, out / f"{t.replace('/', '_')}.png")]
    print("WROTE:\n  " + "\n  ".join(made))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Get the helper onto the Spark**

Run:
```bash
scp scripts/plot_training_curves.py spark:/home/chaotic-curiosity/robotic-simulation/scripts/plot_training_curves.py
```
Expected: copied (1 file).

- [ ] **Step 3: Verify it parses the baseline and the reward rises**

Run:
```bash
ssh spark 'docker exec mjlab-dev bash -lc "cd /workspace && python scripts/plot_training_curves.py --run baseline=mjlab/logs/rsl_rl/g1_velocity/2026-04-17_18-46-23 --out out/arc/_selftest && echo --- && head -3 out/arc/_selftest/baseline_scalars.csv && python - <<PY
import csv
rows=[r for r in csv.DictReader(open(\"out/arc/_selftest/baseline_scalars.csv\")) if r[\"tag\"]==\"Train/mean_reward\"]
vals=[float(r[\"value\"]) for r in rows]
print(\"reward points:\", len(vals), \"first:\", round(vals[0],2), \"last:\", round(vals[-1],2))
assert len(vals) > 10 and vals[-1] > vals[0] + 10, \"reward curve not rising as expected\"
print(\"OK\")
PY"'
```
Expected: CSV header `tag,step,value`; prints `reward points: ... first: ~0 last: ~50`; `OK`; and `out/arc/_selftest/Train_mean_reward.png` exists.

- [ ] **Step 4: Commit the helper**

```bash
git add scripts/plot_training_curves.py && git commit -m "$(printf 'Add plot_training_curves.py: tensorboard scalars -> CSV + curve PNGs\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_012rLA3NvtjLP18n2bDJKxa1')"
```

---

### Task 3: Learning-progression video from the 42 existing checkpoints (no training)

The "watch it learn" artifact, built from assets already on disk. Defaults are already clean (shadows/reflections/debug-viz off).

**Files:**
- Create: `docs/reports/assets/progression_final_still.png` (+ optional early/mid stills)

**Interfaces:**
- Produces: progression MP4(s) on Spark at `out/arc/progression/`; committed stills consumed by report 01.

- [ ] **Step 1: Render the progression (24 evenly-spaced checkpoints, fixed forward command)**

Run (this iterates checkpoints; allow a few minutes):
```bash
ssh spark 'docker exec mjlab-dev bash -lc "cd /workspace/mjlab && MUJOCO_GL=egl python -m mjlab.scripts.record_learning_progression --run-dir logs/rsl_rl/g1_velocity/2026-04-17_18-46-23 --output-dir /workspace/out/arc/progression --num-checkpoints 24 --cameras chase,side --command-lin-vel-x 1.0 --command-ang-vel-z 0.0 2>&1 | tail -20"'
```
Expected: progress over checkpoints; MP4s written (one per camera + a 2×2 grid). Verify:
```bash
ssh spark 'ls -la /home/chaotic-curiosity/robotic-simulation/out/arc/progression/'
```
Expected: `*.mp4` files present, non-zero size.

- [ ] **Step 2: Pull the videos to the Windows box for viewing**

Run:
```bash
mkdir -p docs/reports/assets && scp 'spark:/home/chaotic-curiosity/robotic-simulation/out/arc/progression/*.mp4' docs/reports/assets/
```
Expected: MP4s land in `docs/reports/assets/` (gitignored — not committed).

- [ ] **Step 3: Extract representative stills (random-start, mid, trained)**

Run (on Spark, where ffmpeg + the chase video live; pull the PNGs back):
```bash
ssh spark 'docker exec mjlab-dev bash -lc "cd /workspace/out/arc/progression && f=\$(ls *chase*.mp4 | head -1); ffmpeg -y -i \$f -vf select=eq(n\,5) -vframes 1 still_early.png; ffmpeg -y -i \$f -vf thumbnail -frames:v 1 still_mid.png; ffmpeg -y -sseof -1 -i \$f -vframes 1 still_final.png 2>&1 | tail -2"'
scp 'spark:/home/chaotic-curiosity/robotic-simulation/out/arc/progression/still_*.png' docs/reports/assets/
```
Expected: three PNG stills in `docs/reports/assets/`. (If `ffmpeg` is absent in the container, install once: `pip install imageio-ffmpeg` is unnecessary — use `apt-get install -y ffmpeg` inside the container, or fall back to mjlab's own frame dump.)

- [ ] **Step 4: Commit the stills**

```bash
git add docs/reports/assets/still_early.png docs/reports/assets/still_mid.png docs/reports/assets/still_final.png
git commit -m "$(printf 'Add learning-progression stills (random -> mid -> trained walker)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_012rLA3NvtjLP18n2bDJKxa1')"
```

---

### Task 4: Quiesce host + run the CONTROL training (~1500 iters)

First fresh run. Reproduces the Session-1 recipe exactly. ~31 min at 2048 envs.

**Files:** none committed here (produces a run dir on the Spark; metrics captured in Task 6).

**Interfaces:**
- Produces: a completed run dir `logs/rsl_rl/g1_velocity/<ctrl_ts>/` (record the exact path for Tasks 5/6/7) with 30 checkpoints + tfevents.

- [ ] **Step 1: Quiesce co-tenants and check headroom**

Run:
```bash
ssh spark 'docker stop open-webui compose-arangodb-1 ollama-compose; sudo systemctl stop comfyui.service; sudo swapoff -a; echo "--- free ---"; free -h'
```
Expected: three containers stop; swap row shows `0B` total; ample free memory (>100 GiB).

- [ ] **Step 2: Launch the control run detached, logging to a file**

Run:
```bash
ssh spark 'docker exec -d mjlab-dev bash -lc "cd /workspace/mjlab && MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python -u -m mjlab.scripts.train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 2048 --agent.max-iterations 1500 --agent.run-name arc-control > /workspace/logs/arc-control.log 2>&1"'
```
Then confirm it started:
```bash
ssh spark 'docker exec mjlab-dev bash -lc "sleep 20; pgrep -f mjlab.scripts.train && tail -5 /workspace/logs/arc-control.log"'
```
Expected: a PID prints; log shows `[INFO] Training with: device=cuda:0, seed=42` and early iteration lines.

- [ ] **Step 3: Poll to completion (run as a background monitor, not a blocking sleep)**

Poll the log every ~60s until iteration 1500 or the process exits. Each poll:
```bash
ssh spark 'docker exec mjlab-dev bash -lc "pgrep -f mjlab.scripts.train >/dev/null && echo RUNNING || echo DONE; grep -E \"Mean reward|^ *Mean reward|iteration\" /workspace/logs/arc-control.log | tail -3"'
```
Report the climbing reward to the user. Expected progression near baseline: ~3 @500, ~21 @900, ~31 @1400. Expected end: `DONE`, ~30 checkpoints written.

- [ ] **Step 4: Capture the control run dir path + verify checkpoints**

Run:
```bash
ssh spark 'docker exec mjlab-dev bash -lc "cd /workspace/mjlab && d=\$(grep -l arc-control logs/rsl_rl/g1_velocity/*/params/agent.yaml | xargs -n1 dirname); echo CTRL_DIR=\$d; ls \$d/model_*.pt | wc -l; ls \$d/model_*.pt | sort -V | tail -1"'
```
Expected: prints `CTRL_DIR=logs/rsl_rl/g1_velocity/<ts>`; ~30 checkpoints; final `model_1500.pt`. **Record `CTRL_DIR`.**

- [ ] **Step 5: Checkpoint commit (notes only)**

Append the control run path + final reward to a scratch line in `setup-notes.md` (Session 2 block) and commit:
```bash
git add setup-notes.md && git commit -m "$(printf 'Session 2: control run arc-control complete (record run dir + final reward)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_012rLA3NvtjLP18n2bDJKxa1')"
```

---

### Task 5: Run the TWEAK training (slow gait) + restore host

Identical recipe, one knob changed. Then restore the host regardless of outcome.

**Files:** none committed here (produces a run dir; metrics in Task 6).

**Interfaces:**
- Consumes: quiesced host from Task 4 (still quiesced — do NOT restore between runs).
- Produces: `logs/rsl_rl/g1_velocity/<tweak_ts>/` (record `TWEAK_DIR`).

- [ ] **Step 1: Write the tweak launch script (avoids nested-quote breakage) + run it detached**

The 4-flag override has parentheses and spaces; pushing it through ssh→docker→bash→tyro inline is fragile. Write it to a script on the bind mount (`~/robotic-simulation` = container `/workspace`) and run that:
```bash
ssh spark 'cat > /home/chaotic-curiosity/robotic-simulation/arc-tweak.sh' <<'SH'
cd /workspace/mjlab
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python -u -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Unitree-G1 \
  --env.scene.num-envs 2048 --agent.max-iterations 1500 --agent.run-name arc-tweak-slow \
  "--env.commands.twist.ranges.lin-vel-x=(-0.5, 0.5)" \
  "--env.curriculum.command-vel.params.velocity-stages.0.lin-vel-x=(-0.5, 0.5)" \
  "--env.curriculum.command-vel.params.velocity-stages.1.lin-vel-x=(-0.5, 0.5)" \
  "--env.curriculum.command-vel.params.velocity-stages.2.lin-vel-x=(-0.5, 0.5)"
SH
ssh spark 'docker exec -d mjlab-dev bash -lc "bash /workspace/arc-tweak.sh > /workspace/logs/arc-tweak.log 2>&1"'
ssh spark 'docker exec mjlab-dev bash -lc "sleep 20; pgrep -f mjlab.scripts.train && tail -5 /workspace/logs/arc-tweak.log"'
```
Expected: started; log shows training begun.

- [ ] **Step 2: Poll to completion** (same pattern as Task 4 Step 3, log `arc-tweak.log`). Report reward to user.

- [ ] **Step 3: Capture the tweak run dir + verify**

Run:
```bash
ssh spark 'docker exec mjlab-dev bash -lc "cd /workspace/mjlab && d=\$(ls -d logs/rsl_rl/g1_velocity/*_arc-tweak-slow | tail -1); echo TWEAK_DIR=\$d; echo \"lin_vel_x = -0.5 count (want 4):\"; grep -A1 lin_vel_x \$d/params/env.yaml | grep -c -- \"- -0.5\"; ls \$d/model_*.pt | wc -l"'
```
Expected: `TWEAK_DIR=…`; the `-0.5` count is **4** (command range + 3 curriculum stages all pinned — confirms the knob held); ~30 checkpoints.

- [ ] **Step 4: Restore the host (ALWAYS — even if a run failed)**

Run:
```bash
ssh spark 'sudo swapon -a; sudo systemctl start comfyui.service; docker start ollama-compose open-webui compose-arangodb-1; echo "--- restored ---"; free -h | head -2; docker ps --format "{{.Names}}\t{{.Status}}"'
```
Expected: swap re-enabled; three co-tenants back `Up`; `mjlab-dev` still up.

- [ ] **Step 5: Commit notes**

Append `TWEAK_DIR` + final reward to `setup-notes.md`; commit:
```bash
git add setup-notes.md && git commit -m "$(printf 'Session 2: tweak run arc-tweak-slow complete; host restored\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_012rLA3NvtjLP18n2bDJKxa1')"
```

---

### Task 6: Generate curve + comparison plots

Turn the three run dirs (baseline, control, tweak) into the PNGs the reports embed.

**Files:**
- Create: `docs/reports/assets/repro_Train_mean_reward.png`, `…/repro_Train_mean_episode_length.png`, `…/ab_Train_mean_reward.png`, `…/control_reward_terms.png`, and the backing `*_scalars.csv`.

**Interfaces:**
- Consumes: `plot_training_curves.py` (Task 2); `CTRL_DIR`, `TWEAK_DIR`.

- [ ] **Step 1: Reproducibility overlay (baseline vs control)**

Run (substitute `<CTRL_DIR>`):
```bash
ssh spark 'docker exec mjlab-dev bash -lc "cd /workspace && python scripts/plot_training_curves.py --run baseline=mjlab/logs/rsl_rl/g1_velocity/2026-04-17_18-46-23 --run control=mjlab/<CTRL_DIR> --tags Train/mean_reward Train/mean_episode_length --out out/arc/repro && ls out/arc/repro"'
```
Expected: `Train_mean_reward.png`, `Train_mean_episode_length.png`, two CSVs.

- [ ] **Step 2: A/B overlay (control vs tweak) on reward + achieved-speed error**

Run (substitute both dirs):
```bash
ssh spark 'docker exec mjlab-dev bash -lc "cd /workspace && python scripts/plot_training_curves.py --run control=mjlab/<CTRL_DIR> --run tweak_slow=mjlab/<TWEAK_DIR> --tags Train/mean_reward Episode_Reward/track_linear_velocity Metrics/twist/error_vel_xy --out out/arc/ab && ls out/arc/ab"'
```
Expected: three PNGs + CSVs.

- [ ] **Step 3: Reward-term breakdown for the control run**

Run:
```bash
ssh spark 'docker exec mjlab-dev bash -lc "cd /workspace && python scripts/plot_training_curves.py --run control=mjlab/<CTRL_DIR> --tags Episode_Reward/track_linear_velocity Episode_Reward/upright Episode_Reward/air_time Episode_Reward/action_rate_l2 --out out/arc/terms && ls out/arc/terms"'
```
Expected: four PNGs.

- [ ] **Step 4: Pull the PNGs + CSVs to committed assets (renamed by purpose)**

Run:
```bash
B=/home/chaotic-curiosity/robotic-simulation/out/arc
scp "spark:$B/repro/Train_mean_reward.png" docs/reports/assets/repro_Train_mean_reward.png
scp "spark:$B/repro/Train_mean_episode_length.png" docs/reports/assets/repro_Train_mean_episode_length.png
scp "spark:$B/ab/Train_mean_reward.png" docs/reports/assets/ab_Train_mean_reward.png
scp "spark:$B/ab/Metrics_twist_error_vel_xy.png" docs/reports/assets/ab_speed_error.png
scp "spark:$B/repro/control_scalars.csv" docs/reports/assets/control_scalars.csv
```
Expected: files land in `docs/reports/assets/`.

- [ ] **Step 5: Commit plots**

```bash
git add docs/reports/assets/*.png docs/reports/assets/control_scalars.csv
git commit -m "$(printf 'Add reward-curve + A/B comparison plots\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_012rLA3NvtjLP18n2bDJKxa1')"
```

---

### Task 7: Record the A/B gait clips (control vs tweak, same fixed command)

Show both final policies asked to walk forward at the SAME speed; the slow-trained one struggles. Reuses the progression recorder with a single checkpoint.

**Files:**
- Create: `docs/reports/assets/ab_control_still.png`, `docs/reports/assets/ab_tweak_still.png`

**Interfaces:**
- Consumes: `CTRL_DIR`, `TWEAK_DIR`.

- [ ] **Step 1: Render each final policy at forward 1.0 m/s**

Run (substitute dirs):
```bash
ssh spark 'docker exec mjlab-dev bash -lc "cd /workspace/mjlab && MUJOCO_GL=egl python -m mjlab.scripts.record_learning_progression --run-dir <CTRL_DIR> --output-dir /workspace/out/arc/ab_control --num-checkpoints 1 --include-first False --cameras chase --command-lin-vel-x 1.0 2>&1 | tail -5"'
ssh spark 'docker exec mjlab-dev bash -lc "cd /workspace/mjlab && MUJOCO_GL=egl python -m mjlab.scripts.record_learning_progression --run-dir <TWEAK_DIR> --output-dir /workspace/out/arc/ab_tweak --num-checkpoints 1 --include-first False --cameras chase --command-lin-vel-x 1.0 2>&1 | tail -5"'
```
Expected: one MP4 in each output dir (the final-checkpoint clip).

- [ ] **Step 2: Extract one still each + pull videos and stills**

Run:
```bash
ssh spark 'docker exec mjlab-dev bash -lc "cd /workspace/out/arc && for k in ab_control ab_tweak; do f=\$(ls \$k/*chase*.mp4|head -1); ffmpeg -y -sseof -1 -i \$f -vframes 1 \$k/still.png 2>&1 | tail -1; done"'
scp 'spark:/home/chaotic-curiosity/robotic-simulation/out/arc/ab_control/*.mp4' docs/reports/assets/
scp 'spark:/home/chaotic-curiosity/robotic-simulation/out/arc/ab_tweak/*.mp4' docs/reports/assets/
scp spark:/home/chaotic-curiosity/robotic-simulation/out/arc/ab_control/still.png docs/reports/assets/ab_control_still.png
scp spark:/home/chaotic-curiosity/robotic-simulation/out/arc/ab_tweak/still.png docs/reports/assets/ab_tweak_still.png
```
Expected: stills + (gitignored) MP4s in assets.

- [ ] **Step 3: Commit stills**

```bash
git add docs/reports/assets/ab_control_still.png docs/reports/assets/ab_tweak_still.png
git commit -m "$(printf 'Add A/B gait stills (control vs slow-trained at 1 m/s forward)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_012rLA3NvtjLP18n2bDJKxa1')"
```

---

### Task 8: Write `docs/reports/00-primer.md`

The keystone. Zero assumptions. ~700–1100 words.

**Files:**
- Create: `docs/reports/00-primer.md`

- [ ] **Step 1: Write the primer.** Must cover, each in plain language with an analogy, no undefined jargon:
  1. What a physics **simulator** is (MuJoCo) and why we train in one (cheap, fast, safe, parallel) — note 2048 robots at once.
  2. What the **G1** is (a humanoid with motors at its joints) and what "controlling" it means (choose joint targets 50×/sec).
  3. **Policy** = the robot's "brain": a function from what it senses → what it does. It's a neural network with numbers ("weights") that start random.
  4. **Reward** = a score we define for good behavior (go the commanded speed, stay upright, don't flail). We never tell it *how* to walk.
  5. **Reinforcement learning / PPO** at a hand-wave: try, score, nudge the weights toward higher score, repeat billions of times. Walking *emerges*.
  6. **Episode / termination**: a try ends after 20 s (`time_out`) or if it falls (`fell_over`); higher "episode length" = staying upright longer.
  7. A map of the four reports.
  End with "tweak this to explore": pointers to the knobs explored later.

- [ ] **Step 2: Verify + commit.** Read it back; confirm no term is used before it's defined; check it links to reports 01–03.
```bash
git add docs/reports/00-primer.md && git commit -m "$(printf 'Add report 00: zero-background primer on robot RL\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_012rLA3NvtjLP18n2bDJKxa1')"
```

---

### Task 9: Write `docs/reports/01-watching-it-learn.md`

**Files:**
- Create: `docs/reports/01-watching-it-learn.md`
- Embeds: `assets/still_early.png`, `assets/still_mid.png`, `assets/still_final.png`

- [ ] **Step 1: Write it.** Structure: (a) what you're seeing in the progression video — random twitching at `model_0`, the "aha" climb, the confident walker at `model_2050` (embed the three stills, name the iteration each is from); (b) how to read a **reward curve** — embed `repro_Train_mean_reward.png` later or describe the S-curve with the baseline numbers (500→3.3, 900→21.3, 1400→31.5, 2050→50.5); (c) what **episode length** means and why it tracks reward; (d) the exact command used (`record_learning_progression …`) so they can re-make it; (e) "tweak this to explore": `--command-lin-vel-x`, `--num-checkpoints`, `--cameras`. Link back to 00 for any term.

- [ ] **Step 2: Verify the embedded image paths resolve, then commit.**
```bash
git add docs/reports/01-watching-it-learn.md && git commit -m "$(printf 'Add report 01: watching the G1 learn to walk\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_012rLA3NvtjLP18n2bDJKxa1')"
```

---

### Task 10: Write `docs/reports/02-reproducing-the-benchmark.md`

**Files:**
- Create: `docs/reports/02-reproducing-the-benchmark.md`
- Embeds: `assets/repro_Train_mean_reward.png`, `assets/repro_Train_mean_episode_length.png`, `assets/control_reward_terms.png` (or the four `terms` PNGs)

- [ ] **Step 1: Write it.** Structure: (a) what "reproduce a benchmark" means and why it matters (same recipe → similar curve = the result is real, not luck); (b) the exact training command + host-quiesce steps (and *why* quiesce — unified memory); (c) the overlaid baseline-vs-control plot, with the matched-iteration comparison table (control@500/900/1400 vs baseline 3.3/21.3/31.5) and an honest note on any divergence (different RNG, GPU nondeterminism); (d) **what each reward term is** using `Episode_Reward/*` — embed the term breakdown and explain track_linear_velocity (the main job), upright/pose (don't fall), air_time/foot_clearance (gait shaping), action_rate (smoothness); (e) terminations (`fell_over` vs `time_out`); (f) "tweak this to explore": `--agent.max-iterations` (longer → higher plateau), `--agent.seed`, `--env.scene.num-envs`.

- [ ] **Step 2: Verify + commit.**
```bash
git add docs/reports/02-reproducing-the-benchmark.md && git commit -m "$(printf 'Add report 02: reproducing the walking benchmark\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_012rLA3NvtjLP18n2bDJKxa1')"
```

---

### Task 11: Write `docs/reports/03-turning-the-knobs.md`

**Files:**
- Create: `docs/reports/03-turning-the-knobs.md`
- Embeds: `assets/ab_Train_mean_reward.png`, `assets/ab_speed_error.png`, `assets/ab_control_still.png`, `assets/ab_tweak_still.png`

- [ ] **Step 1: Write it.** Structure: (a) the one-knob idea — control trained on `lin_vel_x ∈ [-1,1]`, tweak on `[-0.5,0.5]`, everything else identical (state it as the controlled experiment it is); (b) side-by-side stills of both asked to go 1 m/s forward — the slow-trained policy is out of its training distribution and struggles; explain *why* (it never practiced fast); (c) the A/B reward + speed-error plots; (d) the **explore menu** with concrete commands: reward weights (`--env.rewards.<term>.weight`), terrain (`Mjlab-Velocity-Rough/Stairs-Unitree-G1`), episode length, network size, `Sprint`; (e) the out-of-scope follow-ups: motion imitation (`Mjlab-Tracking-Flat-Unitree-G1`, the cartwheel pipeline in `docs/cartwheel-journey.md`) and Go1 quadruped. Link back to 00–02.

- [ ] **Step 2: Verify + commit.**
```bash
git add docs/reports/03-turning-the-knobs.md && git commit -m "$(printf 'Add report 03: turning the knobs (velocity-range A/B + explore menu)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_012rLA3NvtjLP18n2bDJKxa1')"
```

---

### Task 12: Finalize — index, notes, review

**Files:**
- Modify: `README.md` (add a `docs/reports/` pointer), `setup-notes.md` (Session 2 summary)
- Create: `docs/reports/README.md` (one-paragraph index + reading order)

- [ ] **Step 1: Add a reports index** `docs/reports/README.md` listing the four reports in reading order with a one-line hook each, and a note that MP4s are pulled separately (gitignored).

- [ ] **Step 2: Point the top-level README** at `docs/reports/` (a line under the docs/ map).

- [ ] **Step 3: Write the Session-2 summary** in `setup-notes.md`: control/tweak run dirs, final rewards, wall-clock, any deviations (this is the canonical "what we did" record).

- [ ] **Step 4: Self-review the four reports for a no-background reader** — pick the harshest sentence in each and simplify it; confirm every embedded asset path resolves (`ls docs/reports/assets/`).

- [ ] **Step 5: Commit + report completion.**
```bash
git add README.md setup-notes.md docs/reports/README.md
git commit -m "$(printf 'Index reports, update README + Session-2 notes\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_012rLA3NvtjLP18n2bDJKxa1')"
git log --oneline -14
```
Then summarize to the user: what was built, where the videos are, and offer to merge `g1-walking-learning-arc` → `main`.

---

## Self-Review

**Spec coverage:**
- Replay/"see the goal" → folded into progression (final segment/still, Task 3) — spec allowed this; the "goal" shot is `still_final.png`. ✔
- Progression from existing checkpoints → Task 3. ✔
- Fresh benchmark/control run → Task 4. ✔
- Tweak A/B → Task 5 (run) + Task 7 (clips) + Task 6 (plots). ✔
- Four reports, no-background, with "tweak to explore" + commands → Tasks 8–11. ✔
- Plots from tensorboard, videos pulled, PNGs committed → Task 2/6/7. ✔
- Ops/safety quiesce+restore, SIGINT-to-python, 2048 envs → Tasks 4/5 + Global Constraints. ✔
- Success criteria (reproducibility shown via overlay; visible gait difference) → Task 6 Step 1 + Task 7. ✔
- Out-of-scope follow-ups noted → Task 11. ✔

**Placeholder scan:** Run-dir paths are written as `<CTRL_DIR>`/`<TWEAK_DIR>` because they're timestamp-generated at execution; Task 4/5 Step 4 capture them explicitly and later tasks substitute. This is a runtime value, not a vague placeholder. Report tasks specify exact sections, embedded asset filenames, and teaching points rather than final prose, because the prose is data-driven (depends on the runs' actual numbers) — each lists concrete required content. No "TBD"/"add error handling"/"similar to" found.

**Type consistency:** Helper CLI (`--run name=dir`, `--tags`, `--out`) used identically in Tasks 2/6. Asset filenames referenced in report tasks match those created/renamed in Tasks 3/6/7 (`still_{early,mid,final}.png`, `repro_*`, `ab_*`). Run-name labels (`arc-control`, `arc-tweak-slow`) consistent across Tasks 4–7. Tag names match the 39 confirmed tensorboard tags.
