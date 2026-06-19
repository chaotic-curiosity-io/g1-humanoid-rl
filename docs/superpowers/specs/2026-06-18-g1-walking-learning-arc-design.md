# Design: "Teaching a G1 to walk" — a hands-on learning arc

**Date:** 2026-06-18
**Branch:** `g1-walking-learning-arc`
**Status:** Approved (verbal), pending implementation plan

## Goal

Produce a set of runs + written reports that take a reader **with zero robotics
background** from "what is this even?" to a working mental model of how a Unitree
G1 humanoid learns to walk with reinforcement learning — and what they could
change to explore further. The reader should be able to explain, in their own
words: what the robot is doing, that it *learned* rather than was programmed, and
name 2–3 things they could tweak.

This is the deep-on-one-task path (`Mjlab-Velocity-Flat-Unitree-G1`), chosen over
breadth or the motion-imitation spotlight because it delivers the richest
*understanding per GPU-hour* at bounded reboot risk, and reuses assets already on
disk (42 baseline checkpoints).

## Audience & voice

Every report is written for someone who has never touched robotics or ML. No
undefined jargon: the first time a term appears (policy, reward, episode, PPO,
environment), it is explained in plain language. Concrete analogies over
notation. Each report is standalone and re-runnable by the reader.

## The four-stage arc

| Stage | What runs on the Spark | Cost | Output |
|---|---|---|---|
| Replay | `play` the existing `model_2050.pt` walker | seconds | short clip of a finished walker |
| Progression | `record_learning_progression.py` over the **42 existing checkpoints** (model_0 → model_2050) | minutes, **no training** | one MP4 showing flailing → walking |
| Benchmark run (control) | fresh `Velocity-Flat` train from scratch, ~1500 iters @ 2048 envs | ~60–75 min | live reward curve; reproducibility check |
| Tweak run (B) | identical config, **commanded velocity range narrowed/slowed** | ~60–75 min | second policy with a visibly different gait |

The two fresh runs execute **sequentially** (unified memory → one training job at
a time) inside a single host-quiesce window.

## The runs — concrete

All invoked from the Windows box as `ssh spark 'docker exec mjlab-dev bash -lc "..."'`.

1. **Replay** — `python -m mjlab.scripts.play Mjlab-Velocity-Flat-Unitree-G1`
   loading `model_2050.pt`, recorded clean (`--no-shadows` etc. per CLAUDE.md
   rendering notes). Hold the commanded velocity fixed for a steady shot.

2. **Progression** — `python -m mjlab.scripts.record_learning_progression.py`
   pointed at `logs/rsl_rl/g1_velocity/2026-04-17_18-46-23/`, with
   `--no-shadows --no-reflections --no-debug-viz` for clean frames. Labels the
   iteration number on each segment. **Uses existing checkpoints — no training.**

3. **Benchmark / control run** — fresh `python -m mjlab.scripts.train
   Mjlab-Velocity-Flat-Unitree-G1` from scratch. Config matched to the recorded
   baseline (`params/env.yaml` + `params/agent.yaml` of the 2026-04-17 run) for
   `num_envs` (target 2048), reward terms, and network. Target ~1500 iterations.
   Logs to W&B (entity `donb-chaotic-curiosity`). This run is also the "A" in the
   A/B.

4. **Tweak / "B" run** — same command and config **except** the
   `UniformVelocityCommand` linear-velocity range is narrowed/slowed (timid gait).
   All else identical (PPO, rewards, network, seed-randomization policy).

## The tweak experiment

Headline knob: **commanded velocity range** — the manual's own canonical demo
("identical PPO, identical rewards, identical network; the only difference is the
commanded velocity range"). Control = default range; B = narrower/slower. The
report shows the *same recipe* under a *different instruction distribution*
yielding a visibly different gait, then closes with a **menu of other knobs** the
reader can explore: reward weights, terrain (`Rough`/`Stairs`), episode length,
network size, number of envs.

## Reports — `docs/reports/`, markdown, committed

Four standalone files. Each: plain-language explanation → the exact command used
→ results (embedded reward-curve **PNG plots** + representative video frames) →
a "tweak this to explore" section.

- `00-primer.md` — zero-assumption vocabulary: simulator, policy, reward vs.
  reinforcement, environment/episode, why thousands of robots run in parallel,
  what PPO is at a hand-wave level. The keystone the others build on.
- `01-watching-it-learn.md` — the progression from existing checkpoints; how to
  read a reward curve; the core metrics (reward, episode length, terminations).
- `02-reproducing-the-benchmark.md` — the fresh control run; live curve; did it
  track the Session-1 baseline at matched iterations; what each reward term and
  termination actually means.
- `03-turning-the-knobs.md` — the velocity-range A/B side by side; cause→effect;
  the explore-further menu; pointers to the out-of-scope follow-ups.

## Artifacts & retrieval

- **Plots** (reward curve, metrics) rendered as PNG from each run's logged
  scalars and **committed** into the reports (render inline on GitHub). Need to
  confirm the metric export path (W&B run dir vs. local summaries / tensorboard).
- **Frame stills** (small PNGs, normal filenames so not caught by `.gitignore`'s
  `Pasted image*`/`Screenshot*` rules) committed for inline display.
- **Full MP4s** are gitignored and large → stay on the Spark; pulled to the
  Windows box over Tailscale (`scp`/`rsync` via the `spark` ssh alias) for
  viewing. Each report links the local path.

## Ops & safety

This host hard-reboots under memory pressure (unified memory; swap-death-spiral),
so the fresh runs are bracketed by the CLAUDE.md procedure:

- **Before:** `ssh spark` → `docker stop open-webui compose-arangodb-1
  ollama-compose`; `sudo systemctl stop comfyui.service`; `sudo swapoff -a`.
  Check `free -h` headroom; keep `num_envs` at 2048.
- **During:** training launched as a backgrounded child; stop cleanly with
  SIGINT to the python child (`kill -INT $(pgrep -f mjlab.scripts.train)`), not
  the wrapping shell. Report the reward number as it climbs.
- **After (always, even on failure):** `sudo swapon -a`; `docker start
  open-webui compose-arangodb-1 ollama-compose`; `sudo systemctl start
  comfyui.service`.

`mjlab-dev` is already started and verified (torch sees the GB10).

## Success criteria

1. A no-background reader can explain what the robot does, that it learned rather
   than was programmed, and name 2–3 tweakable knobs.
2. The fresh control run's reward curve tracks the Session-1 baseline at matched
   iterations — reproducibility **shown** (overlaid plot), not asserted.
3. The tweak run produces a *visibly* different gait, attributable to the one
   changed knob, documented side by side.

## Out of scope (good next sessions)

- Motion imitation / cartwheel (`Mjlab-Tracking-Flat-Unitree-G1`, ~11h training).
- Breadth across `Sprint` / `Stairs` / `Rough` / Go1 quadruped.
- Both noted as follow-ups in `03-turning-the-knobs.md`.

## Open questions to resolve during planning

1. Exact mjlab `train` CLI/override syntax for the commanded velocity range
   (dataclass override? config flag?). Verify in the container before the run.
2. How training scalars are exported for plotting (W&B API pull vs. local
   tensorboard/summary files). Pick the simplest reliable source.
3. The baseline run's exact `num_envs` and reward config to match (read its
   `params/*.yaml`).
4. Confirm `record_learning_progression.py` runs against the 2026-04-17 dir
   without a rebuild and where it writes the MP4.
