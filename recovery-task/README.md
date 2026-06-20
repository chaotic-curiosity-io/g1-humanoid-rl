# `Mjlab-Recovery-Flat-Unitree-G1` — the get-up task source

This directory is the **new mjlab task** written for the get-up / fall-recovery
spine policy. It is the source behind [`docs/reports/getting-up.md`](../docs/reports/getting-up.md)
and the spec [`2026-06-19-spine-getup-recovery.md`](../docs/superpowers/specs/2026-06-19-spine-getup-recovery.md).

Unlike every other task in this repo (which are config overrides on stock mjlab
tasks), get-up needed **new code** — so it is committed here for reproducibility.
mjlab itself lives only on the Spark (it is `.gitignore`d), so these files are a
mirror; they are *deployed* into the container, not run from this checkout.

## What it is

A from-scratch task: the G1 starts each episode **lying on the ground** (four
genuinely-fallen reset poses), with no velocity command and no reference motion,
and is rewarded for getting up and holding an upright stand. Built by cloning the
flat velocity config (`unitree_g1_flat_env_cfg`) and changing four things:

- **`mdp/events.py`** — `reset_root_state_lying_down`: the novel piece. Drops the
  robot into supine / face-down / left-side / right-side at a random heading,
  pelvis ~0.1 m off the floor, zero velocity.
- **`mdp/rewards.py`** — `stand_up_height_reward`: a *monotonic* height ramp
  (gradient from the floor to the standing target), so rising is rewarded at every
  height (a narrow Gaussian gave no gradient on the ground).
- **`mdp/terminations.py`** — `stood_up`: a success criterion (used for offline
  eval; deliberately *not* wired as a termination — see the report for why ending
  the episode on success backfired).
- **`config/g1/env_cfgs.py`** — assembles the recovery config: zeroes the twist
  command, swaps in the lying-down reset, drops the `fell_over` termination and the
  foot-locomotion rewards, and sets the final (Attempt 4) reward weights
  (`stand_up` 10.0, `upright` 2.0, `action_rate_l2` -0.03).
- **`config/g1/__init__.py`** — `register_mjlab_task("Mjlab-Recovery-Flat-Unitree-G1", ...)`;
  auto-discovered by `import mjlab.tasks`.

See the report for the full four-iteration reward-shaping journey that arrived at
these weights.

## Deploy (into the Spark's `mjlab-dev` container)

The mjlab tree is root-owned on the Spark, so copy via `/tmp` + `docker cp` (never
`sudo chown`):

```bash
# from this directory, stage into the container's task tree:
ssh spark "docker exec mjlab-dev bash -lc 'mkdir -p /workspace/mjlab/src/mjlab/tasks/recovery/{config/g1,mdp}'"
tar cf - --exclude=README.md . | ssh spark "cat > /tmp/recovery.tar"
ssh spark "docker exec -i mjlab-dev bash -lc 'cd /workspace/mjlab/src/mjlab/tasks/recovery && tar xf -' < /tmp/recovery.tar"
# verify it registers + loads:
ssh spark "docker exec mjlab-dev bash -lc 'python -c \"from mjlab.tasks.registry import load_env_cfg; load_env_cfg(\\\"Mjlab-Recovery-Flat-Unitree-G1\\\"); print(\\\"ok\\\")\"'"
```

mjlab is pip-editable, so the new files are picked up immediately. Train with:

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Recovery-Flat-Unitree-G1 --env.scene.num-envs 2048 --agent.max-iterations 2500 --agent.run-name recovery'"
```
