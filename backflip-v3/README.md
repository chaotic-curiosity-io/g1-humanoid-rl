# backflip-v3: landing reward for the tracking task

Goal: make the backflip policy (`Mjlab-Tracking-Flat-Unitree-G1`, motion
`smpl_backflip_to_g1.npz`) land on its feet instead of its back. The stock
tracking reward only pays for matching the reference pose; nothing rewards
ending the clip upright on both feet. We add a gated landing reward that is
**OFF by default (weight 0.0)** so the shared cartwheel / any other tracking use
is completely unaffected. Enable for the backflip run via a single CLI override.

## New reward term

- Name: `landing_feet_upright`
- Function: `mjlab.tasks.tracking.mdp.landing_feet_upright`
- Default weight: **0.0** (verified via load-check; cartwheel-safe).
- CLI override to enable for the backflip:
  `--env.rewards.landing_feet_upright.weight 5.0`
- Recommended starting weight: **5.0** (range to sweep: 2.0 - 10.0). The term is
  bounded in [0, 1], so weight is directly comparable to the per-step pose terms
  (`motion_body_pos`/`motion_body_ori` have weight 1.0 each; `motion_global_*`
  0.5 each). 5.0 makes a clean feet-down upright landing worth roughly the same
  as nailing the whole-body pose match in the landing window, which is the
  intended trade. Start at 5.0; raise toward 8-10 if it still backflops, lower
  toward 2-3 if it sacrifices the flip to bail early into a stand.

## Formula

`reward = gate * upright * feet_down`  (all three in [0, 1], product in [0, 1])

- **gate** (phase gate, the requested option (a)): clamped linear ramp on the
  reference motion phase. `phase = command.time_steps / (time_step_total - 1)`.
  `gate = clamp((phase - phase_gate_start) / (1 - phase_gate_start), 0, 1)`.
  Default `phase_gate_start = 0.6` -> active over the last 40% of the clip
  (the landing phase), ramping from 0 at phase 0.6 to 1 at the final frame.
  Using a ramp (not a hard step) gives the policy gradient as it approaches the
  landing window. Phase is per-env, so it is correct under RSI / adaptive
  sampling (which can start an env partway through the clip).
- **upright**: `clamp(-projected_gravity_b.z, 0, 1)`. 1 when the torso is
  vertical, 0 when inverted / on its back. (`projected_gravity_b` is read off
  the robot entity data, same source the velocity-task `upright` reward uses.)
- **feet_down**: per-foot clamped ramp on foot height above ground,
  `clamp((foot_height_target + margin - foot_height) / margin, 0, 1)`, averaged
  over `left_ankle_roll_link` and `right_ankle_roll_link`. 1 at/below
  `foot_height_target` (0.12 m), 0 at/above `target + margin` (0.27 m).
  Height is `body_link_pos_w.z - env_origin.z - ground_z` (terrain-safe).
  NB: the tracking scene has **no feet contact sensor** (only `self_collision`),
  and adding one would touch the shared scene, so we use foot height instead of
  a contact sensor. This keeps the term fully self-contained.

Default params (set in the base config, overridable on the CLI):
`foot_height_target=0.12`, `foot_height_margin=0.15`, `phase_gate_start=0.6`,
`ground_z=0.0`. Foot bodies are set per-robot in the G1 config (empty in base).

NaN-safety: only bounded clamps and products; every divisor is a positive
constant (`max(..., 1e-6)`); no `exp` of an unbounded input. The term cannot
emit NaN/Inf, so it is safe to leave registered at weight 0.0.

## Files created / modified on the Spark (in `mjlab-dev`)

All deployed by writing to `/tmp` on the Spark host then `docker cp` into
`/workspace/mjlab/...` (the tree is root-owned; not chowned). Local mirrors of
the final container versions are saved under this directory.

1. MODIFIED `src/mjlab/tasks/tracking/mdp/rewards.py`
   - Appended the `landing_feet_upright` function at the END of the file
     (nothing above it was changed).
   - Added one import: `from mjlab.managers.scene_entity_config import
     SceneEntityCfg` (used in the new function signature). Ruff reordered the
     import block alphabetically (the only thing `ruff check --fix` touched).
   - Diff: +83 lines, 0 deletions.

2. MODIFIED `src/mjlab/tasks/tracking/tracking_env_cfg.py`
   - Added the `"landing_feet_upright"` entry to the `rewards` dict, immediately
     after `"self_collisions"`, with `weight=0.0` and default params (foot
     `asset_cfg` body_names empty, set per-robot). No existing term touched.
   - Diff: +16 lines, 0 deletions.

3. MODIFIED `src/mjlab/tasks/tracking/config/g1/env_cfgs.py`
   - Set the foot body names on the landing term's `asset_cfg`
     (`left_ankle_roll_link`, `right_ankle_roll_link`), mirroring how
     `ee_body_pos` foot bodies are set. No existing line touched.
   - Diff: +8 lines, 0 deletions.

No new files were created in mjlab; the function lives in the task's existing
`mdp/rewards.py` and is already re-exported via `mdp/__init__.py`
(`from .rewards import *`), so `mdp.landing_feet_upright` resolves with no
`__init__` change needed.

`git diff --stat` (in the container): 3 files changed, 107 insertions(+),
0 deletions. Purely additive; no existing reward weight, threshold, func, or
param was altered.

## Verification (no GPU / no physics; GPU is busy training)

1. `uvx ruff@latest format` + `ruff check --fix` on `src/mjlab/tasks/tracking`:
   clean ("All checks passed!", 1 import-order fix applied, 15 files unchanged).
2. `uvx pyright@1.1.408 --pythonpath /usr/bin/python src/mjlab/tasks/tracking`:
   `0 errors, 0 warnings, 0 informations`.
3. Load-check (config build only, no physics):
   `load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1")` ->
   `LANDING_WEIGHT 0.0`, `CARTWHEEL_SAFE True`,
   `FOOT_BODIES ('left_ankle_roll_link', 'right_ankle_roll_link')`,
   `GATE_START 0.6`, `FOOT_TARGET 0.12`, `FUNC landing_feet_upright`.
   Also verified the `-No-State-Estimation` and `play=True` variants load with
   weight 0.0.
4. Diff-check: only reward-dict change is the added term; no existing weights /
   thresholds altered.

NOT run (GPU busy, per instructions): no training / physics smoke test.
A physics step has not exercised the function path at runtime; the math is
load-checked and statically type-checked only.
