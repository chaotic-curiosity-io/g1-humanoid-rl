"""The landing reward added to mjlab's tracking task for backflip v3.

This is the new function appended to
`mjlab/src/mjlab/tasks/tracking/mdp/rewards.py` on the Spark. It is registered
in `tracking_env_cfg.py` with weight 0.0 (off by default; cartwheel-safe) and
enabled for the backflip run via `--env.rewards.landing-feet-upright.weight 5.0`.
See README.md for the full design, the two other one-line edits, and the train command.
Imports (already present in rewards.py): torch, Entity, SceneEntityCfg, ManagerBasedRlEnv.
"""

def landing_feet_upright(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  foot_height_target: float = 0.12,
  foot_height_margin: float = 0.15,
  phase_gate_start: float = 0.6,
  ground_z: float = 0.0,
) -> torch.Tensor:
  """Reward landing on both feet, upright, gated to the landing phase.

  Designed for acrobatic tracking motions (e.g. a backflip) where the stock
  pose-matching reward pays for the inverted mid-flip but never explicitly
  rewards ending the clip standing on both feet. This term adds that signal,
  active only in the last part of the reference clip so it does not fight the
  inverted/aerial phase.

  reward = gate * upright * feet_down, all three in [0, 1], product in [0, 1].

  - gate: clamped linear ramp on the reference motion phase, 0 below
    ``phase_gate_start`` and 1 at the final frame. The phase is the per-env
    reference-frame index normalised by the clip length, so the gate follows
    wherever the policy is in the motion (robust to RSI / adaptive sampling,
    which can start an env partway through the clip).
  - upright: ``clamp(-projected_gravity_b.z, 0, 1)``. Equals 1 when the torso is
    vertical (gravity points straight down in body frame) and 0 when the robot
    is inverted / on its back. Monotonic, bounded, gradient everywhere.
  - feet_down: per-foot clamped ramp on the foot height above the ground, 1
    at/below ``foot_height_target`` and 0 at/above
    ``foot_height_target + foot_height_margin``; averaged over the configured
    foot bodies. No contact sensor is required (the tracking scene has none),
    so this term is fully self-contained and does not touch the shared scene.

  All operations are bounded clamps/products; there are no divisions by learned
  quantities and no exp of unbounded inputs, so the term cannot emit NaN/Inf and
  is safe to leave registered at weight 0.0 for non-backflip motions.

  Args:
    command_name: Name of the ``MotionCommand`` term (e.g. ``"motion"``).
    asset_cfg: Robot asset config whose ``body_names`` select the foot/ankle
      bodies to check for ground contact (resolved to ``body_ids``).
    foot_height_target: Foot z (above ground) at/below which a foot counts as
      fully "down" (score 1). Meters.
    foot_height_margin: Width of the ramp above the target over which the
      foot-down score decays to 0. Meters.
    phase_gate_start: Reference-motion phase in [0, 1] at which the landing gate
      begins to ramp in. The last ``1 - phase_gate_start`` fraction of the clip
      is the landing window.
    ground_z: World-frame ground height of the (flat) terrain. Subtracted along
      with the per-env origin so the check is correct regardless of env layout.

  Returns:
    Per-env reward tensor of shape [B], values in [0, 1].
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  asset = env.scene[asset_cfg.name]

  # Gate: reference motion phase in [0, 1], clamped ramp to the landing window.
  total = max(int(command.motion.time_step_total) - 1, 1)
  phase = command.time_steps.to(torch.float32) / float(total)  # [B], in [0, 1]
  gate_denom = max(1.0 - float(phase_gate_start), 1e-6)
  gate = torch.clamp((phase - float(phase_gate_start)) / gate_denom, 0.0, 1.0)  # [B]

  # Upright: 1 when torso vertical, 0 when inverted.
  proj_grav_b = asset.data.projected_gravity_b  # [B, 3], ~(0,0,-1) upright
  upright = torch.clamp(-proj_grav_b[:, 2], 0.0, 1.0)  # [B]

  # Feet down: clamped ramp on foot height above ground, averaged over feet.
  foot_ids = asset_cfg.body_ids
  foot_z_w = asset.data.body_link_pos_w[:, foot_ids, 2]  # [B, n_foot]
  env_origin_z = env.scene.env_origins[:, 2].unsqueeze(-1)  # [B, 1]
  foot_height = foot_z_w - env_origin_z - float(ground_z)  # [B, n_foot]
  margin = max(float(foot_height_margin), 1e-6)
  feet_down_per = torch.clamp(
    (float(foot_height_target) + margin - foot_height) / margin, 0.0, 1.0
  )  # [B, n_foot]
  feet_down = feet_down_per.mean(dim=-1)  # [B]

  return gate * upright * feet_down
