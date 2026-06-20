"""Recovery-task reset events.

The robot must learn to get up from the ground, so it is reset into a genuinely
fallen pose (rather than the upright keyframe the velocity task uses).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_root_state_lying_down(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  height_range: tuple[float, float] = (0.08, 0.15),
  yaw_range: tuple[float, float] = (-3.14, 3.14),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset the robot root to a genuinely fallen lying pose.

  Each environment is randomly assigned one of four fallen orientations, then
  a random yaw is applied on top:

  * Supine (on back):   roll=pi,    pitch=0
  * Face-down (prone):  roll=0,     pitch=pi/2
  * Left side:          roll=pi/2,  pitch=0
  * Right side:         roll=-pi/2, pitch=0

  The pelvis is sampled a small distance above the ground plane and all
  velocities are zeroed.

  Args:
    env: The environment.
    env_ids: Environment IDs to reset. If None, resets all environments.
    height_range: Range for pelvis z above ``env.scene.env_origins``.
      Widened to 0.08-0.15 m to give the torso clearance for all four poses.
    yaw_range: Range for the yaw of the lying pose.
    asset_cfg: Asset configuration.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[asset_cfg.name]
  n = len(env_ids)

  # Random yaw about the world up axis.
  yaw = sample_uniform(yaw_range[0], yaw_range[1], (n,), env.device)

  # Four genuinely-fallen orientations, chosen uniformly at random per env.
  # Index: 0=supine, 1=face-down, 2=left-side, 3=right-side.
  pose_idx = torch.randint(0, 4, (n,), device=env.device)

  roll_values = torch.tensor(
    [math.pi, 0.0, math.pi / 2, -math.pi / 2], device=env.device
  )
  pitch_values = torch.tensor([0.0, math.pi / 2, 0.0, 0.0], device=env.device)

  roll = roll_values[pose_idx]
  pitch = pitch_values[pose_idx]

  orientations = quat_from_euler_xyz(roll, pitch, yaw)  # [N, 4] (w, x, y, z).

  # Pelvis position: env origin + small z offset off the ground.
  z = sample_uniform(height_range[0], height_range[1], (n,), env.device)
  positions = env.scene.env_origins[env_ids].clone()
  positions[:, 2] = positions[:, 2] + z

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )

  # Zero linear + angular velocity.
  asset.write_root_link_velocity_to_sim(
    torch.zeros(n, 6, device=env.device), env_ids=env_ids
  )
