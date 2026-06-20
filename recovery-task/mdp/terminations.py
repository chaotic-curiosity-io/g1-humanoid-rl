"""Recovery-task terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def stood_up(
  env: ManagerBasedRlEnv,
  min_height: float,
  max_tilt_rad: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """True where the robot has successfully stood up.

  Success = root higher than ``min_height`` AND torso tilt (angle between the
  body up axis and world up) below ``max_tilt_rad``. Tilt is derived from the
  projected gravity z-component: ``acos(clamp(-pg_z, -1, 1))``. Instantaneous
  (single-step) check — sufficient for v1.
  """
  asset: Entity = env.scene[asset_cfg.name]

  root_z = asset.data.root_link_pos_w[:, 2]
  height_ok = root_z > min_height

  projected_gravity = asset.data.projected_gravity_b
  tilt = torch.acos(torch.clamp(-projected_gravity[:, 2], -1.0, 1.0))
  upright_ok = tilt < max_tilt_rad

  return height_ok & upright_ok
