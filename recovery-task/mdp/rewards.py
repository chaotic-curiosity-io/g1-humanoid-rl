"""Recovery-task rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def stand_up_height_reward(
  env: ManagerBasedRlEnv,
  target_height: float,
  floor_height: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Monotonic reward on the root height: a clamped linear ramp from the floor
  to the standing target.

  Returns 0 when the pelvis is on the ground (``floor_height``) and 1 when it
  reaches ``target_height`` (the standing keyframe height), with a non-zero
  gradient *everywhere* in between. This is the key fix over a narrow Gaussian,
  which is ~flat at zero while the robot is down and so gives no signal to begin
  rising — the policy then has no gradient telling it to get off the ground.
  """
  asset: Entity = env.scene[asset_cfg.name]
  root_z = asset.data.root_link_pos_w[:, 2]
  progress = (root_z - floor_height) / (target_height - floor_height)
  return torch.clamp(progress, 0.0, 1.0)
