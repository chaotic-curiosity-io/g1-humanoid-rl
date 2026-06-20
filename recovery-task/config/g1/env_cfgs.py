"""Unitree G1 recovery (get-up / fall-recovery) environment configuration.

Built by cloning the flat velocity config and mutating it: the robot starts
lying on the ground (prone or supine) and is rewarded for standing back up.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.recovery.mdp import (
  reset_root_state_lying_down,
  stand_up_height_reward,
)
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


def unitree_g1_recovery_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Unitree G1 fall-recovery configuration.

  Starts from the flat velocity config (plane terrain, G1 robot, full reward
  shaping) and converts it into a get-up task.
  """
  cfg = unitree_g1_flat_env_cfg(play=play)

  ##
  # Commands: zero the twist command (no locomotion target).
  ##
  # The "pose" reward depends on the twist command, so we keep the command but
  # pin every range to zero and make every env a standing env.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.rel_standing_envs = 1.0

  ##
  # Events: reset the robot lying down instead of upright.
  ##
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_root_state_lying_down,
    mode="reset",
    params={
      "height_range": (0.08, 0.15),
      "yaw_range": (-3.14, 3.14),
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  # Keep joint scatter, but loosen it so the lying pose is varied.
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.2, 0.2)

  ##
  # Terminations: the robot STARTS fallen, so drop fall/bounds terms and add a
  # success termination for standing up.
  ##
  cfg.terminations.pop("fell_over")
  cfg.terminations.pop("out_of_terrain_bounds", None)
  # v3: do NOT terminate on success. Ending the episode the moment the robot
  # stands up *removes* all its remaining reward, so in v2 the policy learned to
  # hover in a half-crouch (~0.5 m) and farm the height reward for the full
  # episode rather than finish the stand (which would cut the episode short).
  # With only a timeout, the reward-maximizing behavior is to get up fast and
  # HOLD the stand for the whole episode (height=1.0 + upright maxed every step).
  # (`stood_up` lives in mjlab.tasks.recovery.mdp for the offline eval script;
  # it is no longer wired as a termination here.)

  ##
  # Rewards: drop foot-locomotion rewards (meaningless while lying), add a
  # stand-up height reward.
  ##
  for reward_name in (
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "soft_landing",
    # v2: drop the velocity-tracking terms. With a zero command they reward
    # *zero velocity* — i.e. lying perfectly still — which a fallen robot
    # satisfies trivially. In v1 this "stillness hack" dominated: reward rose
    # to ~62 while the stood_up success rate decayed to ~0.08 as the policy
    # learned that lying still out-scored the risky business of getting up.
    "track_linear_velocity",
    "track_angular_velocity",
  ):
    cfg.rewards.pop(reward_name, None)
  # Kept: upright, pose, dof_pos_limits, action_rate_l2, body_ang_vel,
  # angular_momentum, self_collisions.
  # v2: strengthen the orientation signal so the robot must become vertical,
  # not merely raise its pelvis (a bridge/headstand would game height alone).
  cfg.rewards["upright"].weight = 2.0
  # v4: cut the action-rate penalty so the active balancing needed to extend
  # fully is cheap. In v3 the robot settled into a stable wide deep-squat (~0.52 m,
  # upright) because rising the last bit costs balance effort for little marginal
  # height reward — a low-effort local optimum. Reducing this penalty + raising
  # the height weight below tilts the trade toward a full stand.
  cfg.rewards["action_rate_l2"].weight = -0.03
  # v2: monotonic height ramp (gradient from the ground). v4: weight 5 -> 10 so
  # the last 0.24 m from crouch to full stand is clearly worth the effort.
  cfg.rewards["stand_up"] = RewardTermCfg(
    func=stand_up_height_reward,
    weight=10.0,
    params={
      "target_height": 0.76,
      "floor_height": 0.1,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  ##
  # Curriculum: none for recovery.
  ##
  cfg.curriculum = {}

  return cfg
