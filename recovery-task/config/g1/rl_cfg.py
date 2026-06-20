"""RL configuration for the Unitree G1 recovery (get-up) task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg


def unitree_g1_recovery_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO runner config for the G1 recovery task.

  Identical PPO architecture to the baseline G1 velocity config, but writes to
  a separate ``g1_recovery`` experiment directory so recovery checkpoints do
  not mix with the walk baseline.
  """
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_recovery"
  return cfg
