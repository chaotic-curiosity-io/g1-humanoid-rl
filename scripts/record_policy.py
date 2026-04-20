"""Render an mjlab tracking policy checkpoint to a multi-camera video (offscreen).

Drives a 1-env play-mode tracking env, runs the trained policy for N steps,
and records offscreen frames from several preset cameras. No viser, no viewer
—just a clean MP4 file per angle plus an optional 2x2 grid. Intended to be
runnable on CPU while a training job owns the GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import mediapy as media
import numpy as np
import torch
from PIL import Image

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends


CAMERA_PRESETS = {
  "chase": (135.0, -10.0, 3.0),
  "side": (90.0, -5.0, 3.0),
  "front": (180.0, -5.0, 3.2),
  "top": (90.0, -70.0, 4.5),
}


def _set_camera(offline, name: str) -> None:
  az, el, dist = CAMERA_PRESETS[name]
  cam = offline._cam
  cam.azimuth = az
  cam.elevation = el
  cam.distance = dist


def _compose_grid(per_cam, cams, scale: float = 0.5) -> list[np.ndarray]:
  n = min(len(v) for v in per_cam.values())
  cols = 2 if len(cams) >= 2 else 1
  rows = (len(cams) + cols - 1) // cols
  sample = per_cam[cams[0]][0]
  tw = max(1, int(sample.shape[1] * scale))
  th = max(1, int(sample.shape[0] * scale))
  out = []
  for i in range(n):
    canvas = Image.new("RGB", (tw * cols, th * rows), (0, 0, 0))
    for idx, cam in enumerate(cams):
      r, c = idx // cols, idx % cols
      tile = Image.fromarray(per_cam[cam][i])
      if scale != 1.0:
        tile = tile.resize((tw, th), Image.BILINEAR)
      canvas.paste(tile, (c * tw, r * th))
    out.append(np.asarray(canvas))
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task-id", default="Mjlab-Tracking-Flat-Unitree-G1")
  ap.add_argument("--checkpoint-file", required=True)
  ap.add_argument("--motion-file", required=True)
  ap.add_argument("--output-dir", required=True)
  ap.add_argument("--cameras", default="chase,side,front,top")
  ap.add_argument("--num-steps", type=int, default=500)
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=720)
  ap.add_argument("--device", default="cpu")
  ap.add_argument("--show-ghost", type=int, default=1,
                  help="If 1, keep motion-command debug visualizers enabled (reference ghost).")
  ap.add_argument("--dump-telemetry", type=str, default=None,
                  help="If set, dump per-step pelvis pose + env terminations to this .npz.")
  ap.add_argument("--disable-terminations", type=int, default=0,
                  help="If 1, remove all non-time-out terminations so the policy runs the full reference.")
  ap.add_argument("--termination-threshold", type=float, default=None,
                  help="If set, override anchor_pos/ee_body_pos thresholds (match training).")
  args = ap.parse_args()

  configure_torch_backends()

  env_cfg = load_env_cfg(args.task_id, play=True)
  agent_cfg = load_rl_cfg(args.task_id)

  # Wire the motion file through the tracking command cfg.
  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = args.motion_file

  env_cfg.scene.num_envs = 1
  env_cfg.viewer.width = args.width
  env_cfg.viewer.height = args.height
  env_cfg.viewer.enable_shadows = False
  env_cfg.viewer.enable_reflections = False

  if args.disable_terminations:
    keep = {"time_out"}
    env_cfg.terminations = {k: v for k, v in env_cfg.terminations.items() if k in keep}
    print(f"[record] terminations reduced to: {list(env_cfg.terminations.keys())}")
  elif args.termination_threshold is not None:
    for name in ("anchor_pos", "ee_body_pos"):
      if name in env_cfg.terminations:
        env_cfg.terminations[name].params["threshold"] = args.termination_threshold
    print(f"[record] termination thresholds set to {args.termination_threshold}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode="rgb_array")
  if not args.show_ghost:
    env.update_visualizers = lambda *a, **k: None  # type: ignore[method-assign]
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
  runner.load(
    args.checkpoint_file,
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)

  offline = env._offline_renderer
  assert offline is not None

  cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
  per_cam: dict[str, list[np.ndarray]] = {c: [] for c in cams}
  fps = int(round(env.metadata.get("render_fps", 30)))
  print(f"[record] render_fps={fps}, num_steps={args.num_steps}")

  telem_pelvis_pos: list[np.ndarray] = []
  telem_pelvis_quat: list[np.ndarray] = []
  telem_reset: list[int] = []

  robot = env.scene["robot"]
  import mujoco as mj
  mj_model = env.sim.mj_model
  pelvis_body_idx = robot.body_names.index("pelvis")

  with torch.no_grad():
    wrapped.reset()
    for step in range(args.num_steps):
      obs = wrapped.get_observations()
      action = policy(obs)
      _, _, dones, _ = wrapped.step(action)
      if args.dump_telemetry is not None:
        pelvis_pos = robot.data.body_link_pos_w[0, pelvis_body_idx].cpu().numpy().copy()
        pelvis_quat = robot.data.body_link_quat_w[0, pelvis_body_idx].cpu().numpy().copy()
        telem_pelvis_pos.append(pelvis_pos)
        telem_pelvis_quat.append(pelvis_quat)
        telem_reset.append(int(dones[0].item()))
      for cam_name in cams:
        _set_camera(offline, cam_name)
        frame = env.render()
        if frame is None:
          continue
        if frame.ndim == 4:
          frame = frame[0]
        if frame.dtype != np.uint8:
          frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        per_cam[cam_name].append(frame)
      if step % 50 == 0:
        print(f"[record] step {step}/{args.num_steps}")

  env.close()

  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  for cam, frames in per_cam.items():
    path = out_dir / f"{cam}.mp4"
    media.write_video(str(path), frames, fps=fps)
    print(f"[record] wrote {len(frames)} frames -> {path}")
  if len(cams) >= 2:
    grid = _compose_grid(per_cam, cams, scale=0.5)
    media.write_video(str(out_dir / "grid.mp4"), grid, fps=fps)
    print(f"[record] wrote {len(grid)} frames -> {out_dir/'grid.mp4'}")
  if args.dump_telemetry is not None:
    np.savez(
      args.dump_telemetry,
      pelvis_pos=np.array(telem_pelvis_pos),
      pelvis_quat=np.array(telem_pelvis_quat),
      reset=np.array(telem_reset, dtype=np.int32),
      fps=np.array([fps], dtype=np.int64),
    )
    print(f"[record] wrote telemetry -> {args.dump_telemetry}")


if __name__ == "__main__":
  main()
