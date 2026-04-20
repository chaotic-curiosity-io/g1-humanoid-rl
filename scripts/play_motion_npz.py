"""Render an mjlab tracking motion.npz to a video.

Drives the G1 env's qpos frame-by-frame from the npz (root pose + joint pos),
rendering multiple camera angles via mjlab's offscreen renderer. No training,
no agent — just visualizes the reference motion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mediapy as media
import numpy as np
from PIL import Image

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends


CAMERA_PRESETS = {
  "chase": (135.0, -10.0, 3.0),
  "side": (90.0, -5.0, 3.0),
  "front": (180.0, -5.0, 3.2),
  "top": (90.0, -70.0, 4.5),
}


def _set_camera(offline_renderer, name: str) -> None:
  az, el, dist = CAMERA_PRESETS[name]
  cam = offline_renderer._cam
  cam.azimuth = az
  cam.elevation = el
  cam.distance = dist


def _compose_grid(per_cam_frames, cams, scale: float = 0.5):
  n = min(len(f) for f in per_cam_frames.values())
  cols = 2 if len(cams) >= 2 else 1
  rows = (len(cams) + cols - 1) // cols
  sample = per_cam_frames[cams[0]][0]
  tw = max(1, int(sample.shape[1] * scale))
  th = max(1, int(sample.shape[0] * scale))
  out = []
  for i in range(n):
    canvas = Image.new("RGB", (tw * cols, th * rows), (0, 0, 0))
    for idx, cam in enumerate(cams):
      r, c = idx // cols, idx % cols
      tile = Image.fromarray(per_cam_frames[cam][i])
      if scale != 1.0:
        tile = tile.resize((tw, th), Image.BILINEAR)
      canvas.paste(tile, (c * tw, r * th))
    out.append(np.asarray(canvas))
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--motion-file", required=True)
  ap.add_argument("--task-id", default="Mjlab-Velocity-Flat-Unitree-G1")
  ap.add_argument("--output-dir", required=True)
  ap.add_argument("--cameras", default="chase,side,front,top")
  ap.add_argument("--width", type=int, default=1920)
  ap.add_argument("--height", type=int, default=1080)
  ap.add_argument("--shadows", type=int, default=0)
  ap.add_argument("--reflections", type=int, default=0)
  ap.add_argument("--loops", type=int, default=2)
  ap.add_argument("--hold-frames", type=int, default=25)
  args = ap.parse_args()

  configure_torch_backends()

  data = np.load(args.motion_file)
  fps = int(round(float(data["fps"][0])))
  body_pos_w = data["body_pos_w"]
  body_quat_w = data["body_quat_w"]
  joint_pos = data["joint_pos"]
  n_frames = joint_pos.shape[0]
  print(f"[play] {n_frames} frames at {fps} fps, duration {n_frames/fps:.2f}s")

  env_cfg = load_env_cfg(args.task_id, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.viewer.width = args.width
  env_cfg.viewer.height = args.height
  env_cfg.viewer.enable_shadows = bool(args.shadows)
  env_cfg.viewer.enable_reflections = bool(args.reflections)

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
  env.update_visualizers = lambda *a, **k: None  # type: ignore[method-assign]
  offline = env._offline_renderer
  assert offline is not None

  cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
  per_cam: dict[str, list[np.ndarray]] = {c: [] for c in cams}

  import torch
  import mujoco as mj

  robot = env.scene["robot"]
  mj_model = env.sim.mj_model
  body_root_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, "pelvis")

  def set_frame(i: int) -> None:
    # Set qpos directly on the first env via warp bindings.
    # body_pos_w[:, 0] is the pelvis (root) world pos.
    root_pos = torch.tensor(body_pos_w[i, 0], device="cuda:0", dtype=torch.float32)
    root_quat = torch.tensor(body_quat_w[i, 0], device="cuda:0", dtype=torch.float32)
    jp = torch.tensor(joint_pos[i], device="cuda:0", dtype=torch.float32).unsqueeze(0)
    jv = torch.zeros_like(jp)
    # root state: [pos(3), quat_wxyz(4), lin_vel(3), ang_vel(3)]
    root_state = torch.zeros(1, 13, device="cuda:0", dtype=torch.float32)
    root_state[0, 0:3] = root_pos
    root_state[0, 3:7] = root_quat
    env_ids = torch.tensor([0], device="cuda:0", dtype=torch.long)
    robot.write_root_state_to_sim(root_state, env_ids)
    robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()

  # Warm-up: place the robot in the first frame's pose.
  env.reset()
  set_frame(0)
  for _ in range(args.hold_frames):
    for c in cams:
      _set_camera(offline, c)
      f = env.render()
      if f is None:
        continue
      if f.ndim == 4:
        f = f[0]
      per_cam[c].append(f)

  # Playback loops.
  for loop in range(args.loops):
    for i in range(n_frames):
      set_frame(i)
      for c in cams:
        _set_camera(offline, c)
        f = env.render()
        if f is None:
          continue
        if f.ndim == 4:
          f = f[0]
        per_cam[c].append(f)

  # Hold final frame.
  set_frame(n_frames - 1)
  for _ in range(args.hold_frames):
    for c in cams:
      _set_camera(offline, c)
      f = env.render()
      if f is None:
        continue
      if f.ndim == 4:
        f = f[0]
      per_cam[c].append(f)

  env.close()

  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  for c, frames in per_cam.items():
    path = out_dir / f"{c}.mp4"
    media.write_video(str(path), frames, fps=fps)
    print(f"[play] wrote {len(frames)} frames -> {path}")
  if len(cams) >= 2:
    grid = _compose_grid(per_cam, cams, scale=0.5)
    path = out_dir / "grid.mp4"
    media.write_video(str(path), grid, fps=fps)
    print(f"[play] wrote {len(grid)} frames -> {path}")


if __name__ == "__main__":
  main()
