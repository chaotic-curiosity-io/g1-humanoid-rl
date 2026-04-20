"""Render a 'learning progression' video from a training run's checkpoints.

Walks a training run directory, loads each of N evenly-spaced checkpoints, and
records a short deterministic rollout (same seed, same fixed velocity command)
from a single robot. The clips are stitched into one MP4 per camera angle that
shows a single robot evolving from stumbling to walking as training progresses.

Supports rendering multiple named camera angles in a single pass (reusing the
simulation state), an optional 2x2 grid composition, and toggling manager/sensor
debug visuals off for clean frames.
"""

from __future__ import annotations

import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mediapy as media
import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


# Camera angle presets. Values are (azimuth_deg, elevation_deg, distance_m).
# In mujoco's free/tracking camera, elevation is negative when looking down at
# the subject and positive when looking up; azimuth rotates CCW around z.
CAMERA_PRESETS: dict[str, tuple[float, float, float]] = {
  "chase": (135.0, -10.0, 3.0),
  "side": (90.0, -5.0, 3.0),
  "front": (180.0, -5.0, 3.2),
  "top": (90.0, -70.0, 4.5),
  "hero": (60.0, -15.0, 3.5),
}


@dataclass(frozen=True)
class ProgressionConfig:
  run_dir: Path
  """Training run directory containing model_*.pt checkpoints."""

  output_dir: Path = Path("learning_progression")
  """Directory for the output videos. One MP4 per camera angle is written."""

  cameras: str = "chase,side,front,top"
  """Comma-separated list of camera angle presets to render (see CAMERA_PRESETS)."""

  grid: bool = True
  """Also emit a 2x2 grid composition video when >=2 cameras are requested."""

  num_checkpoints: int = 20
  """How many evenly-spaced checkpoints to sample."""

  frames_per_checkpoint: int = 60
  """Frames to record per checkpoint clip (after warmup drop)."""

  warmup_frames: int = 10
  """Frames to discard at the start of each clip."""

  command_lin_vel_x: float = 1.0
  command_lin_vel_y: float = 0.0
  command_ang_vel_z: float = 0.0

  seed: int = 42
  width: int = 1920
  height: int = 1080
  fps: int | None = None
  """Output video fps. Defaults to the env's render fps."""

  device: str | None = None
  label_iterations: bool = True
  include_first: bool = True
  include_last: bool = True
  debug_viz: bool = False
  """If False, skip manager/sensor debug visualizations while recording."""

  shadows: bool = False
  """If False, disable shadow casting (removes shadow-map acne/jitter)."""

  reflections: bool = False
  """If False, disable ground material reflections (removes reflection shimmer)."""

  shadow_size: int = 4096
  """Shadow map resolution when shadows are enabled."""

  crossfade_frames: int = 0
  """If >0, overlap adjacent clips with a linear crossfade of N frames."""

  checkpoint_stride: int | None = None
  """If set, overrides num_checkpoints: take every Nth checkpoint instead."""

  grid_scale: float = 0.5
  """Per-tile downscale applied before composing the grid (1.0 = full res)."""


def _discover_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
  pat = re.compile(r"^model_(\d+)\.pt$")
  ckpts: list[tuple[int, Path]] = []
  for p in run_dir.glob("model_*.pt"):
    m = pat.match(p.name)
    if m is None:
      continue
    ckpts.append((int(m.group(1)), p))
  ckpts.sort(key=lambda x: x[0])
  return ckpts


def _select_checkpoints(
  all_ckpts: list[tuple[int, Path]], cfg: ProgressionConfig
) -> list[tuple[int, Path]]:
  if not all_ckpts:
    raise RuntimeError("No checkpoints found in run_dir.")

  if cfg.checkpoint_stride is not None and cfg.checkpoint_stride > 0:
    chosen = all_ckpts[:: cfg.checkpoint_stride]
  else:
    n = min(cfg.num_checkpoints, len(all_ckpts))
    if n <= 1:
      chosen = [all_ckpts[-1]]
    else:
      idxs = np.linspace(0, len(all_ckpts) - 1, num=n).round().astype(int)
      chosen = [all_ckpts[i] for i in idxs]

  present_iters = {it for it, _ in chosen}
  if cfg.include_first and all_ckpts[0][0] not in present_iters:
    chosen.insert(0, all_ckpts[0])
  if cfg.include_last and all_ckpts[-1][0] not in present_iters:
    chosen.append(all_ckpts[-1])

  seen: set[int] = set()
  deduped: list[tuple[int, Path]] = []
  for it, p in chosen:
    if it in seen:
      continue
    seen.add(it)
    deduped.append((it, p))
  deduped.sort(key=lambda x: x[0])
  return deduped


def _infer_task_id(run_dir: Path, available: list[str]) -> str:
  experiment_name = run_dir.parent.name
  candidates: list[str] = []
  for task_id in available:
    rl_cfg = load_rl_cfg(task_id)
    if getattr(rl_cfg, "experiment_name", None) == experiment_name:
      candidates.append(task_id)
  if not candidates:
    raise RuntimeError(
      f"Could not infer task id for experiment '{experiment_name}'. "
      f"Pass --task-id explicitly. Available: {available}"
    )
  if len(candidates) > 1:
    raise RuntimeError(
      f"Multiple tasks share experiment '{experiment_name}': {candidates}. "
      "Pass --task-id explicitly."
    )
  return candidates[0]


def _force_fixed_command(env: ManagerBasedRlEnv, cfg: ProgressionConfig) -> None:
  cmd_mgr = env.command_manager
  if "twist" not in cmd_mgr.active_terms:
    return
  term = cmd_mgr.get_term("twist")
  value = torch.tensor(
    [cfg.command_lin_vel_x, cfg.command_lin_vel_y, cfg.command_ang_vel_z],
    device=term.vel_command_b.device,
    dtype=term.vel_command_b.dtype,
  )
  term.vel_command_b[:] = value
  if hasattr(term, "vel_command_w"):
    term.vel_command_w[:] = value
  for flag in ("is_heading_env", "is_standing_env", "is_world_env", "is_forward_env"):
    buf = getattr(term, flag, None)
    if isinstance(buf, torch.Tensor):
      buf.zero_()


def _label_frame(
  frame: np.ndarray,
  iteration: int,
  progress: float,
  total_iters: int,
  camera_name: str | None = None,
) -> np.ndarray:
  try:
    from PIL import Image, ImageDraw, ImageFont
  except ImportError:
    return frame

  img = Image.fromarray(frame)
  draw = ImageDraw.Draw(img, "RGBA")
  w, h = img.size

  try:
    font_size = max(18, h // 24)
    font = ImageFont.truetype(
      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
    )
  except OSError:
    font = ImageFont.load_default()

  label = f"iter {iteration:>5} / {total_iters}"
  pad = max(8, h // 80)
  bbox = draw.textbbox((0, 0), label, font=font)
  tw = bbox[2] - bbox[0]
  th = bbox[3] - bbox[1]
  draw.rectangle(
    (pad, pad, pad + tw + 2 * pad, pad + th + 2 * pad),
    fill=(0, 0, 0, 140),
  )
  draw.text((pad * 2, pad * 1.5), label, fill=(255, 255, 255, 255), font=font)

  if camera_name is not None:
    cam_label = camera_name
    cbbox = draw.textbbox((0, 0), cam_label, font=font)
    ctw = cbbox[2] - cbbox[0]
    cth = cbbox[3] - cbbox[1]
    draw.rectangle(
      (w - pad - ctw - 2 * pad, pad, w - pad, pad + cth + 2 * pad),
      fill=(0, 0, 0, 140),
    )
    draw.text(
      (w - pad - ctw - pad, pad * 1.5), cam_label, fill=(255, 255, 255, 255), font=font
    )

  bar_h = max(4, h // 120)
  margin = pad * 2
  bar_y = h - margin - bar_h
  draw.rectangle(
    (margin, bar_y, w - margin, bar_y + bar_h), fill=(255, 255, 255, 60)
  )
  filled = int((w - 2 * margin) * max(0.0, min(1.0, progress)))
  if filled > 0:
    draw.rectangle(
      (margin, bar_y, margin + filled, bar_y + bar_h),
      fill=(255, 255, 255, 220),
    )

  return np.asarray(img)


def _crossfade(
  a: list[np.ndarray], b: list[np.ndarray], n: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
  if n <= 0 or not a or not b:
    return a, [], b
  n = min(n, len(a), len(b))
  a_head = a[:-n]
  b_tail = b[n:]
  blended: list[np.ndarray] = []
  for i in range(n):
    alpha = (i + 1) / (n + 1)
    fa = a[-n + i].astype(np.float32)
    fb = b[i].astype(np.float32)
    mix = ((1 - alpha) * fa + alpha * fb).clip(0, 255).astype(np.uint8)
    blended.append(mix)
  return a_head, blended, b_tail


def _parse_cameras(spec: str) -> list[str]:
  names = [n.strip() for n in spec.split(",") if n.strip()]
  unknown = [n for n in names if n not in CAMERA_PRESETS]
  if unknown:
    raise ValueError(
      f"Unknown camera preset(s): {unknown}. "
      f"Available: {sorted(CAMERA_PRESETS.keys())}"
    )
  if not names:
    raise ValueError("At least one camera must be specified.")
  return names


def _set_camera(renderer, name: str) -> None:
  """Apply a named camera preset to the offscreen renderer's free camera."""
  az, el, dist = CAMERA_PRESETS[name]
  cam = renderer._cam
  cam.azimuth = az
  cam.elevation = el
  cam.distance = dist


def _postprocess_frame(
  frame: np.ndarray,
  iteration: int,
  progress: float,
  total_iters: int,
  camera_name: str,
  cfg: ProgressionConfig,
) -> np.ndarray:
  if isinstance(frame, np.ndarray) and frame.ndim == 4:
    frame = frame[0]
  if frame.dtype != np.uint8:
    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
  if cfg.label_iterations:
    frame = _label_frame(
      frame, iteration, progress, total_iters, camera_name=camera_name
    )
  return frame


def _compose_grid(
  per_camera_frames: dict[str, list[np.ndarray]],
  camera_order: list[str],
  scale: float,
) -> list[np.ndarray]:
  """Tile per-camera frame lists into a 2x2 (or 1xN) grid, frame by frame."""
  from PIL import Image

  n = min(len(frames) for frames in per_camera_frames.values())
  if n == 0:
    return []

  cols = 2 if len(camera_order) >= 2 else 1
  rows = (len(camera_order) + cols - 1) // cols

  sample = per_camera_frames[camera_order[0]][0]
  tile_w = max(1, int(sample.shape[1] * scale))
  tile_h = max(1, int(sample.shape[0] * scale))
  grid_w = tile_w * cols
  grid_h = tile_h * rows

  out: list[np.ndarray] = []
  for i in range(n):
    canvas = Image.new("RGB", (grid_w, grid_h), (0, 0, 0))
    for idx, cam in enumerate(camera_order):
      r, c = idx // cols, idx % cols
      tile = Image.fromarray(per_camera_frames[cam][i])
      if scale != 1.0:
        tile = tile.resize((tile_w, tile_h), Image.BILINEAR)
      canvas.paste(tile, (c * tile_w, r * tile_h))
    out.append(np.asarray(canvas))
  return out


def run_recording(task_id: str, cfg: ProgressionConfig) -> None:
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  run_dir = cfg.run_dir.expanduser().resolve()
  if not run_dir.is_dir():
    raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

  cameras = _parse_cameras(cfg.cameras)

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = 1
  env_cfg.viewer.width = cfg.width
  env_cfg.viewer.height = cfg.height
  env_cfg.viewer.enable_shadows = cfg.shadows
  env_cfg.viewer.enable_reflections = cfg.reflections
  env_cfg.seed = cfg.seed

  all_ckpts = _discover_checkpoints(run_dir)
  ckpts = _select_checkpoints(all_ckpts, cfg)
  total_iters = all_ckpts[-1][0]
  print(f"[progress] {len(ckpts)} / {len(all_ckpts)} checkpoints selected.")
  print(f"[progress] iters: {[it for it, _ in ckpts]}")
  print(f"[progress] cameras: {cameras}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  if cfg.shadows and env._offline_renderer is not None:
    # Bump shadow precision to reduce acne/jitter on the ground.
    env._offline_renderer._model.vis.quality.shadowsize = cfg.shadow_size
  if not cfg.debug_viz:
    # Skip manager/sensor debug drawing to avoid ground jitter from visualizers.
    env.update_visualizers = lambda *a, **k: None  # type: ignore[method-assign]

  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)

  render_fps = int(round(env.metadata.get("render_fps", 30)))
  out_fps = cfg.fps or render_fps
  print(f"[progress] env render_fps={render_fps}, output fps={out_fps}")

  if env._offline_renderer is None:
    raise RuntimeError("Expected env._offline_renderer to be initialized.")
  offline = env._offline_renderer

  clip_frames_per_cam: dict[str, list[list[np.ndarray]]] = {c: [] for c in cameras}

  for i, (iteration, ckpt_path) in enumerate(ckpts):
    print(f"[progress] [{i + 1}/{len(ckpts)}] iter={iteration} :: {ckpt_path.name}")
    runner.load(
      str(ckpt_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

    env.seed(cfg.seed)
    wrapped.reset()
    _force_fixed_command(env, cfg)

    per_cam_frames: dict[str, list[np.ndarray]] = {c: [] for c in cameras}
    progress = i / max(1, len(ckpts) - 1)

    with torch.no_grad():
      for step in range(cfg.frames_per_checkpoint + cfg.warmup_frames):
        _force_fixed_command(env, cfg)
        obs = wrapped.get_observations()
        action = policy(obs)
        wrapped.step(action)
        if step < cfg.warmup_frames:
          continue
        for cam_name in cameras:
          _set_camera(offline, cam_name)
          frame = env.render()
          if frame is None:
            continue
          frame = _postprocess_frame(
            frame, iteration, progress, total_iters, cam_name, cfg
          )
          per_cam_frames[cam_name].append(frame)

    for cam_name in cameras:
      clip_frames_per_cam[cam_name].append(per_cam_frames[cam_name])

  env.close()

  # Stitch per-camera videos.
  cfg.output_dir.mkdir(parents=True, exist_ok=True)
  per_camera_final: dict[str, list[np.ndarray]] = {}
  for cam_name, clips in clip_frames_per_cam.items():
    out_frames: list[np.ndarray] = []
    if cfg.crossfade_frames > 0 and len(clips) > 1:
      for i in range(len(clips)):
        if i == 0:
          continue
        head, blended, tail = _crossfade(clips[i - 1], clips[i], cfg.crossfade_frames)
        clips[i - 1] = head
        clips[i] = tail
        out_frames.extend(head)
        out_frames.extend(blended)
      out_frames.extend(clips[-1])
    else:
      for frames in clips:
        out_frames.extend(frames)

    if not out_frames:
      print(f"[progress] [WARN] no frames for camera {cam_name}, skipping.")
      continue

    path = cfg.output_dir / f"{cam_name}.mp4"
    print(f"[progress] writing {len(out_frames)} frames -> {path}")
    media.write_video(str(path), out_frames, fps=out_fps)
    per_camera_final[cam_name] = out_frames

  if cfg.grid and len(per_camera_final) >= 2:
    print(f"[progress] composing grid ({len(per_camera_final)} cams)")
    grid_frames = _compose_grid(per_camera_final, cameras, cfg.grid_scale)
    if grid_frames:
      path = cfg.output_dir / "grid.mp4"
      print(f"[progress] writing {len(grid_frames)} frames -> {path}")
      media.write_video(str(path), grid_frames, fps=out_fps)

  print(f"[progress] done. duration ~{len(next(iter(per_camera_final.values()))) / out_fps:.1f}s")


def main() -> None:
  maybe_print_top_level_help("record-learning-progression")

  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks + ["auto"]),
    add_help=False,
    return_unknown_args=True,
  )

  cfg = tyro.cli(
    ProgressionConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
  )

  if chosen_task == "auto":
    task_id = _infer_task_id(cfg.run_dir.expanduser().resolve(), all_tasks)
    print(f"[progress] auto-detected task: {task_id}")
  else:
    task_id = chosen_task

  run_recording(task_id, cfg)


if __name__ == "__main__":
  main()
