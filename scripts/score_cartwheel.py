"""Score cartwheel attempts from a record_policy.py telemetry dump.

Input: .npz with fields pelvis_pos (T, 3), pelvis_quat (T, 4 wxyz), reset (T,), fps (1,).
Output: per-episode summary + overall cartwheel completion count.

A "cartwheel completion" requires one episode where the pelvis rolls through
|roll| > 150 degrees at some point (full inversion) and then recovers to
|roll| < 45 degrees (lands upright), without a termination in between.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--telemetry", required=True)
  args = ap.parse_args()

  d = np.load(args.telemetry)
  pelvis_pos = d["pelvis_pos"]
  pelvis_quat_wxyz = d["pelvis_quat"]
  reset = d["reset"]
  fps = int(d["fps"][0])
  n = pelvis_pos.shape[0]

  quat_xyzw = pelvis_quat_wxyz[:, [1, 2, 3, 0]]
  eulers = R.from_quat(quat_xyzw).as_euler("xyz", degrees=True)
  roll_deg = eulers[:, 0]
  pitch_deg = eulers[:, 1]

  # Segment by reset signal.
  episode_starts = [0] + [i + 1 for i, r in enumerate(reset[:-1]) if r]
  episode_ends = [i for i, r in enumerate(reset) if r] + [n - 1]
  episodes = list(zip(episode_starts, episode_ends))

  completions = 0
  print(f"[score] {n} frames at {fps} fps, {len(episodes)} episodes total")
  for idx, (s, e) in enumerate(episodes):
    if e - s < 5:
      continue
    ep_roll = np.abs(roll_deg[s:e + 1])
    ep_pitch = np.abs(pitch_deg[s:e + 1])
    peak_roll = ep_roll.max()
    peak_pitch = ep_pitch.max()
    end_roll = ep_roll[-1]
    end_pitch = ep_pitch[-1]
    z_peak = pelvis_pos[s:e + 1, 2].max()
    z_min = pelvis_pos[s:e + 1, 2].min()
    dur_s = (e - s + 1) / fps

    inverted = peak_roll > 150.0 or peak_pitch > 120.0
    recovered = end_roll < 45.0 and end_pitch < 45.0
    is_completion = inverted and recovered
    if is_completion:
      completions += 1

    tag = "✓" if is_completion else (" " if inverted else ".")
    print(
      f"[score] ep {idx:3d} {tag} "
      f"dur={dur_s:.2f}s "
      f"|roll_peak|={peak_roll:5.0f} |pitch_peak|={peak_pitch:5.0f} "
      f"z_peak={z_peak:.2f} z_min={z_min:.2f} "
      f"end_roll={end_roll:4.0f} end_pitch={end_pitch:4.0f}"
    )

  inverted_count = sum(
    1 for s, e in episodes
    if e - s > 5
    and (np.abs(roll_deg[s:e + 1]).max() > 150
         or np.abs(pitch_deg[s:e + 1]).max() > 120)
  )
  print(
    f"\n[score] SUMMARY: {completions} / {len(episodes)} episodes are full cartwheel completions "
    f"(inverted AND recovered)"
  )
  print(
    f"[score]          {inverted_count} / {len(episodes)} reached inversion at all"
  )


if __name__ == "__main__":
  main()
