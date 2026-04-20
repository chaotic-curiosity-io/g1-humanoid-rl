"""Convert MimicKit's smpl_backflip.pkl to a GMR-format G1 motion pkl.

Bypasses SMPL-X body model weights: we use MimicKit's shipped smpl.xml MJCF as
the authoritative skeleton (the motion was authored against it), do numpy FK on
the axis-angle pose stream, then call GMR's retargeter programmatically to
produce a G1-compatible pkl. That pkl feeds the already-proven pipeline:
batch_gmr_pkl_to_csv -> mjlab csv_to_npz -> training.
"""

from __future__ import annotations

import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR


# --- 1. Parse MimicKit's SMPL humanoid MJCF into a flat skeleton table. ---

def parse_mjcf_skeleton(xml_path: Path):
  tree = ET.parse(xml_path)
  root = tree.getroot()
  bodies: list[tuple[str, int, np.ndarray]] = []

  def walk(body_el, parent_idx: int) -> None:
    name = body_el.get("name")
    pos = np.array(
      [float(x) for x in body_el.get("pos", "0 0 0").split()],
      dtype=np.float64,
    )
    idx = len(bodies)
    bodies.append((name, parent_idx, pos))
    for child in body_el:
      if child.tag == "body":
        walk(child, idx)

  for wb in root.findall("worldbody"):
    for body in wb.findall("body"):
      walk(body, -1)
  return bodies


# --- 2. Forward kinematics. ---

def forward_kinematics(
  bodies: list[tuple[str, int, np.ndarray]],
  root_pos: np.ndarray,
  root_rotvec: np.ndarray,
  joint_rotvecs: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
  """Compute world (pos, quat_wxyz) for each body.

  The MimicKit motion format encodes each of the 23 non-root bodies as a 3D
  axis-angle ("exp-map") rotation in MJCF body order. Joint i's local rotation
  is applied in the parent's frame.
  """
  n = len(bodies)
  world_pos = np.zeros((n, 3))
  world_rot: list[R] = [None] * n  # type: ignore[list-item]

  local_rots: list[R] = [None] * n  # type: ignore[list-item]
  local_rots[0] = R.from_rotvec(root_rotvec)
  for i in range(1, n):
    local_rots[i] = R.from_rotvec(joint_rotvecs[i - 1])

  for i, (_, parent_idx, offset) in enumerate(bodies):
    if parent_idx == -1:
      world_pos[i] = root_pos
      world_rot[i] = local_rots[i]
    else:
      p = parent_idx
      world_pos[i] = world_pos[p] + world_rot[p].apply(offset)
      world_rot[i] = world_rot[p] * local_rots[i]

  return {
    bodies[i][0]: (world_pos[i].copy(), world_rot[i].as_quat(scalar_first=True))
    for i in range(n)
  }


# --- 3. SMPL-X joint name -> MimicKit MJCF body name. ---

SMPLX_TO_MIMICKIT = {
  "pelvis": "Pelvis",
  "spine3": "Chest",
  "left_hip": "L_Hip",
  "right_hip": "R_Hip",
  "left_knee": "L_Knee",
  "right_knee": "R_Knee",
  # SMPL-X "left_foot"/"right_foot" = ball-of-foot (matches MimicKit L_Toe/R_Toe).
  "left_foot": "L_Toe",
  "right_foot": "R_Toe",
  "left_shoulder": "L_Shoulder",
  "right_shoulder": "R_Shoulder",
  "left_elbow": "L_Elbow",
  "right_elbow": "R_Elbow",
  "left_wrist": "L_Wrist",
  "right_wrist": "R_Wrist",
}


# --- 4. MimicKit pkl loader (tolerates missing classes). ---

class _FlexibleClass:
  def __init__(self, *_, **kw):
    for k, v in kw.items():
      setattr(self, k, v)


class _RobustUnpickler(pickle.Unpickler):
  def find_class(self, module, name):
    try:
      return super().find_class(module, name)
    except (ImportError, AttributeError):
      return type(name, (_FlexibleClass,), {})


def load_mimickit_motion(path: Path) -> tuple[np.ndarray, int]:
  with open(path, "rb") as f:
    data = _RobustUnpickler(f).load()
  if isinstance(data, dict):
    frames = np.asarray(data["frames"], dtype=np.float64)
    fps = int(data["fps"])
  else:
    frames = np.asarray(data.frames, dtype=np.float64)
    fps = int(data.fps)
  return frames, fps


# --- 5. Main driver. ---

def convert(smpl_pkl: Path, smpl_xml: Path, out_pkl: Path) -> None:
  bodies = parse_mjcf_skeleton(smpl_xml)
  assert len(bodies) == 24, f"expected 24 SMPL bodies, got {len(bodies)}"

  frames, src_fps = load_mimickit_motion(smpl_pkl)
  assert frames.shape[1] == 75, (
    f"expected 75 cols per frame ([pos 3] + [root axis-angle 3] + "
    f"[joint axis-angles 69]), got {frames.shape[1]}"
  )
  n_frames = frames.shape[0]
  print(f"loaded {n_frames} frames @ {src_fps} fps from {smpl_pkl.name}")

  gmr = GMR(
    src_human="smplx",
    tgt_robot="unitree_g1",
    actual_human_height=1.75,
    verbose=False,
  )

  out_root_pos: list[np.ndarray] = []
  out_root_rot: list[np.ndarray] = []
  out_dof_pos: list[np.ndarray] = []

  for i in tqdm(range(n_frames), desc="retargeting"):
    frame = frames[i]
    root_pos = frame[0:3]
    root_rotvec = frame[3:6]
    joint_rotvecs = frame[6:].reshape(23, 3)

    world = forward_kinematics(bodies, root_pos, root_rotvec, joint_rotvecs)
    human_data = {
      smplx_name: world[mimickit_name]
      for smplx_name, mimickit_name in SMPLX_TO_MIMICKIT.items()
    }
    qpos = gmr.retarget(human_data, offset_to_ground=(i == 0))
    out_root_pos.append(qpos[0:3].copy())
    out_root_rot.append(qpos[3:7].copy())  # MuJoCo wxyz
    out_dof_pos.append(qpos[7:].copy())

  root_rot_wxyz = np.array(out_root_rot)
  # GMR pkl convention: root_rot is xyzw.
  root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]

  out = {
    "fps": src_fps,
    "root_pos": np.array(out_root_pos),
    "root_rot": root_rot_xyzw,
    "dof_pos": np.array(out_dof_pos),
    "local_body_pos": None,
    "link_body_list": None,
  }
  out_pkl.parent.mkdir(parents=True, exist_ok=True)
  with open(out_pkl, "wb") as f:
    pickle.dump(out, f)
  print(f"saved: {out_pkl}")


if __name__ == "__main__":
  root = Path("/home/chaotic-curiosity/robotic-simulation/pose-pipeline")
  convert(
    smpl_pkl=root / "assets/mimickit/MimicKit_Data/motions/smpl/smpl_backflip.pkl",
    smpl_xml=root / "assets/mimickit/MimicKit_Data/assets/smpl/smpl.xml",
    out_pkl=root / "outputs/gmr_pkl/smpl_backflip_to_g1.pkl",
  )
