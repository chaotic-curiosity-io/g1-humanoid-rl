"""Side-by-side visualization of multiple training checkpoints.

Loads N checkpoints from a single training run, creates one env with
N * ENVS_PER_CKPT parallel robots, and routes each group of ENVS_PER_CKPT
robots to the policy of one checkpoint. Launches the Viser viewer so you
can see random-weight robots, mid-training robots, and fully-trained
robots all simulating in the same 3D scene at once.

Also starts a server-side orbit camera that rotates around the scene at
a configurable rate — useful for the "cinematic" demo view.

Usage (inside the container):
    cd /workspace
    MUJOCO_GL=egl python scripts/watch_learning.py

Then open http://localhost:8081 in a browser. Robots are arranged in a
grid; the mapping from grid position to checkpoint is printed in the
terminal at startup.

Tuning knobs in this file:
    ENVS_PER_CKPT    — envs per checkpoint (1024 total at 128 is a good sweet spot)
    FRAME_RATE       — viewer tick rate (lower = less websocket traffic = smoother browser)
    ORBIT_ENABLED    — turn off if you want free camera control
    ORBIT_*          — center, radius, elevation, angular speed
"""

import math
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import viser

import mjlab.tasks  # noqa: F401 - registers tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer


TASK_ID = "Mjlab-Velocity-Flat-Unitree-G1"
CKPT_DIR = Path("/workspace/mjlab/logs/rsl_rl/g1_velocity/2026-04-17_18-46-23")

# Diverse slice of training. Left-to-right = untrained → fully trained.
CHECKPOINTS = [
    "model_0.pt",
    "model_50.pt",
    "model_200.pt",
    "model_500.pt",
    "model_800.pt",
    "model_1200.pt",
    "model_1600.pt",
    "model_2050.pt",
]
ENVS_PER_CKPT = 128  # 8 × 128 = 1024 robots total (halved from 2048 for smoother rendering)

# Lower tick rate = fewer state updates sent over websocket per second.
FRAME_RATE = 30.0

# Orbit cam. Off by default — user can flip the GUI toggle to enable.
# The orbit logic PRESERVES the current camera radius and elevation, so
# you can freely zoom/tilt with the mouse while orbiting.
ORBIT_DEFAULT_ON = False
ORBIT_CENTER = (0.0, 0.0, 0.9)        # look at torso height
ORBIT_DEFAULT_DEG_PER_SEC = 6.0        # one full revolution every 60 s (GUI slider overrides)
ORBIT_UPDATE_HZ = 20.0                 # how often to nudge the camera (lower = less jitter)


def _start_orbit_for_client(
    client: viser.ClientHandle, state: dict
) -> None:
    """Smooth server-driven orbit with user-override detection.

    Maintains (azimuth, radius, elevation) server-side and advances only
    azimuth per tick. On each tick we compare the client's current camera
    position to what we last wrote — if they differ by more than
    RESYNC_THRESHOLD meters, we assume the user moved the camera (zoom,
    tilt, pan) and resync (radius, elevation) from their input. This
    avoids the read→rotate→write feedback loop that causes jitter, and
    still lets you zoom mid-orbit.
    """
    RESYNC_THRESHOLD = 0.25  # meters of drift that count as user input

    def loop() -> None:
        last_t = time.time()
        center = np.array(ORBIT_CENTER)
        az = r = el = None
        last_written = None
        while True:
            now = time.time()
            dt = now - last_t
            last_t = now

            if not state["enabled"]:
                # Idle: drop cached orbit state so next enable re-reads
                # from whatever the user has since set up.
                az = r = el = None
                last_written = None
                time.sleep(1.0 / ORBIT_UPDATE_HZ)
                continue

            # Read current camera — either for first-sync or drift detection.
            try:
                current = np.array(client.camera.position, dtype=float)
            except Exception:
                return

            need_resync = (
                az is None
                or last_written is None
                or float(np.linalg.norm(current - last_written)) > RESYNC_THRESHOLD
            )
            if need_resync:
                offset = current - center
                r_new = float(np.linalg.norm(offset))
                if r_new < 1e-3:
                    r_new = 1.0
                r = r_new
                el = math.asin(max(-1.0, min(1.0, offset[2] / r)))
                az = math.atan2(offset[1], offset[0])

            # Advance azimuth only.
            az += math.radians(state["deg_per_sec"] * dt)

            new_pos = np.array([
                center[0] + r * math.cos(el) * math.cos(az),
                center[1] + r * math.cos(el) * math.sin(az),
                center[2] + r * math.sin(el),
            ])
            try:
                client.camera.position = tuple(new_pos.tolist())
                client.camera.look_at = tuple(center.tolist())
            except Exception:
                return
            last_written = new_pos

            time.sleep(1.0 / ORBIT_UPDATE_HZ)

    threading.Thread(target=loop, daemon=True).start()


def main() -> None:
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = len(CHECKPOINTS) * ENVS_PER_CKPT

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner

    policies = []
    for ckpt_name in CHECKPOINTS:
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        ckpt_path = CKPT_DIR / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        runner.load(
            str(ckpt_path),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        policies.append(runner.get_inference_policy(device=device))

    print()
    print("=" * 72)
    print(
        f"Env index → checkpoint mapping "
        f"(total {env_cfg.scene.num_envs} envs, {ENVS_PER_CKPT} per checkpoint)"
    )
    print("=" * 72)
    for i, name in enumerate(CHECKPOINTS):
        start = i * ENVS_PER_CKPT
        end = (i + 1) * ENVS_PER_CKPT - 1
        print(f"  envs[{start:>4}..{end:>4}]  ->  {name}")
    print("=" * 72)
    print(
        "Orbit cam: toggle in the Viser GUI (left panel → 'Orbit camera'). "
        "Off by default; zoom/elevation with mouse are preserved when on."
    )
    print()

    def compound_policy(obs: torch.Tensor) -> torch.Tensor:
        chunks = []
        for i, policy in enumerate(policies):
            start = i * ENVS_PER_CKPT
            end = (i + 1) * ENVS_PER_CKPT
            chunks.append(policy(obs[start:end]))
        return torch.cat(chunks, dim=0)

    # Pre-create the Viser server so we can attach orbit GUI + handler before
    # mjlab's viewer takes over.
    server = viser.ViserServer(label="mjlab")

    # Shared orbit state driven by the GUI controls below.
    orbit_state: dict = {
        "enabled": ORBIT_DEFAULT_ON,
        "deg_per_sec": ORBIT_DEFAULT_DEG_PER_SEC,
    }

    with server.gui.add_folder("Orbit camera"):
        cb = server.gui.add_checkbox("Orbit", initial_value=ORBIT_DEFAULT_ON)
        speed = server.gui.add_slider(
            "Speed (deg/s)", min=0.0, max=45.0, step=0.5,
            initial_value=ORBIT_DEFAULT_DEG_PER_SEC,
        )

        @cb.on_update
        def _(_) -> None:
            orbit_state["enabled"] = cb.value

        @speed.on_update
        def _(_) -> None:
            orbit_state["deg_per_sec"] = speed.value

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        _start_orbit_for_client(client, orbit_state)

    viewer = ViserPlayViewer(
        env,
        compound_policy,
        frame_rate=FRAME_RATE,
        viser_server=server,
        checkpoint_manager=None,
    )
    viewer.run()
    env.close()


if __name__ == "__main__":
    main()
