# g1-humanoid-rl

Reinforcement-learning experiments teaching a Unitree G1 humanoid whole-body skills in simulation. Built on [mjlab](https://github.com/mujocolab/mjlab) (Isaac-Lab-style API on MuJoCo-Warp), running on an NVIDIA DGX Spark (aarch64, CUDA 13, sm_121 Blackwell).

*A [Chaotic Curiosity](https://chaoticcuriosity.io) project. A hands-on, zero-background walkthrough of the walking policy is published at [chaotic-curiosity-io.github.io/g1-humanoid-rl](https://chaotic-curiosity-io.github.io/g1-humanoid-rl/).*

Two trained skills so far:

1. **Velocity tracking (walking)** — stock `Mjlab-Velocity-Flat-Unitree-G1` PPO policy, trained from scratch in ~46 min at 2048 envs. Reward 50.5, ep length 995/1000.
2. **Cartwheel (tracking)** — custom pipeline: MimicKit G1 reference → mjlab tracking task → 20k-iter PPO policy that performs repeated full cartwheels with proper inversions and two-footed landings.

The writeup of how the cartwheel was produced — including the failure modes, the bugs in the evaluation, and the final fix — is in [`docs/cartwheel-journey.md`](docs/cartwheel-journey.md) and is written for a non-technical reader.

## Video results

[Videos folder (Google Drive)](https://drive.google.com/drive/folders/16wbpiDciA7adQpgEWA_arBY-FpS41w_N)

| # | What you'll see | Link |
|---|-----------------|------|
| 01 | **Headline cartwheel result.** 40 s, 4 camera angles (chase/side/front/top), 1080p. Policy running continuously (terminations off) so multiple real cartwheels are visible back-to-back. | [01_cartwheel_final.mp4](https://drive.google.com/file/d/1X4G6ytX5fF0simB-_ZoElN8AAi3phwEq/view) |
| 02 | Same run, side view only — cleanest view of the pitch axis. | [02_cartwheel_final_side.mp4](https://drive.google.com/file/d/1rotCLFdgHB6_LUiTWx8o7i9zHBboo5wb/view) |
| 03 | Mid-training snapshot at iter 10 000 (policy about halfway to final). | [03_cartwheel_mid_10k.mp4](https://drive.google.com/file/d/1vOS8_KwyG6mT1F8RAB4B8m0FxuN1NejM/view) |
| 04 | Earlier snapshot at iter 4 500 — the policy was already completing cartwheels here. | [04_cartwheel_mid_4500.mp4](https://drive.google.com/file/d/1L5IQ4g37vXRuAO22yNiUGP-amPEHC2YW/view) |
| 05 | **Failed iteration B**, shown for contrast. Looser thresholds + warm-start hit a bad local optimum where the policy flops instead of cartwheeling. Illustrates the scorer bug. | [05_iterB_fail_for_contrast.mp4](https://drive.google.com/file/d/1wg8L8X5Cog5CxOoLFyEJswWKSU--ARMD/view) |
| 06 | Velocity-policy learning progression — 22 checkpoints from iter 0 to 2050, same robot seen getting progressively better at walking in a 26 s video. | [06_velocity_policy_progression.mp4](https://drive.google.com/file/d/1dJMfoeCxdqBSoKXlsX7VXq7nAHLZhdPU/view) |
| 07 | OmniXtreme G1 backflip reference motion replayed (no policy). Illustrates why SMPL-X → humanoid retargeting quality degrades during aerial phases. | [07_backflip_omnixtreme_ref.mp4](https://drive.google.com/file/d/1yfqD2SRSo6xqoPHbdBTCNaZl3Vl09bbm/view) |

## Repository contents

This repo holds only the small, source-of-truth artifacts. Everything heavy (mjlab upstream, training checkpoints, cloned retargeting libraries, downloaded datasets, W&B and Warp caches) is reproducible from the commands in [`setup-notes.md`](setup-notes.md).

```
docs/
  cartwheel-journey.md       — iteration-by-iteration log of the cartwheel training, written for a non-technical reader
  session-01-writeup.md      — end-to-end walkthrough of setting up mjlab on the DGX Spark and training the baseline walking policy
  dgx-spark-manual.md        — local copy of the DGX Spark setup manual (April 2026)
  reports/                   — 4-part beginner-friendly learning series on walking RL (start at docs/reports/README.md)
scripts/
  record_policy.py           — headless multi-camera renderer for a trained tracking/velocity policy; supports --disable-terminations and telemetry dump
  play_motion_npz.py         — headless replay of a tracking motion.npz (reference motion, no policy)
  score_cartwheel.py         — quantitative scorer consuming telemetry from record_policy.py (note: roll-angle-only, known to be spoofable by crashes — see journal)
  smpl_backflip_to_g1.py     — custom SMPL→G1 retargeter that bypasses SMPL-X body-model weights by using MimicKit's shipped smpl.xml MJCF for forward kinematics; plugs directly into GMR's IK
  record_learning_progression.py — multi-checkpoint, multi-camera progression recorder (stitches one MP4 per angle + optional 2×2 grid from N evenly-spaced checkpoints)
  watch_learning.py          — host-side helper that loads 8 checkpoints into one 1024-env Viser scene, routing each 128-env block to a different policy
  plot_training_curves.py    — tensorboard training scalars → CSV + curve PNGs; overlays multiple runs for comparison (used by the docs/reports series)
setup-notes.md               — append-as-we-go log of actual setup deviations per session
.gitignore                   — keeps secrets, checkpoints, caches, and downloaded datasets out of the repo
```

## Reproducing

1. **Host setup on a DGX Spark** (aarch64, CUDA 13, sm_121). Ubuntu 24.04 base. Follow [`docs/session-01-writeup.md`](docs/session-01-writeup.md) Phase A and B to get the `mjlab-dev` NGC PyTorch container running with the right bind-mounts. Total env-setup time ~30 min if you don't hit surprises. If you do, [`setup-notes.md`](setup-notes.md) has every deviation we encountered the first time through.
2. **Pull the upstream projects** (not checked into this repo):
   - `mjlab` → `~/robotic-simulation/mjlab/` (editable pip install inside `mjlab-dev`)
   - `MimicKit` → any location; download its asset pack from the SharePoint link in the MimicKit README
   - `GMR` → cloned but only needed for the retargeting workflow, not for training
   - `g1_spinkick_example` → provides the MimicKit `.pkl` → mjlab `.csv` converter
3. **Train the walking policy**: see the exact command in [`docs/session-01-writeup.md`](docs/session-01-writeup.md) Phase E. ~45 min at 2048 envs.
4. **Train the cartwheel policy**: pipeline is documented step-by-step in [`docs/cartwheel-journey.md`](docs/cartwheel-journey.md). Short version:
   ```sh
   # 1. Convert MimicKit cartwheel pkl to a single-cartwheel CSV.
   python repos/g1_spinkick_example/pkl_to_csv.py \
     --pkl-file <MimicKit>/motions/g1/g1_cartwheel.pkl \
     --csv-file motions/g1_cartwheel_single.csv \
     --add-start-transition --add-end-transition \
     --transition-duration 0.5 --pad-duration 0.5
   # 2. CSV -> mjlab motion.npz (inside the mjlab-dev container).
   MUJOCO_GL=egl python -m mjlab.scripts.csv_to_npz \
     --input-file motions/g1_cartwheel_single.csv \
     --output-name mimickit_cartwheel_single \
     --input-fps 60 --output-fps 50 --render False
   # 3. Train (inside the mjlab-dev container). ~11 h on a DGX Spark.
   MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python -m mjlab.scripts.train \
     Mjlab-Tracking-Flat-Unitree-G1 \
     --env.commands.motion.motion-file motions/mimickit_cartwheel_single.npz \
     --env.scene.num-envs 4096 \
     --agent.max-iterations 20000 \
     --agent.run-name cartwheel-iterC-single \
     --env.terminations.anchor-pos.params.threshold 0.5 \
     --env.terminations.ee-body-pos.params.threshold 0.5
   # 4. Render the trained policy (from outside the container, GPU).
   MUJOCO_GL=egl python scripts/record_policy.py \
     --checkpoint-file logs/rsl_rl/g1_tracking/<run>/model_19999.pt \
     --motion-file motions/mimickit_cartwheel_single.npz \
     --output-dir out/cartwheel_final \
     --num-steps 2000 --width 1920 --height 1080 \
     --device cuda:0 --show-ghost 0 --disable-terminations 1
   ```

## Key lessons from the cartwheel run

- **Tracking task terminations are a two-edged knife.** The stock 0.25 m thresholds are too tight for aerial motions; the policy never sees a completed flip. Loosening to 0.5 m works, but only if you don't then evaluate at 0.25 m.
- **Render must match training on thresholds.** Our first "success" video was an artifact of the eval having stricter thresholds than training — every attempt got cut off mid-flip.
- **Quantitative scorers lie.** A pelvis-roll-through-180° scorer is fooled by crash-rolls where the robot is face-planting. Always pair auto-scoring with frame-by-frame visual inspection.
- **SMPL-X → humanoid retargeting degrades in aerial phases.** The reference motion itself is only as good as the ground-contact constraint. For acrobatic motions, hand-curated retargeting (e.g., MimicKit's shipped G1 set) beats any automatic retargeter I tried.
- **Reference-motion duration matters.** We accidentally asked the policy to learn two cartwheels back-to-back when the tooling cycled a 2.73 s source up to a 4 s target duration. Making the reference shorter than the policy's reach dramatically unblocked training.

## Credit

- mjlab — [mujocolab/mjlab](https://github.com/mujocolab/mjlab)
- Reference motions — [xbpeng/MimicKit](https://github.com/xbpeng/MimicKit) (Jason Peng)
- Retargeting tooling — [YanjieZe/GMR](https://github.com/YanjieZe/GMR), [mujocolab/g1_spinkick_example](https://github.com/mujocolab/g1_spinkick_example)
- OmniXtreme reference data — [Perkins729/OmniXtreme](https://github.com/Perkins729/OmniXtreme)
