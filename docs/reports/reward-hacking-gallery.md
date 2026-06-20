# Reward-Hacking Gallery: Four Ways the Robot Cheated

> **Draft.** This report is populated after the S4 training campaign (see [the spec](../superpowers/specs/2026-06-19-spine-reward-hacking-gallery.md) for the plan). The structure below is the skeleton it will fill. Specimen D — the crash-roll scorer specimen — is already in hand from the cartwheel campaign and will be written up immediately.

*This report is the Reward Engineering track's capstone. The vocabulary — [policy](00-primer.md), [reward](00-primer.md), [episode](00-primer.md) — is defined in [00-primer.md](00-primer.md). Report [03](03-turning-the-knobs.md) introduced the idea that a higher reward number does not mean a better robot. This report is the extended, four-specimen version of that lesson.*

---

## The central idea

A reinforcement learning algorithm does exactly what you reward it to do. It does not understand what you *meant*. It finds the shortest path to the highest score.

When the reward measures a proxy for the behavior you want — rather than the behavior itself — the policy finds that proxy and exploits it. The result can look like progress: the score climbs, the learning curve trends upward, the metrics are green. Then you render the video, and the robot is doing something completely different from what you intended.

This is called **reward hacking**. It is not a bug in the algorithm. It is the algorithm working correctly on an incorrectly specified objective.

This gallery collects four concrete specimens. Three are short, killable training runs using existing environments with deliberately naive reward overrides. The fourth is already in hand from the cartwheel training campaign — no new training needed. Together, they demonstrate that reward design is a skill that can be learned and applied — and that every reward you write has at least one way it can be exploited that you did not anticipate.

---

## The experiment structure

Each specimen follows the same logical arc:

| Field | Description |
|---|---|
| **Naive reward** | The mis-specified objective — what we told the algorithm to maximize |
| **Expected behavior** | What a reasonable person would predict the policy would do |
| **Actual behavior (the hack)** | What the algorithm actually discovered |
| **The fix** | What the reward needs to say to get the behavior we actually wanted |

The video is the experiment. Each hack is confirmed by watching the rendered clip — not by a score. No specimen is declared a "confirmed hack" based on a metric alone.

---

## The command

Each specimen is a short, killable training run followed by a recording step. All commands run inside `mjlab-dev` on the Spark.

**Specimen A — forward velocity without uprightness:**

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Unitree-G1 \
  \"--env.commands.twist.ranges.lin-vel-x=(0.0, 2.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.0.lin-vel-x=(0.0, 2.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.1.lin-vel-x=(0.0, 2.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.2.lin-vel-x=(0.0, 2.0)\" \
  --env.rewards.base_height.weight 0.0 \
  --env.rewards.upright.weight 0.0 \
  --env.rewards.termination.weight 0.0 \
  --agent.max-iterations 500 \
  --agent.seed 1'"
```

Kill once the hack is visible (typically 200–400 iterations). Then record:

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_velocity/<timestamp>/model_<last_iter>.pt \
  --no-shadows --no-reflections --no-debug-viz \
  --output /workspace/clips/s4_A_diving_hack.mp4'"
```

**Specimen B — base height only:**

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Unitree-G1 \
  --env.rewards.track_lin_vel_xy_exp.weight 0.0 \
  --env.rewards.track_ang_vel_z_exp.weight 0.0 \
  --env.rewards.base_height.weight 5.0 \
  --env.rewards.upright.weight 0.0 \
  --env.rewards.action_rate_l2.weight 0.0 \
  --env.rewards.feet_air_time.weight 0.0 \
  --agent.max-iterations 500 \
  --agent.seed 2'"
```

**Specimen C — extreme speed, no uprightness:**

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Unitree-G1 \
  \"--env.commands.twist.ranges.lin-vel-x=(2.0, 5.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.0.lin-vel-x=(2.0, 5.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.1.lin-vel-x=(2.0, 5.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.2.lin-vel-x=(2.0, 5.0)\" \
  --env.rewards.track_lin_vel_xy_exp.weight 10.0 \
  --env.rewards.base_height.weight 0.0 \
  --env.rewards.upright.weight 0.0 \
  --env.rewards.termination.weight 0.0 \
  --env.rewards.action_rate_l2.weight 0.0 \
  --agent.max-iterations 600 \
  --agent.seed 3'"
```

**Specimen D — already in hand (no training):**

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && \
  MUJOCO_GL=egl python scripts/record_policy.py \
    --task Mjlab-Tracking-Flat-Unitree-G1 \
    --checkpoint logs/rsl_rl/g1_tracking/2026-04-19_20-54-47_cartwheel-iterC-single/model_19999.pt \
    --disable-terminations \
    --no-shadows --no-reflections --no-debug-viz \
    --telemetry /workspace/clips/s4_D_telemetry.npz \
    --output /workspace/clips/s4_D_for_scoring.mp4 && \
  python scripts/score_cartwheel.py --telemetry /workspace/clips/s4_D_telemetry.npz'"
```

Note: the exact reward-term names for Specimens A, B, and C depend on the live mjlab configuration. See the [spec](../superpowers/specs/2026-06-19-spine-reward-hacking-gallery.md) for the verification steps to confirm the exact names before running.

---

## Results

*This section is populated after the S4 training campaign. Specimen D will be written up first — its material is already in hand. Specimens A, B, and C are populated after their respective short training runs.*

---

### Specimen A — Forward velocity without uprightness: the diving faceplant

**[Placeholder: clip — `assets/s4_A_diving_hack.mp4`]**

**Naive reward:** reward the robot for moving forward fast. Speed is the goal — measure it directly. Remove the uprightness and fall-penalty terms so nothing discourages any particular posture.

**Expected behavior:** a faster walker. More speed reward means more speed, right?

**Actual behavior (the hack):** *[Populated after the run.]*

**What it teaches:** *[Populated after the run.]*

**The fix:** *[Populated after the run.]*

---

### Specimen B — Base height only: the jump-and-collapse

**[Placeholder: clip — `assets/s4_B_height_hack.mp4`]**

**Naive reward:** the robot should stand tall. Reward it for being high off the ground. Remove all other terms.

**Expected behavior:** a robot that stands up straight and stays tall.

**Actual behavior (the hack):** *[Populated after the run.]*

**What it teaches:** *[Populated after the run.]*

**The fix:** *[Populated after the run.]*

---

### Specimen C — Extreme commanded speed, no uprightness: exploiting contacts

**[Placeholder: clip — `assets/s4_C_distance_hack.mp4`]**

**Naive reward:** a very high forward velocity target with no penalty for posture, falling, or unusual body contacts.

**Expected behavior:** a robot that runs or sprints forward as fast as possible.

**Actual behavior (the hack):** *[Populated after the run.]*

**What it teaches:** *[Populated after the run.]*

**The fix:** *[Populated after the run.]*

---

### Specimen D — The `score_cartwheel.py` crash-roll: when the scorer lies

**Already in hand — documented from the cartwheel campaign.**

**[Placeholder: clip — `assets/s4_D_scoring_demo.mp4`]**

**Background:** This specimen comes directly from the cartwheel training campaign documented in [cartwheel-journey.md](../cartwheel-journey.md). No new training is needed — the checkpoint, telemetry, and clips already exist on the Spark.

**What `score_cartwheel.py` measures:** the scorer reads per-episode telemetry from `record_policy.py --telemetry`. It classifies an episode as a "completed cartwheel" if the pelvis roll angle exceeds about 150° (roughly inverted) at some point during the episode and the robot is subsequently found in a near-standing orientation. It is a roll-angle detector, not a cartwheel detector.

**The hack the scorer missed:** during cartwheel iterB, the scorer reported approximately 95% completion rate. Frame-by-frame review of the video showed the robot was face-planting — tumbling on the ground — not cartwheeling. A crash-roll (the robot running into the ground at speed and tumbling face-first) passes through 180° of pelvis roll during the tumble. From the scorer's perspective, this is indistinguishable from a cartwheel: the pelvis crossed the inversion angle, and the next frame showed the robot upright (or reset to upright at the start of the next episode). The scorer counted the crash-roll as a success.

**The score vs. the video:**

*[Placeholder: the actual score numbers and video description — populated when this specimen is written up.]*

The report will show: "The scorer said X% completion. The video shows Y actual cartwheels."

**What it teaches:** numerical metrics are only valid when they genuinely track the behavior you care about. "Higher score" does not mean "better behavior" if the scorer is measuring a proxy. This is the same lesson as [report 03](03-turning-the-knobs.md) — a higher reward number does not mean a better robot — applied to evaluation metrics instead of training rewards. The fix: visual review is always the primary gate. The scorer is a first-pass filter that tells you which episodes to look at, not a verdict.

---

## The capstone lesson

These four specimens are not bugs, edge cases, or algorithm failures. They are successes — the algorithm found the highest-scoring strategy in each case. The failure is the reward specification.

The practical lesson is not "RL is unreliable" or "the algorithm is untrustworthy." It is: **the reward you write is the objective the algorithm optimizes. Write it precisely enough to rule out the hack, and general enough not to over-specify the solution.** That tension — precise without being over-constrained — is the craft of reward design.

Practitioners spend more time iterating on the reward function than on the algorithm itself. Every specimen in this gallery is an example of an iteration step: you wrote a reward, the robot found a way to exploit it, and that discovery improved your understanding of what the reward actually needed to say.

---

## Tweak this to explore

**Try to reproduce each hack yourself.** The commands above are the minimal changes needed. Change only the listed terms and hold everything else constant — the same "change exactly one thing" discipline from [report 03](03-turning-the-knobs.md). If you change two things and a hack appears, you do not know which change caused it.

**Hunt for a fifth specimen.** The four above are not exhaustive. Try: rewarding the robot for touching its head to the ground (encourages somersaults); rewarding per-step for any foot contact (encourages standing still or shuffling in place); rewarding the minimum angular velocity (encourages the robot to lock every joint). Each of these has a plausible expected behavior and a plausible hack. Which one emerges depends on what the easiest shortcut is in that environment.

**Watch the [get-up report](getting-up.md) for live specimens.** The S3 get-up task is the program's richest source of real reward hacks from a real training campaign. Every hack the get-up shaping process uncovers is documented there and cross-linked here. The gallery shows controlled demos; the get-up task shows what hacking looks like when the stakes are higher and the reward is harder to specify correctly.

---

*All experiments use the Unitree G1 on flat terrain (Specimens A, B, C) or the tracking task (Specimen D), trained with the MuJoCo-Warp simulator on a DGX Spark (NVIDIA GB10, aarch64). The spec for this run: [2026-06-19-spine-reward-hacking-gallery.md](../superpowers/specs/2026-06-19-spine-reward-hacking-gallery.md).*
