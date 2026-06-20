# Spec: S4 — Reward-Hacking Gallery (Reward Engineering)

**Date:** 2026-06-19
**Branch:** `g1-skills-curriculum`
**Status:** Ready for training campaign
**Track:** Reward engineering — cross-cutting capstone (spine task 4 of 4)
**Feeds into:** `docs/reports/reward-hacking-gallery.md`; linked from the syllabus `docs/reports/README.md`

---

## Goal

Deliberately break the robot — four times, in four different ways — to teach the single most important lesson in reinforcement learning: **reward design is the job**.

A reinforcement learning algorithm does exactly what you reward it to do. It does not understand what you *meant*; it finds the shortest path to the highest score. When the reward is even slightly mis-specified — when it measures a proxy for the behavior you want rather than the behavior itself — the policy finds that proxy and exploits it. The result can look like progress at first: the score climbs, the loss falls, the metrics are green. Then you render the video, and the robot is doing something completely different from what you intended.

This phenomenon has a name: **reward hacking**. It is not a bug in the algorithm. It is the algorithm working correctly on an incorrectly specified objective.

This gallery collects four concrete specimens — each a short, killable experiment that induces a visible hack and then shows the fix. Three are fresh runs using existing environments with naive reward overrides; one is already in hand from the cartwheel training campaign. Together they form the capstone of the reward-engineering track: a hands-on demonstration that the craft of RL is not writing the training loop, but writing the reward.

**This is placed second in the execution order** (after S1 running, before S3 get-up and S2 backflip) because:

1. It is cheap — most specimens are killable as soon as the hack is visible, often within 15–30 minutes.
2. It builds directly on S1's infrastructure: the same velocity env, same `record_policy.py` workflow.
3. The S3 get-up task is the program's richest source of new reward hacks; having seen the gallery first makes those iterations legible when they happen.

---

## Audience & voice

This report is written for someone with no background in robotics or machine learning. Every term is defined on first use.

- **Policy:** the neural network that controls the robot — its "brain." Every 20 ms it reads sensor data (joint angles, body velocity, orientation) and outputs motor commands.
- **Reward:** a number the training algorithm uses as feedback. Higher reward means the robot did something we wanted — or, in a hacked policy, something we *said* we wanted without actually meaning it. The policy learns to maximize its total reward over the course of an episode.
- **Episode:** one uninterrupted simulation run. The robot starts standing, the policy runs, and eventually the simulation resets (because the robot fell, hit a time limit, or completed a trajectory).
- **Reward hacking:** when a policy finds a way to maximize its numerical score without doing the thing the designer actually intended. The algorithm is not "cheating" — it is doing its job exactly right, on a reward that was subtly wrong.
- **Weight / reward weight:** every reward term has a multiplier — its weight — that controls how much it contributes to the total score. Cranking one weight up while leaving others unchanged changes what the policy cares about most.
- **Uprightness:** a measure of how vertical the robot's torso is. When uprightness is high, the robot is standing tall. When it is low, the robot is leaning, crouching, or face-planted.
- **Base height:** the height of the robot's pelvis above the ground. High when standing, low when crouched or fallen.
- **Base velocity / forward velocity:** how fast the robot's center of mass is moving in the forward direction. A robot mid-dive has very high forward velocity for a brief moment — even though it is about to crash.
- **Contacts:** points where the robot's body is touching the ground. Hands, knees, elbows, and even the head can all make ground contact; most of these are not intended foot placements.

Concrete analogies are preferred over mathematical notation. Each report is standalone: a reader who picks it up without reading the rest of the series can follow the reasoning.

---

## Staged-arc table

Each specimen is a separate mini-arc. They share the same infrastructure (same `mjlab-dev` container, same `record_policy.py` workflow) and can be run in any order. All are short and killable: the training run is stopped as soon as the hack is visibly reproducible.

| Stage | What runs on the Spark | Estimated cost | Output |
|---|---|---|---|
| **Specimen A — induce** | Short `Mjlab-Velocity-Flat-Unitree-G1` run with uprightness weight zeroed out; kill once the diving/lunging gait is visible (typically within 200–400 iterations) | ~5–15 min GPU | Checkpoint showing the diving hack |
| **Specimen A — capture** | `record_policy.py` on the hacked checkpoint | ~5 min CPU | MP4 clip showing the robot lunging/diving |
| **Specimen B — induce** | Short run with only a `base_height` reward (all other terms zeroed); kill once jump-and-collapse behavior is reproducible | ~10–20 min GPU | Checkpoint showing the height hack |
| **Specimen B — capture** | `record_policy.py` on the hacked checkpoint | ~5 min CPU | MP4 clip showing repeated jumping and collapsing |
| **Specimen C — induce** | Short run rewarding cumulative displacement from start position; kill once launched/contact-exploit behavior is visible | ~10–30 min GPU | Checkpoint showing the distance hack |
| **Specimen C — capture** | `record_policy.py` on the hacked checkpoint | ~5 min CPU | MP4 clip showing the launched or contact-exploiting robot |
| **Specimen D — already in hand** | No new training; use the existing cartwheel iterB checkpoint and `score_cartwheel.py` telemetry | 0 GPU | Score output + existing clips showing crash-roll counted as "completed" |
| **Fixes — document** | For each specimen: show the fix (either the corrected CLI command or a prose description of the corrected reward term) and, where practical, a short corrective run to confirm the fix | Minutes to ~30 min GPU each | Before/after CLI commands; optional corrective clips |

The specimen runs can overlap with recording steps from other specimens since recording is CPU-runnable while a training job occupies the GPU.

---

## Concrete runs

All commands are issued from the Windows host and execute inside `mjlab-dev` on the Spark.

---

### Specimen A — Forward velocity without uprightness: the diving faceplant

**The naive idea:** reward the robot directly for how fast it moves forward. Speed is what we want — why not measure it directly?

**The naive reward (CLI override):**

Zero out the uprightness/fall-penalty terms while keeping the velocity-tracking reward. The exact weight to zero depends on the live mjlab reward config (see Open questions), but the pattern is:

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

Kill the run once the reward curve flattens or the hack is confirmed:

```bash
ssh spark "docker exec mjlab-dev bash -lc \"kill -INT \$(pgrep -f mjlab.scripts.train)\""
```

**Capture the hack:**

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_velocity/<timestamp>/model_<last_iter>.pt \
  --no-shadows --no-reflections --no-debug-viz \
  --output /workspace/clips/s4_A_diving_hack.mp4'"
```

**What a beginner would expect:** a faster walker. More speed reward means the robot walks faster. Simple.

**What actually happens:** the robot discovers that the fastest way to move its center of mass forward is not to walk — it is to throw itself forward and fall. A standing body that pitches its torso into a forward dive accelerates the pelvis rapidly for the brief duration of the fall. This scores very high on "forward base velocity" while doing zero walking. The policy learns the dive because the dive, for a brief instant, is the fastest thing a bipedal body can do. Without uprightness or fall-penalty to discourage it, the dive is the optimal strategy.

**What it teaches:** speed alone is the wrong reward. The actual goal is *sustained* forward motion from a *standing posture*. These require explicitly separate terms.

**The fix:** re-enable the upright posture term and the fall/contact penalty alongside the velocity reward. The policy then learns that diving scores highly only momentarily — staying upright is required to keep collecting velocity reward over the rest of the episode.

---

### Specimen B — Base height only: the jump-and-collapse

**The naive idea:** the robot should stand tall. Reward it for being high off the ground.

**The naive reward (CLI override):**

Zero out the velocity tracking and other terms; set only the base-height reward to a high weight.

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

Kill once the behavior is reproducible:

```bash
ssh spark "docker exec mjlab-dev bash -lc \"kill -INT \$(pgrep -f mjlab.scripts.train)\""
```

**Capture the hack:**

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_velocity/<timestamp>/model_<last_iter>.pt \
  --no-shadows --no-reflections --no-debug-viz \
  --output /workspace/clips/s4_B_height_hack.mp4'"
```

**What a beginner would expect:** a robot that stands up straight and stays tall.

**What actually happens:** the policy discovers that the highest base height it can achieve — even briefly — is by jumping. A jump produces a large instantaneous reward from the height term. But a jump is followed by a fall, which is followed by a reset, which is followed by another jump. The policy converges on a frantic cycle of explosive upward launches and immediate collapses, each cycle collecting one spike of height reward before the robot hits the ground and the episode resets. There is no walking, no balance, and no sustained posture — just repeated vertical launches.

**What it teaches:** height at a single moment is the wrong reward. The goal is *sustained* height over time while remaining stable. Rewarding the instantaneous peak incentivizes finding the single highest moment, not maintaining a useful posture. The fix requires a stability term — a bonus for spending time at a target height, not just touching it once — plus a penalty for erratic joint commands.

**The fix:** add a stability/duration term alongside the height reward. A robot that maintains height for several consecutive timesteps should score more than one that spikes and collapses. Pair it with an `action_rate` smoothness penalty to discourage the explosive launch strategy.

---

### Specimen C — Distance from start: getting launched / exploiting contacts

**The naive idea:** reward the robot for how far it has traveled from its starting position. Distance traveled equals progress.

**The naive reward (CLI override):**

This specimen requires a small reward-term addition because the stock velocity env does not directly expose cumulative displacement as a reward. The cleanest induction is to raise the velocity-tracking reward to a high weight alongside a very high commanded speed and disable the uprightness term — so the policy finds that the fastest way to accumulate displacement is to exploit ground contacts (sliding, rolling, being propelled by a ground-reaction force) rather than walking:

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

Kill once the behavior is reproducible:

```bash
ssh spark "docker exec mjlab-dev bash -lc \"kill -INT \$(pgrep -f mjlab.scripts.train)\""
```

**Capture the hack:**

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && MUJOCO_GL=egl python scripts/record_policy.py \
  --task Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_velocity/<timestamp>/model_<last_iter>.pt \
  --no-shadows --no-reflections --no-debug-viz \
  --output /workspace/clips/s4_C_distance_hack.mp4'"
```

**What a beginner would expect:** a robot that walks or runs forward as fast as possible.

**What actually happens:** at very high commanded speeds with no uprightness penalty, the policy discovers that the fastest forward displacement comes not from leg-driven locomotion but from exploiting ground contacts in unexpected ways — falling forward and using the impact force to propel the body, spinning so that ground friction acts as a thruster, or repeatedly rolling and using angular momentum. In extreme cases the robot appears to be "launched" across the arena by a contact force it has learned to create deliberately. None of these strategies resemble walking; all of them produce high forward velocity for at least part of the episode.

**What it teaches:** commanding a target velocity the robot cannot realistically achieve with a correct gait creates an incentive to find any physical mechanism that produces that speed, including highly unnatural ones. Rewards that measure "how much" without constraining "how" will find the fastest "how" — which is often not the one you wanted. The fix includes: a realistic speed ceiling; uprightness and contact-penalty terms that cost the policy for unintended body contacts (hands, knees, head on the ground); and a fall-termination penalty so crashes are not profitable.

---

### Specimen D — The `score_cartwheel.py` crash-roll: a scorer fooled by physics

**Already in hand — no new training required.**

This specimen comes directly from the cartwheel training campaign documented in `docs/cartwheel-journey.md`. It requires no new GPU time: the checkpoint, telemetry, and clips already exist on the Spark.

**Background:**

During the cartwheel training campaign, `score_cartwheel.py` reported a high completion rate — roughly 95 % of episodes classified as "successful cartwheels." On inspection of the rendered video, the vast majority of those "completions" were not cartwheels at all.

**What `score_cartwheel.py` actually measures:**

The scorer reads per-episode telemetry produced by `record_policy.py --telemetry`. It classifies an episode as a "completed cartwheel" if the robot's pelvis roll angle exceeds a threshold (|roll| > 150°, roughly inverted) at some point during the episode and the robot is subsequently found in a near-standing orientation. It is, in other words, a roll-angle detector — not a cartwheel detector.

**The hack the scorer missed:**

A robot that runs into the ground at speed, tumbles face-first, and rolls along the ground will also pass through 180° of roll during the tumble. From the scorer's perspective, this is indistinguishable from a cartwheel: the pelvis crossed the inversion angle, and the robot ended up upright (or reset to upright). The scorer counted the crash-roll as a success. The video showed something very different.

**Running the scorer on the existing telemetry:**

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

To demonstrate the spoofing effect, also run the scorer on a policy that is known from visual review to produce crash-rolls (e.g. cartwheel iterB checkpoint, if still on disk):

```bash
ssh spark "docker exec mjlab-dev bash -lc 'ls logs/rsl_rl/g1_tracking/'"
```

Locate an intermediate checkpoint. Then:

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace && \
  MUJOCO_GL=egl python scripts/record_policy.py \
    --task Mjlab-Tracking-Flat-Unitree-G1 \
    --checkpoint logs/rsl_rl/g1_tracking/<iterB-timestamp>/model_<N>.pt \
    --disable-terminations \
    --no-shadows --no-reflections --no-debug-viz \
    --telemetry /workspace/clips/s4_D_iterB_telemetry.npz \
    --output /workspace/clips/s4_D_iterB_for_scoring.mp4 && \
  python scripts/score_cartwheel.py --telemetry /workspace/clips/s4_D_iterB_telemetry.npz'"
```

Pull both clips and scores to the Windows box:

```bash
scp spark:/workspace/clips/s4_D*.mp4 docs/reports/assets/
```

Compare the score output to the rendered video. The report will show: "The scorer said X % completion. The video shows Y actual cartwheels."

**What a beginner would expect:** a high scorer number means the robot is doing cartwheels.

**What actually happens:** the scorer is measuring roll angle, not cartwheels. Any motion that passes through the inversion angle — including a crash-roll on the ground — satisfies the completion criterion. A policy that face-plants and tumbles can outscore a policy that achieves a partial but genuine rotation, because the crash-roll reliably hits the 180° threshold while a partial rotation might peak at 120°. The score is not lying — it is reporting exactly what it measures. What it measures is not what you think.

**What it teaches:** numerical metrics are only valid when they genuinely track the behavior you care about. "Higher score" does not mean "better behavior" if the scorer is measuring a proxy. This is the same lesson as report [03 — Turning the Knobs](../../reports/03-turning-the-knobs.md) ("higher reward ≠ better robot") applied to evaluation metrics instead of training rewards. The fix: **visual review is always the primary gate.** The scorer is a first-pass filter that tells you which episodes to look at — not a verdict.

---

## The experiment

**Structure:** four side-by-side specimens. Each follows the same logical arc:

| Field | Description |
|---|---|
| **Naive reward** | The mis-specified reward term — what we told the algorithm to maximize |
| **Expected behavior** | What a reasonable person would predict the policy would do |
| **Actual behavior (the hack)** | What the algorithm actually discovered — the shortest path to the score |
| **The fix** | What the reward needs to say to get the behavior we actually wanted |

**No single "control" run** is needed here — the control for each specimen is common sense: what a beginner would predict. The video is the experiment. Each specimen is self-contained.

**The capstone message:** these four specimens are not bugs, edge cases, or failures of the algorithm. They are successes — the algorithm found the highest-scoring strategy. The failure is the reward specification. Getting the reward right — specific enough to rule out the hack, general enough not to over-constrain the solution — is the job. This is why practitioners spend more time iterating on the reward function than on the algorithm itself.

**A note on the relationship to S3 (get-up):** The S3 get-up task is intentionally the program's hardest reward-design challenge. The hacks that S3's shaping process uncovers are documented in `docs/reports/getting-up.md` and cross-linked here — they are real specimens from a real training campaign, where the stakes were higher than a quick killable demo. S4 is the gallery; S3 is the live experiment.

---

## Artifacts & retrieval

| Artifact | Location on Spark | Committed to repo |
|---|---|---|
| Specimen A clip (diving/lunging hack) | `/workspace/clips/s4_A_diving_hack.mp4` | `docs/reports/assets/s4_A_diving_hack.mp4` |
| Specimen B clip (jump-and-collapse hack) | `/workspace/clips/s4_B_height_hack.mp4` | `docs/reports/assets/s4_B_height_hack.mp4` |
| Specimen C clip (distance/contact hack) | `/workspace/clips/s4_C_distance_hack.mp4` | `docs/reports/assets/s4_C_distance_hack.mp4` |
| Specimen D score output + clips | `/workspace/clips/s4_D_*.mp4` | `docs/reports/assets/s4_D_scoring_demo.mp4` |
| Specimen D telemetry | `/workspace/clips/s4_D_telemetry.npz` | NOT committed — stays on Spark |
| Hacked checkpoints (A, B, C) | `logs/rsl_rl/g1_velocity/<timestamp>/model_<N>.pt` | NOT committed — stays on Spark |

MP4 files are committed under `docs/reports/assets/` only — `.gitignore` contains the deliberate exception `!docs/reports/assets/*.mp4`. Do not commit MP4s anywhere else. Do not commit checkpoints, `.npz` telemetry files, or the hacked policy weights.

Pull all clips from the Spark before committing:

```bash
scp spark:/workspace/clips/s4_*.mp4 docs/reports/assets/
```

---

## Ops & safety

These runs are short and killable. The host-quiesce bracket is still required (it prevents the swap-death-spiral hard reboot), but the exposure window is short: most specimens train for fewer than 600 iterations (roughly 10–30 minutes), and recording is CPU-runnable.

**Before each specimen training run (on the Spark):**

```sh
docker stop open-webui compose-arangodb-1 ollama-compose
# Confirm headroom: workload needs ~13 GiB; aim for ~110 GiB free
free -h
# swapoff -a and systemctl stop comfyui.service need an interactive sudo session
# if memory is tight, open an interactive session first: ssh spark then sudo swapoff -a
```

**Kill a run once the hack is visible:**

Send SIGINT directly to the Python child — the `bash -c '... > log'` wrapper does not propagate SIGINT on the first try:

```sh
docker exec mjlab-dev bash -lc "kill -INT \$(pgrep -f mjlab.scripts.train)"
```

Never use a bare `pgrep -f` or `grep` of the train command in a counting or watcher loop — it matches its own command line and loops forever or miscounts. Use the bracket trick instead: `ps -eo cmd | grep "[p]ython -u -m mjlab.scripts.train"`.

**After each specimen run (always, even on failure):**

```sh
docker start open-webui compose-arangodb-1 ollama-compose
```

**Specimen D requires no training run** — all commands are recording and scoring only. The host-quiesce bracket can be skipped for that specimen if no training job is in flight.

**Viser:** training prints a `http://localhost:8080` URL from inside the container; translate to `:8081` on the host (port 8080 is owned by open-webui when running, but open-webui is stopped during training).

**Sequence:** the three training specimens (A, B, C) are run one at a time on the single GPU. Each is bracketed independently. Specimen D can be run at any point — even while a training job occupies the GPU, since it is recording-only and the offscreen renderer runs on CPU.

---

## Success criteria

The following criteria must all be met before S4 is declared successful and `reward-hacking-gallery.md` is published.

1. **At least three reproducible, visually clear hacks.** The rendered clips for Specimens A, B, and C each unambiguously show the hack — not merely a degraded walk, but a qualitatively different, clearly unintended behavior. A viewer with no ML background should watch each clip and say "that is not what the robot was supposed to do."

2. **The crash-roll scorer specimen is included.** Specimen D must appear in the gallery — either as a clip demonstrating the spoofing or as a score-vs-video comparison — with an explicit statement that the scorer counted a crash-roll as a completion. The `score_cartwheel.py` limitation is stated as a fact, not a caveat.

3. **Each specimen documents the fix.** The report entry for each specimen must state what the reward needs to say to produce the intended behavior — not just that it hacked. The gallery's purpose is not to demonstrate that RL fails but to show that reward design is a skill that can be learned and applied.

4. **Visual review is the verdict for all specimens.** No specimen is declared "confirmed hack" based on a metric alone. Each hack is confirmed by watching the rendered clip. This is the same visual-verification gate applied throughout the program, here applied to failure modes instead of successes.

5. **Report written and published.** `docs/reports/reward-hacking-gallery.md` is populated with all four specimens (clips, expected vs. actual behavior, fixes), committed to `main`, and visible on GitHub Pages.

---

## Open questions

The following items were targeted for resolution by the read-only Spark probe in Task 12 of the implementation plan.

**Probe run: 2026-06-19 — `sed -n 1071,1470p /workspace/mjlab/logs/rsl_rl/g1_velocity/2026-04-17_18-46-23/params/env.yaml` + `cat scripts/record_policy.py` (local checkout)**

1. **Exact reward-term names for Specimens A and B — RESOLVED WITH CORRECTIONS.** The complete reward-term key list from the baseline Flat velocity task `env.yaml`:

   | YAML key | Function | Default weight |
   |---|---|---|
   | `track_linear_velocity` | `rewards.track_linear_velocity` | 2.0 |
   | `track_angular_velocity` | `rewards.track_angular_velocity` | 2.0 |
   | `upright` | `rewards.upright` | 1.0 |
   | `pose` | `rewards.variable_posture` | 1.0 |
   | `body_ang_vel` | `rewards.body_angular_velocity_penalty` | -0.05 |
   | `angular_momentum` | `rewards.angular_momentum_penalty` | -0.02 |
   | `dof_pos_limits` | `envs.mdp.rewards.joint_pos_limits` | -1.0 |
   | `action_rate_l2` | `envs.mdp.rewards.action_rate_l2` | -0.1 |
   | `air_time` | `rewards.feet_air_time` | **0.0** |
   | `foot_clearance` | `rewards.feet_clearance` | -2.0 |
   | `foot_swing_height` | `rewards.feet_swing_height` | -0.25 |
   | `foot_slip` | `rewards.feet_slip` | -0.1 |
   | `soft_landing` | `rewards.soft_landing` | -0.00001 |
   | `self_collisions` | `rewards.self_collision_cost` | -1.0 |

   **Specimen A corrections:** The command `--env.rewards.base_height.weight 0.0` and `--env.rewards.termination.weight 0.0` are **wrong** — neither `base_height` nor `termination` exists as a reward term. For the diving faceplant, zero out: `--env.rewards.upright.weight 0.0` (confirmed key) and consider also `--env.rewards.pose.weight 0.0` (the `variable_posture` term also penalizes bad posture).

   **Specimen B corrections:** There is **no `base_height` reward term.** To induce a jump-and-collapse hack, options are: (a) set `air_time` weight very high (`--env.rewards.air_time.weight 20.0`) since that term directly rewards time in the air; or (b) write a custom one-line `base_height` reward term (requires new code). The `air_time` approach is cleaner for a killable demo. Update the Specimen B command to use `--env.rewards.air_time.weight 20.0` instead.

   The correct CLI override flag for air-time is `--env.rewards.air_time.weight` (not `feet_air_time`).

2. **Contact-penalty structure for Specimen C — RESOLVED.** The baseline reward config has `self_collisions` (penalizes self-collision via a force sensor named `self_collision`) and `soft_landing` (penalizes high-velocity foot contact), but **no explicit hand/knee/head contact cost**. There is no term that directly penalizes unintended body-ground contact. This means the Specimen C induction — high commanded speed with uprightness zeroed — should cleanly produce contact exploitation, because there is nothing in the reward to penalize it. No custom reward term is needed; the existing override-based induction approach is valid.

3. **Whether Specimen D needs an intermediate checkpoint — deferred.** The probe did not list the g1_tracking run directories (not part of this probe's scope). **⚠ still unresolved:** run `ssh spark "docker exec mjlab-dev bash -lc 'ls /workspace/mjlab/logs/rsl_rl/g1_tracking/'"` to check whether iterA or iterB intermediate checkpoints are still on disk. If no intermediate checkpoint with visible crash-rolls is available, document the specimen from `cartwheel-journey.md` using the final policy's honest score as the comparison.

4. **`record_policy.py` flag names — RESOLVED WITH IMPORTANT CORRECTIONS.** The script in the local checkout (`scripts/record_policy.py`) was read directly. Key findings that affect S4 recording commands:

   - **`--disable-terminations` confirmed.** Takes an int (`1` to disable). ✓
   - **`--dump-telemetry` (not `--telemetry`).** The spec commands use `--telemetry`; the actual flag is `--dump-telemetry`.
   - **`--no-shadows`, `--no-reflections`, `--no-debug-viz` do NOT exist.** Shadows and reflections are disabled unconditionally in code; no CLI flags for these. Remove them from all recording commands.
   - **The script is tracking-only** (task hardcoded to `Mjlab-Tracking-Flat-Unitree-G1`). **It cannot be used for S4 Specimens A, B, C** which use the velocity task. A separate recording approach is needed for those specimens — either a velocity-capable version of `record_policy.py` or `mjlab.scripts.play` with output.
   - **`--output-dir` (not `--output`).** The spec commands use `--output`; the actual flag is `--output-dir`.
   - **`--checkpoint-file` (not `--checkpoint`).** Spec uses `--checkpoint`; actual flag is `--checkpoint-file`.
   - **The script is NOT on the Spark.** `/workspace/scripts/` contains only `plot_training_curves.py` and `watch_learning.py`. Copy before use: `scp scripts/record_policy.py spark:/tmp/ && ssh spark "docker cp /tmp/record_policy.py mjlab-dev:/workspace/scripts/record_policy.py"`.
   - **Specimen D can use `record_policy.py` as-is** (it is a tracking task). Specimens A, B, C cannot.
