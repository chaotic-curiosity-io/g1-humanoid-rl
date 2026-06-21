# Imitation Learning: Teaching G1 to Backflip (a partial success)

*This report is the Imitation track's spine task. It builds on [imitation-cartwheel.md](imitation-cartwheel.md), which introduces motion imitation and the cartwheel campaign's hard-won lessons. Terms like [policy](00-primer.md), [reward](00-primer.md), and [episode](00-primer.md) are in [00-primer.md](00-primer.md). Unlike the other three spine reports, this one ends on an **honest partial result** — and that, too, is a real and useful outcome.*

---

## What you are about to see

A backflip is the hardest motion in this series: it needs a genuine **airborne phase**, a full **backward rotation** through inversion, *and* a stable **landing on the feet**. In motion imitation, the policy isn't rewarded for any of those in the abstract — it's rewarded for one thing only: making the robot's body match a pre-recorded reference backflip, frame by frame. If it matches closely all the way through, a backflip emerges. If it drifts too far from the reference, the episode ends.

We ran two full training attempts (~11 hours each). The first never left the ground. The second — after a diagnosed fix — **gets airborne and rotates**, but does not yet land the flip on its feet. This report documents both, because the *diagnosis between them* is the real lesson, and because knowing how to read "genuine progress, not yet solved" is itself a skill.

---

## The setup

The retargeted reference was already on hand from earlier pipeline work — a clean, single backflip (`smpl_backflip_to_g1.npz`, 88 frames at 50 fps ≈ 1.8 s). That mattered: the cartwheel campaign's worst time-sink was a *bad* reference (an accidental double flip), so starting from a verified single, feasible reference removed the biggest risk up front.

The task is the same `Mjlab-Tracking-Flat-Unitree-G1` that produced the cartwheel, with one critical knob: the **termination thresholds** — how far the robot's body (`anchor_pos`), end-effectors (`ee_body_pos`), and orientation (`anchor_ori`) may drift from the reference before the episode is cut short.

---

## Attempt 1 — thresholds too tight: it never leaves the ground

The first run used the cartwheel's proven thresholds: `anchor_pos` and `ee_body_pos` at 0.5 m, `anchor_ori` at 0.8 rad.

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Tracking-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 --agent.max-iterations 20000 \
  --env.commands.motion.motion-file /workspace/pose-pipeline/motions/smpl_backflip_to_g1.npz \
  --env.terminations.anchor-pos.params.threshold 0.5 \
  --env.terminations.ee-body-pos.params.threshold 0.5'"
```

Rendered mid-training and again near the end (`--disable-terminations`, so nothing is cut), the result was the same: the robot **winds up and crouches — but never leaves the ground.** The green ghost is the reference flipping overhead; the white robot collapses beneath it.

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/s2_v2_still.png">
  <source src="assets/s2_v1_grounded_side.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/s2_v1_grounded_side.mp4">download the clip</a> instead.
</video>

**Why?** A backflip travels much further from the start pose than a cartwheel — straight up, then fully inverted. The instant the robot couldn't follow the reference up into the air, its body/limb error crossed the 0.5 m threshold and the **episode terminated**. The policy was punished for *trying* to leave the ground (every attempt ended immediately) and so never accumulated any reward for the airborne phase. It learned the safest thing available: wind up, stay down, track the start and end of the motion where the reference is near the floor.

This is the cartwheel lesson in its sharpest form: **a termination threshold that's too tight starves the policy of any experience of the hard phase, so it can never learn it.**

---

## Attempt 2 — loosen the thresholds: it gets airborne

The fix follows directly from the diagnosis: give episodes room to survive the high-error aerial phase. Position thresholds 0.5 → **1.0 m**, orientation 0.8 → **1.5 rad** (a flip is 360° of rotation in under a second, so even a small timing offset spikes the orientation error — that threshold had to open up too).

```bash
  --env.terminations.anchor-pos.params.threshold 1.0 \
  --env.terminations.ee-body-pos.params.threshold 1.0 \
  --env.terminations.anchor-ori.params.threshold 1.5
```

The change worked — mechanically, exactly as intended. The robot now **launches off the ground and rotates backward through the air**:

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/s2_v2_still.png">
  <source src="assets/s2_v2_attempt_side.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/s2_v2_attempt_side.mp4">download the clip</a> instead.
</video>

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/s2_v2_still.png">
  <source src="assets/s2_v2_attempt_chase.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/s2_v2_attempt_chase.mp4">download the clip</a> instead.
</video>

It gets up, tips backward into a partial inversion — and then comes down onto its back rather than completing the rotation to land on its feet. **A genuine airborne backflip *attempt*, not a landed backflip.**

### The metrics tell the same story

The two training metrics, v1 (tight) vs v2 (loose), make the diagnosis visible. First, the body-tracking error:

![anchor position error, v1 vs v2](assets/s2_anchor_err.png)

Counterintuitively, v2's error is *higher* (~0.63 vs v1's ~0.44) — and that's the good news. v1 was pinned right against its 0.5 m ceiling: it never went anywhere near the parts of the motion that produce large error. v2's error sits comfortably under its 1.0 m ceiling *because the robot is actually out in the air*, where matching the reference is genuinely hard. Higher error here means **more of the flip is being attempted**, not less skill.

The end-effector terminations confirm it:

![ee_body terminations, v1 vs v2](assets/s2_ee_term.png)

v2 (orange) terminates consistently less than v1 (blue) — ~105 vs ~175 per rollout window, and still falling at the end — because its episodes survive deeper into the flip instead of being cut at takeoff.

---

## The honest verdict

Across every render of the final v2 policy (frame-by-frame, multiple angles — the visual gate, never the score), the result is consistent: **the robot performs a genuine backward airborne rotation but does not land it on its feet.** It is a partial backflip.

That is a real, hard-won step:

| Attempt | Thresholds | Behaviour |
|---|---|---|
| v1 | 0.5 / 0.8 / 0.5 (tight) | winds up, **never leaves the ground** |
| v2 | 1.0 / 1.5 / 1.0 (loose) | **launches and rotates in the air**, lands on its back |

And it's an honest place to stop and report, for a reason worth internalising: **not every skill is solved within the training budget you give it, and recognising "genuine progress, short of the goal" is part of the craft.** The backflip is simply harder than the cartwheel (the cartwheel itself needed three reference/threshold iterations) — full inversion plus a feet-first landing is a tall order, and two ~11-hour runs got it airborne but not landed.

### What a v3 would change

The diagnosis points cleanly at the next lever. The tracking reward pays for *matching the reference pose* — but nothing **specifically rewards ending on the feet**, so a partway rotation that crashes onto the back scores nearly as well as a clean landing. A v3 would add an explicit **landing term** (reward feet-down + upright in the final phase of the motion), exactly the way the [get-up task](getting-up.md) needed an explicit "hold the stand" reward to stop settling for a crouch. It may also need more iterations, or a reference whose landing is emphasised. Each attempt is another ~11-hour run — so this is a deliberate next-session decision, not an automatic one.

---

## Tweak this to explore

**Add a landing reward.** The clearest next experiment: a reward term active in the last ~0.3 s of the motion that pays for feet-on-ground + torso-upright. This is the missing incentive — the same shaping insight that took [get-up](getting-up.md) from a stable crouch to a full stand.

**Sweep the thresholds further.** v1→v2 was 0.5→1.0. Does 1.5 help the rotation complete, or does it just let the policy drift sloppily? There's a sweet spot between "cut off too early" and "no discipline at all."

**Inspect the reference's landing.** Play `smpl_backflip_to_g1.npz` frame by frame (`play_motion_npz.py`) and check the *landing* portion specifically — if the reference's own landing is brief or awkward, the policy has little to imitate there.

**Compare to the cartwheel.** The [cartwheel](imitation-cartwheel.md) *did* complete (sideways rotation, lower height) on the same task and pipeline. Rendering both side by side shows exactly why backward-and-up is the harder of the two.

---

*Unitree G1 on flat terrain, MuJoCo-Warp on a DGX Spark (NVIDIA GB10, aarch64). Motion-tracking task, single backflip reference, 4096 parallel robots, 20000 iterations per attempt; the variable between attempts is the termination-threshold set. The spec for this run: [2026-06-19-spine-backflip.md](../superpowers/specs/2026-06-19-spine-backflip.md).*
