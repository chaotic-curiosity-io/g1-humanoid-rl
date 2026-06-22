# Imitation Learning: Teaching G1 to Backflip (in three attempts)

*This report is the Imitation track's spine task. It builds on [imitation-cartwheel.md](imitation-cartwheel.md), which introduces motion imitation and the cartwheel campaign's hard-won lessons. Terms like [policy](00-primer.md), [reward](00-primer.md), and [episode](00-primer.md) are in [00-primer.md](00-primer.md). This is the longest road in the series — **three full training attempts**, each diagnosing and fixing a specific failure of the one before, ending in a landed backflip.*

---

## What you are about to see

A backflip is the hardest motion in this series: it needs a genuine **airborne phase**, a full **backward rotation** through inversion, *and* a stable **landing on the feet**. In motion imitation, the policy isn't rewarded for any of those in the abstract — it's rewarded for matching a pre-recorded reference backflip, frame by frame. If it matches closely all the way through, a backflip emerges. If it drifts too far from the reference, the episode ends.

Three full training attempts (~11 hours each) got us there:

| Attempt | The fix | Result |
|---|---|---|
| **v1** | cartwheel's tight thresholds (0.5 m / 0.8 rad) | **never leaves the ground** |
| **v2** | loosen thresholds (1.0 / 1.5) so episodes survive the air | **gets airborne, rotates — but lands on its back** |
| **v3** | add an explicit landing reward (feet-down + upright) | **full inversion, lands on its feet** ✓ |

The arc is the lesson: each attempt fixed exactly one diagnosed deficit, and "match the reference" alone was not enough — the takeoff, the survival-in-air, and the landing each had to be earned separately.

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

But v2 still lands on its back. Watching it frame by frame, the diagnosis is clean: the tracking reward pays for *matching the reference pose*, but **nothing specifically rewards ending on the feet** — so a partway rotation that crashes onto the back scores nearly as well as a clean landing. The optimiser had no reason to prefer feet-first.

---

## Attempt 3 — add a landing reward: it lands

The fix is a new reward term, `landing_feet_upright`, that pays for **feet-on-the-ground + torso-upright**, but only in the **last 40% of the motion** (the landing window) so it doesn't fight the inverted mid-flip. It's the same shaping insight that took the [get-up task](getting-up.md) from a stable crouch to a full stand: when a specific phase of the behaviour is missing, give it its own explicit reward.

```bash
  --env.rewards.landing-feet-upright.weight 5.0   # added on top of v2's loosened thresholds
```

*(The term ships off by default — weight 0 — so it leaves the cartwheel and every other tracking use untouched; it's switched on only for this run.)*

The result is a real, landed backflip. Takeoff, **full aerial inversion (the robot goes upside-down)**, rotation, and a feet-first landing it recovers to standing from:

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/s2_v3_landed_still.png">
  <source src="assets/s2_v3_landed_side.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/s2_v3_landed_side.mp4">download the clip</a> instead.
</video>

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/s2_v3_landed_still.png">
  <source src="assets/s2_v3_landed_chase.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/s2_v3_landed_chase.mp4">download the clip</a> instead.
</video>

Two things in the metrics confirm what the video shows. The landing reward **climbed steadily and held** (it rises only when the robot actually ends feet-down and upright) — and, tellingly, the body-tracking error **dropped to ~0.43** (v2 was stuck at ~0.63). Better full-motion tracking *and* a high landing reward together is the signature of a flip that's completed and landed, not bailed.

---

## The verdict

**v3 lands the backflip** — confirmed frame by frame from two angles (the visual gate, never the score). The honest nuance: the landing is a *recovering crouch* — the robot comes down on its feet and stays upright, but it's a controlled stumble-to-stand, not a crisp gymnast's stick. It is unmistakably a backflip: it launches, fully inverts in the air, and lands on its feet.

The three-attempt arc is the whole lesson in one task:

| Attempt | Change | Behaviour |
|---|---|---|
| v1 | tight thresholds | winds up, never leaves the ground |
| v2 | loosen thresholds | launches and rotates, lands on its back |
| v3 | + landing reward | **full inversion, lands on its feet** |

"Match the reference" was never a single switch. The takeoff needed loose enough termination to survive; the landing needed its own explicit reward. Each attempt isolated and fixed exactly one missing piece — the same iterative, watch-the-video, fix-one-thing discipline that runs through every report in this series.

---

## Tweak this to explore

**Sharpen the landing.** v3 lands in a recovering crouch. Raise the `landing_feet_upright` weight (try 8–10) or extend its gate window, and see whether the landing tightens toward a crisp stick — or whether the robot starts bailing the rotation early to grab the landing reward sooner (a classic over-weighting failure).

**Sweep the thresholds.** v1→v2 was 0.5→1.0. With the landing reward now in place, does an even looser threshold help the rotation complete more reliably, or just invite sloppy tracking? There's a sweet spot between "cut off too early" and "no discipline at all."

**Inspect the reference's landing.** Play `smpl_backflip_to_g1.npz` frame by frame (`play_motion_npz.py`) and check the *landing* portion — a reference with a clean, deliberate landing gives the policy more to imitate there.

**Compare to the cartwheel.** The [cartwheel](imitation-cartwheel.md) completed on the same task and pipeline with no landing reward at all (sideways rotation, lower height). Rendering both side by side shows exactly why backward-and-up needed three attempts where sideways needed one.

---

*Unitree G1 on flat terrain, MuJoCo-Warp on a DGX Spark (NVIDIA GB10, aarch64). Motion-tracking task, single backflip reference, 4096 parallel robots, 20000 iterations per attempt. v1→v2 changed the termination thresholds; v3 added the `landing_feet_upright` reward (committed under [`backflip-v3/`](../../backflip-v3/)). The spec for this run: [2026-06-19-spine-backflip.md](../superpowers/specs/2026-06-19-spine-backflip.md).*
