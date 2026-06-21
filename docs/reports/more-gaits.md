# More Gaits: Promoting the Deferred Tier-1 Specs

*This report is part of the Locomotion track. It builds directly on [03 — Turning the knobs](03-turning-the-knobs.md) and [running and flight](running-and-flight.md): same velocity task, same "change the command, override the curriculum" technique, no new code. If [policy](00-primer.md) / [reward](00-primer.md) / [curriculum](03-turning-the-knobs.md) are unfamiliar, start at [00-primer.md](00-primer.md).*

---

## What this is

The curriculum's [ready-to-run specs](../superpowers/specs/2026-06-19-tier1-gait-tweaks-compact.md) describe a menu of gait variations that were *specced but not trained* — each a one-line change to the commanded velocity on the existing walking task. This report **promotes two of them to real trained policies**, to show that the deferred specs are genuinely runnable and to document what actually happens when you do.

Both are pure command changes — no new reward, no code — on `Mjlab-Velocity-Flat-Unitree-G1`, 1500 iterations each (~25 min). The only subtlety is the one from [report 03](03-turning-the-knobs.md): the flat task's **speed curriculum re-randomizes the command every reset**, so to pin a gait you must override the command *and* the curriculum stages, with `=`-syntax tuples.

---

## Gait 1 — Spin in place

Zero the linear command, set a constant yaw rate. The robot should turn on the spot rather than travel.

```bash
ssh spark "docker exec mjlab-dev bash -lc 'cd /workspace/mjlab && python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 2048 --agent.max-iterations 1500 \
  \"--env.commands.twist.ranges.lin-vel-x=(0.0, 0.0)\" \
  \"--env.commands.twist.ranges.lin-vel-y=(0.0, 0.0)\" \
  \"--env.commands.twist.ranges.ang-vel-z=(1.0, 1.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.0.lin-vel-x=(0.0, 0.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.1.lin-vel-x=(0.0, 0.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.2.lin-vel-x=(0.0, 0.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.0.ang-vel-z=(1.0, 1.0)\" \
  \"--env.curriculum.command-vel.params.velocity-stages.1.ang-vel-z=(1.0, 1.0)\"'"
```

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/gait_spin_still.png">
  <source src="assets/gait_spin_chase.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/gait_spin_chase.mp4">download the clip</a> instead.
</video>

**Result: a clean turn-in-place.** The robot stays over its footprint and pivots, stepping its feet around to rotate. Final reward ~76 (higher than the ~50 of the forward walker — turning in place with no translation target is, in reward terms, an easier command to satisfy well). It tracks the yaw command and keeps its linear drift near zero. This one worked on the first try.

---

## Gait 2 — Walk backward

Command a negative forward velocity. Everything else the same.

```bash
  "--env.commands.twist.ranges.lin-vel-x=(-1.0, -1.0)"
  "--env.curriculum.command-vel.params.velocity-stages.0.lin-vel-x=(-1.0, -1.0)"
  "--env.curriculum.command-vel.params.velocity-stages.1.lin-vel-x=(-1.0, -1.0)"
  "--env.curriculum.command-vel.params.velocity-stages.2.lin-vel-x=(-1.0, -1.0)"
```

<video controls autoplay loop muted playsinline preload="auto" width="100%" poster="assets/gait_backward_still.png">
  <source src="assets/gait_backward_side.mp4" type="video/mp4">
  Your browser doesn't support embedded video — <a href="assets/gait_backward_side.mp4">download the clip</a> instead.
</video>

**Result: an honest "partly there."** The robot stays upright and steps backward, but it tracks the −1.0 m/s command noticeably less cleanly than the forward walker tracks its target — final reward ~39 (vs ~50 forward) and a much higher commanded-velocity error. Backward walking is genuinely harder for this robot: the G1's mass distribution, ankle range, and the reward shaping were all tuned around *forward* locomotion, so the same recipe produces a less polished backward gait in the same 1500 iterations. A longer run, or a reward term that specifically rewards backward-stability, would likely close the gap.

That contrast is the useful part: **the same one-line technique can yield a first-try success (spin) and a needs-more-work result (backward) on the same task** — a reminder that "it's just a command change" doesn't guarantee equal quality across commands.

---

## Tweak this to explore

**The rest of the Tier-1 menu.** The [compact specs](../superpowers/specs/2026-06-19-tier1-gait-tweaks-compact.md) list more: sideways (strafe) walking (`lin-vel-y`), energy-efficiency (crank the `action_rate_l2` / joint penalties and measure cost-of-transport), and a prescribed hop/march (the one that *does* need a small new contact-schedule reward term). Each is the same promote-a-spec exercise.

**Push the backward gait further.** Train it to 3000+ iterations, or combine a modest backward command with a small `air_time` or stability incentive, and see whether the tracking error comes down to forward-walker levels.

**Diagonal and curved paths.** Combine `lin-vel-x`, `lin-vel-y`, and `ang-vel-z` to command a curving walk — a single policy doing all of it is the [goal-conditioned multi-skill](../superpowers/specs/2026-06-19-tier1-gait-tweaks-compact.md) idea in miniature.

---

*Unitree G1 on flat terrain, MuJoCo-Warp on a DGX Spark (NVIDIA GB10, aarch64). Two deferred Tier-1 specs promoted to trained policies; 2048 parallel robots, 1500 iterations each, pure command/curriculum overrides. Specs: [2026-06-19-tier1-gait-tweaks-compact.md](../superpowers/specs/2026-06-19-tier1-gait-tweaks-compact.md).*
