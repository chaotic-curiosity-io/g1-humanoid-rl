#!/usr/bin/env python3
"""Extract RSL-RL tensorboard scalars to CSV and render PNG curve plots.

Runs inside the mjlab-dev container (needs tensorboard + matplotlib, both present).
Pass --run multiple times (optionally name=dir) to overlay runs on each tag.

Examples
--------
python scripts/plot_training_curves.py --run logs/rsl_rl/g1_velocity/<ctrl> --out out/arc/ctrl
python scripts/plot_training_curves.py \
    --run baseline=logs/rsl_rl/g1_velocity/2026-04-17_18-46-23 \
    --run control=logs/rsl_rl/g1_velocity/<ctrl> \
    --tags Train/mean_reward Train/mean_episode_length --out out/arc/repro
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_event_file(run_dir: Path) -> Path:
    hits = sorted(glob.glob(str(run_dir / "events.out.tfevents*")))
    if not hits:
        raise FileNotFoundError(f"no tfevents in {run_dir}")
    return Path(hits[-1])


def load_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    ea = EventAccumulator(str(find_event_file(run_dir)), size_guidance={"scalars": 0})
    ea.Reload()
    return {
        tag: [(s.step, s.value) for s in ea.Scalars(tag)]
        for tag in ea.Tags().get("scalars", [])
    }


def write_csv(scalars: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "step", "value"])
        for tag, series in scalars.items():
            for step, val in series:
                w.writerow([tag, step, val])


def plot_tag(runs: dict[str, dict], tag: str, out_png: Path) -> bool:
    plt.figure(figsize=(8, 5))
    plotted = False
    for name, scalars in runs.items():
        series = scalars.get(tag)
        if not series:
            continue
        xs = [s for s, _ in series]
        ys = [v for _, v in series]
        plt.plot(xs, ys, label=name, linewidth=2)
        plotted = True
    if not plotted:
        plt.close()
        return False
    plt.xlabel("training iteration")
    plt.ylabel(tag)
    plt.title(tag)
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()
    return True


def parse_run(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, p = spec.split("=", 1)
        return name, Path(p)
    return Path(spec).name, Path(spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="run dir, optionally name=dir; repeatable to overlay")
    ap.add_argument("--tags", nargs="*",
                    default=["Train/mean_reward", "Train/mean_episode_length"])
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    runs = {name: load_scalars(p) for name, p in map(parse_run, args.run)}
    out = Path(args.out)
    for name, scalars in runs.items():
        write_csv(scalars, out / f"{name}_scalars.csv")
    made = [str(out / f"{t.replace('/', '_')}.png")
            for t in args.tags if plot_tag(runs, t, out / f"{t.replace('/', '_')}.png")]
    print("WROTE:\n  " + "\n  ".join(made))


if __name__ == "__main__":
    main()
