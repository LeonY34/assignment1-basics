#!/usr/bin/env python3
"""Parse training logs and plot loss against iteration."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRIC_PATTERN = re.compile(
    r"\biteration=(?P<iteration>\d+)(?:/\d+)?\s*\|\s*loss=(?P<loss>[-+\d.eE]+)"
)


def parse_loss(log_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted iterations and losses, keeping the last duplicate iteration."""
    losses_by_iteration: dict[int, float] = {}
    with log_path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            match = METRIC_PATTERN.search(line)
            if match:
                losses_by_iteration[int(match["iteration"])] = float(match["loss"])

    if not losses_by_iteration:
        raise ValueError(f"No 'iteration=... | loss=...' records found in {log_path}")

    iterations = np.array(sorted(losses_by_iteration))
    losses = np.array([losses_by_iteration[i] for i in iterations])
    return iterations, losses


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Compute a trailing moving average with sensible values at the beginning."""
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    starts = np.maximum(0, np.arange(len(values)) + 1 - window)
    ends = np.arange(len(values)) + 1
    return (cumulative[ends] - cumulative[starts]) / (ends - starts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a loss curve from a training log.")
    parser.add_argument("log", type=Path, help="Path to the .log file")
    parser.add_argument("-o", "--output", type=Path, help="Output image (default: <log>_loss.png)")
    parser.add_argument(
        "-w", "--smooth-window", type=int, default=10, help="Moving-average window; 1 disables smoothing (default: 10)"
    )
    parser.add_argument("--dpi", type=int, default=160, help="Output resolution (default: 160)")
    parser.add_argument("--show", action="store_true", help="Also open an interactive plot window")
    args = parser.parse_args()

    if args.smooth_window < 1:
        parser.error("--smooth-window must be at least 1")
    if not args.log.is_file():
        parser.error(f"log file does not exist: {args.log}")

    output = args.output or args.log.with_name(f"{args.log.stem}_loss.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    iterations, losses = parse_loss(args.log)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, losses, color="tab:blue", alpha=0.3, linewidth=1, label="Loss")
    if args.smooth_window > 1:
        smoothed = moving_average(losses, args.smooth_window)
        ax.plot(
            iterations,
            smoothed,
            color="tab:blue",
            linewidth=2,
            label=f"Moving average ({args.smooth_window} points)",
        )

    ax.set(title=args.log.stem, xlabel="Iteration", ylabel="Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=args.dpi)
    print(f"Parsed {len(losses)} points; saved loss plot to {output}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
