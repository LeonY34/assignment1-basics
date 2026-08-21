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
EVAL_METRIC_PATTERN = re.compile(
    r"\biteration=(?P<iteration>\d+)(?:/\d+)?\s*\|\s*eval_loss=(?P<loss>[-+\d.eE]+)"
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


def parse_eval_loss(log_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted evaluation iterations and losses."""
    losses_by_iteration: dict[int, float] = {}
    with log_path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            match = EVAL_METRIC_PATTERN.search(line)
            if match:
                losses_by_iteration[int(match["iteration"])] = float(match["loss"])

    iterations = np.array(sorted(losses_by_iteration), dtype=int)
    losses = np.array([losses_by_iteration[i] for i in iterations], dtype=float)
    return iterations, losses


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Compute a trailing moving average with sensible values at the beginning."""
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    starts = np.maximum(0, np.arange(len(values)) + 1 - window)
    ends = np.arange(len(values)) + 1
    return (cumulative[ends] - cumulative[starts]) / (ends - starts)


def plot_losses(
    log_path: Path,
    output: Path | None = None,
    smooth_window: int = 10,
    dpi: int = 160,
    show: bool = False,
) -> Path:
    """Plot training and evaluation losses parsed from one run log."""
    if smooth_window < 1:
        raise ValueError("smooth_window must be at least 1")
    if not log_path.is_file():
        raise FileNotFoundError(f"log file does not exist: {log_path}")

    output = output or log_path.with_name(f"{log_path.stem}_loss.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    iterations, losses = parse_loss(log_path)
    eval_iterations, eval_losses = parse_eval_loss(log_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, losses, color="tab:blue", alpha=0.3, linewidth=1, label="Loss")
    if smooth_window > 1:
        smoothed = moving_average(losses, smooth_window)
        ax.plot(
            iterations,
            smoothed,
            color="tab:blue",
            linewidth=2,
            label=f"Moving average ({smooth_window} points)",
        )
    if len(eval_iterations):
        ax.plot(
            eval_iterations,
            eval_losses,
            color="tab:red",
            marker="o",
            linewidth=2,
            label="Evaluation loss",
        )

    ax.set(title=log_path.stem, xlabel="Iteration", ylabel="Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    return output


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

    try:
        output = plot_losses(
            log_path=args.log,
            output=args.output,
            smooth_window=args.smooth_window,
            dpi=args.dpi,
            show=args.show,
        )
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    _, losses = parse_loss(args.log)
    _, eval_losses = parse_eval_loss(args.log)
    print(
        f"Parsed {len(losses)} training points and {len(eval_losses)} evaluation points; "
        f"saved loss plot to {output}"
    )


if __name__ == "__main__":
    main()
