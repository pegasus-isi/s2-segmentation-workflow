#!/usr/bin/env python3

"""Aggregate multiple training_history.json files into paper Fig 12-style plots.

Run the workflow at several GPU counts (e.g. 1, 2, 4, 6, 8) — each run
produces a ``training_history{_branch}.json`` file recording per-epoch
wall time, samples/sec, and the ``training_meta`` block with the replica
count. Point this script at those JSON files and it emits four
subplots:

  (a) distributed-training speedup vs the 1-GPU baseline,
  (b) samples processed per second per epoch,
  (c) total training time vs #GPUs,
  (d) average time per epoch vs #GPUs.

Plus a CSV summary ``speedup_summary.csv`` so the numbers can drop
straight into the paper. By default the script looks for files matching
``training_history*.json`` under ``--output-dir``; pass ``--history``
paths explicitly to override.
"""

import argparse
import csv
import glob
import json
import logging
import os
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Run:
    path: str
    replicas: int
    mode: str
    batch_size: int
    epochs: int
    samples_per_epoch: int
    total_time_s: float
    mean_epoch_time_s: float
    mean_samples_per_sec: float


def load_run(path):
    with open(path) as f:
        h = json.load(f)
    meta = h.get("training_meta", {})
    epoch_times = h.get("epoch_time_seconds") or []
    sps = h.get("samples_per_second") or []
    total = h.get("training_time_seconds")
    if not epoch_times:
        logger.warning(f"{path}: no epoch_time_seconds — falling back to total/epochs")
    if total is None and epoch_times:
        total = float(sum(epoch_times))
    return Run(
        path=path,
        replicas=int(meta.get("replicas", 1)),
        mode=str(meta.get("mode", "unknown")),
        batch_size=int(meta.get("batch_size", 0)),
        epochs=int(meta.get("epochs", len(epoch_times) or 0)),
        samples_per_epoch=int(meta.get("samples_per_epoch", 0)),
        total_time_s=float(total or 0.0),
        mean_epoch_time_s=float(np.mean(epoch_times)) if epoch_times else 0.0,
        mean_samples_per_sec=float(np.mean(sps)) if sps else 0.0,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--history", nargs="*", default=None,
                    help="Paths to training_history*.json files (one per "
                         "GPU count). If omitted, the script auto-discovers "
                         "training_history*.json under --output-dir.")
    ap.add_argument("--output-dir", default=".",
                    help="Directory containing history files / where the "
                         "plot and CSV are written (default: cwd).")
    ap.add_argument("--output", default="speedup_plot.png",
                    help="Output plot filename (relative to --output-dir).")
    ap.add_argument("--csv", default="speedup_summary.csv",
                    help="Output CSV filename (relative to --output-dir).")
    ap.add_argument("--title", default="Distributed U-Net training scaling",
                    help="Figure title.")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    if args.history:
        paths = sorted(args.history)
    else:
        paths = sorted(glob.glob(
            os.path.join(args.output_dir, "training_history*.json")))
    if not paths:
        logger.error("No training_history*.json found.")
        sys.exit(1)
    logger.info(f"Loading {len(paths)} run(s): {paths}")

    runs = [load_run(p) for p in paths]
    # Sort by replica count so the curves are monotonic.
    runs.sort(key=lambda r: r.replicas)

    replicas = np.array([r.replicas for r in runs], dtype=float)
    total_times = np.array([r.total_time_s for r in runs], dtype=float)
    epoch_times = np.array([r.mean_epoch_time_s for r in runs], dtype=float)
    samples_per_sec = np.array([r.mean_samples_per_sec for r in runs],
                               dtype=float)

    # Use the smallest replica count as the speedup baseline (typically 1).
    base_time = total_times[0] if total_times[0] > 0 else np.nan
    speedup = base_time / total_times if base_time else np.zeros_like(total_times)

    # ── CSV summary ────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, args.csv)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "history_file", "mode", "replicas", "batch_size", "epochs",
            "samples_per_epoch", "total_time_s", "mean_epoch_time_s",
            "mean_samples_per_sec", "speedup_vs_baseline",
        ])
        for r, s in zip(runs, speedup):
            w.writerow([
                os.path.basename(r.path), r.mode, r.replicas, r.batch_size,
                r.epochs, r.samples_per_epoch, f"{r.total_time_s:.3f}",
                f"{r.mean_epoch_time_s:.3f}",
                f"{r.mean_samples_per_sec:.2f}", f"{s:.3f}",
            ])
    logger.info(f"Wrote {csv_path}")

    # ── Plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) speedup vs ideal
    ax = axes[0, 0]
    ideal = replicas / replicas[0]
    ax.plot(replicas, ideal, "k--", label="Ideal (linear)", alpha=0.6)
    ax.plot(replicas, speedup, "o-", color="#1f6feb", label="Measured")
    ax.set_xlabel("Number of replicas (GPUs)")
    ax.set_ylabel(f"Speedup vs {int(replicas[0])}-GPU baseline")
    ax.set_title("(a) Distributed training speedup")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (b) samples/sec
    ax = axes[0, 1]
    ax.bar(replicas.astype(int).astype(str), samples_per_sec,
           color="#1a7f37", alpha=0.85)
    ax.set_xlabel("Number of replicas (GPUs)")
    ax.set_ylabel("Samples / sec (mean across epochs)")
    ax.set_title("(b) Throughput per epoch")
    ax.grid(True, alpha=0.3, axis="y")

    # (c) total training time
    ax = axes[1, 0]
    ax.plot(replicas, total_times, "o-", color="#cf222e")
    ax.set_xlabel("Number of replicas (GPUs)")
    ax.set_ylabel("Total training time (s)")
    ax.set_title("(c) Total training time")
    ax.grid(True, alpha=0.3)

    # (d) per-epoch time
    ax = axes[1, 1]
    ax.plot(replicas, epoch_times, "o-", color="#9a6700")
    ax.set_xlabel("Number of replicas (GPUs)")
    ax.set_ylabel("Mean time per epoch (s)")
    ax.set_title("(d) Time per epoch")
    ax.grid(True, alpha=0.3)

    fig.suptitle(args.title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(args.output_dir, args.output)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
