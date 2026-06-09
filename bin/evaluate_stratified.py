#!/usr/bin/env python3

"""Stratified evaluation by tile cloud/shadow coverage (paper Table V + Fig 13).

Splits the test set into two strata based on per-tile cloud/shadow
fraction (default threshold 0.10 — "≥10%" vs "<10%" as in the paper),
evaluates the trained U-Net on each stratum separately, and emits:

  - evaluation_results_high_cloud.json  (overall loss / acc / F1 / P / R)
  - evaluation_results_low_cloud.json
  - high_cloud_confusion_matrix.png
  - low_cloud_confusion_matrix.png
  - high_cloud_per_class_metrics.json
  - low_cloud_per_class_metrics.json
  - high_cloud_metrics_table.png
  - low_cloud_metrics_table.png
  - stratified_summary.json (paper Table V row for this branch)

Use the ``--prefix`` option to namespace outputs when running this for
multiple branches (e.g. ``--prefix orig_`` and ``--prefix filtered_``).
"""

import argparse
import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from keras import backend as K  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix as sk_confusion_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Custom Keras metrics (must match training exactly) ────────────────────

def recall_m(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pp = K.sum(K.round(K.clip(y_true, 0, 1)))
    return tp / (pp + K.epsilon())


def precision_m(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pred_p = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return tp / (pred_p + K.epsilon())


def f1_m(y_true, y_pred):
    p = precision_m(y_true, y_pred)
    r = recall_m(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))


# ── Plot helper (mirrors generate_plots.plot_confusion_matrix) ────────────

def plot_cm(y_true, y_pred, class_names, out_path, title, dpi=150):
    labels = list(range(len(class_names)))
    cm = sk_confusion_matrix(y_true, y_pred, labels=labels)
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1, row_sum)
    cm_norm = cm.astype(float) / row_sum

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(
                j, i,
                f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_metrics_table(per_class, class_names, title, out_path, dpi=150):
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    rows = []
    for i, name in enumerate(class_names):
        rows.append([
            name,
            f"{per_class['precision'][i]:.4f}",
            f"{per_class['recall'][i]:.4f}",
            f"{per_class['f1-score'][i]:.4f}",
            f"{int(per_class['support'][i]):,}",
        ])
    fig, ax = plt.subplots(figsize=(8, 1 + 0.5 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def evaluate_subset(model, X, y_cat, label, class_names, out_dir, prefix,
                    batch_size, dpi):
    """Evaluate model on (X, y_cat), write JSON / CM / metrics table.

    Returns a small dict with accuracy / f1 / precision / recall / support
    so the caller can assemble the Table V summary row.
    """
    if len(X) == 0:
        logger.warning(f"[{label}] empty subset — skipping evaluation.")
        return None

    logger.info(f"[{label}] X={X.shape} y={y_cat.shape}")
    loss, acc, f1, p, r = model.evaluate(X, y_cat, verbose=1,
                                         batch_size=batch_size)
    summary = {
        "stratum": label,
        "n_tiles": int(X.shape[0]),
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "f1_score": float(f1),
        "precision": float(p),
        "recall": float(r),
    }
    eval_path = os.path.join(out_dir, f"{prefix}evaluation_results_{label}.json")
    with open(eval_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"[{label}] {eval_path}: acc={acc * 100:.2f}% F1={f1:.4f}")

    # Per-pixel confusion matrix + per-class report.
    y_pred_proba = model.predict(X, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=-1)
    y_true = np.argmax(y_cat, axis=-1)

    labels = list(range(len(class_names)))
    cm_path = os.path.join(out_dir, f"{prefix}{label}_confusion_matrix.png")
    plot_cm(y_true.flatten(), y_pred.flatten(), class_names, cm_path,
            title=f"Confusion Matrix — {label.replace('_', ' ')}", dpi=dpi)

    report = classification_report(
        y_true.flatten(), y_pred.flatten(),
        labels=labels, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    per_class = {
        "precision": [report[c]["precision"] for c in class_names],
        "recall":    [report[c]["recall"]    for c in class_names],
        "f1-score":  [report[c]["f1-score"]  for c in class_names],
        "support":   [report[c]["support"]   for c in class_names],
    }
    pc_path = os.path.join(out_dir, f"{prefix}{label}_per_class_metrics.json")
    with open(pc_path, "w") as f:
        json.dump(per_class, f, indent=2)
    logger.info(f"Saved {pc_path}")

    mt_path = os.path.join(out_dir, f"{prefix}{label}_metrics_table.png")
    plot_metrics_table(per_class, class_names,
                       title=f"Stratified metrics — {label.replace('_', ' ')}",
                       out_path=mt_path, dpi=dpi)

    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Trained model.hdf5")
    ap.add_argument("--test-data", required=True, help="X_test.npy")
    ap.add_argument("--test-labels", required=True, help="y_test_cat.npy")
    ap.add_argument("--test-cloud-fractions", required=True,
                    help="Per-test-tile cloud/shadow fractions "
                         "(produced by preprocess_data --cloud-fraction).")
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="Cloud-fraction cutoff between strata (default: 0.10, "
                         "matching the paper's '≥10% / <10%' split).")
    ap.add_argument("--output-dir", default=".",
                    help="Directory for stratified evaluation outputs (default: cwd).")
    ap.add_argument("--prefix", default="",
                    help="Optional prefix for output filenames "
                         "(e.g. 'orig_' or 'filtered_').")
    ap.add_argument("--class-names", default="Thick Ice,Thin Ice,Open Water",
                    help="Comma-separated class labels (default: paper legend).")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    class_names = [c.strip() for c in args.class_names.split(",") if c.strip()]

    X_test = np.load(args.test_data)
    y_test_cat = np.load(args.test_labels)
    fracs = np.load(args.test_cloud_fractions)

    if len(fracs) != len(X_test):
        logger.error(
            f"Cloud-fraction array length {len(fracs)} != X_test length "
            f"{len(X_test)}; alignment broken — abort.")
        sys.exit(1)

    valid = fracs >= 0
    high_mask = valid & (fracs >= args.threshold)
    low_mask = valid & (fracs < args.threshold)
    dropped = int((~valid).sum())
    logger.info(
        f"Stratify at {args.threshold}: high={int(high_mask.sum())}, "
        f"low={int(low_mask.sum())}, dropped={dropped} (no cloud-fraction key)")

    custom_objects = {"recall_m": recall_m, "precision_m": precision_m, "f1_m": f1_m}
    model = tf.keras.models.load_model(args.model, custom_objects=custom_objects)

    summaries = {}
    for label, mask in (("high_cloud", high_mask), ("low_cloud", low_mask)):
        s = evaluate_subset(
            model, X_test[mask], y_test_cat[mask],
            label=label, class_names=class_names,
            out_dir=args.output_dir, prefix=args.prefix,
            batch_size=args.batch_size, dpi=args.dpi,
        )
        if s is not None:
            summaries[label] = s

    summary_path = os.path.join(
        args.output_dir, f"{args.prefix}stratified_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "threshold": args.threshold,
            "dropped_no_fraction": dropped,
            "high_cloud": summaries.get("high_cloud"),
            "low_cloud": summaries.get("low_cloud"),
        }, f, indent=2)
    logger.info(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
