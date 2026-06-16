#!/usr/bin/env python3

"""Generate publication figures and tables from workflow outputs.

Produces:
  - training_curves.png    — Loss, accuracy, F1, precision/recall vs epoch
  - confusion_matrix.png   — Normalized confusion matrix (Fig 13)
  - prediction_samples.png — Side-by-side input/truth/prediction (Fig 14)
  - metrics_table.png      — Classification metrics table (Table IV)
  - per_class_metrics.json — Per-class precision/recall/F1 as JSON
"""

import argparse
import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Custom Keras metrics (needed to load the model) ──────────────────

def recall_m(y_true, y_pred):
    from keras import backend as K
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def precision_m(y_true, y_pred):
    from keras import backend as K
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def f1_m(y_true, y_pred):
    from keras import backend as K
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# ── Plot functions ───────────────────────────────────────────────────

def plot_training_curves(history, output_dir, dpi=150, prefix=""):
    """2x2 subplot grid: loss, accuracy, F1, precision/recall vs epoch."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = range(1, len(history["loss"]) + 1)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["loss"], "b-", label="Train")
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], "r-", label="Validation")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history["accuracy"], "b-", label="Train")
    if "val_accuracy" in history:
        ax.plot(epochs, history["val_accuracy"], "r-", label="Validation")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F1 Score
    ax = axes[1, 0]
    ax.plot(epochs, history["f1_m"], "b-", label="Train")
    if "val_f1_m" in history:
        ax.plot(epochs, history["val_f1_m"], "r-", label="Validation")
    ax.set_title("F1 Score")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Precision & Recall
    ax = axes[1, 1]
    ax.plot(epochs, history["precision_m"], "b-", label="Train Precision")
    ax.plot(epochs, history["recall_m"], "b--", label="Train Recall")
    if "val_precision_m" in history:
        ax.plot(epochs, history["val_precision_m"], "r-", label="Val Precision")
    if "val_recall_m" in history:
        ax.plot(epochs, history["val_recall_m"], "r--", label="Val Recall")
    ax.set_title("Precision & Recall")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(output_dir, prefix + "training_curves.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir, dpi=150, labels=None, prefix=""):
    """Normalized confusion matrix matching paper Fig 13."""
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

    cm = sk_confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

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
        title="Normalized Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
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
    out_path = os.path.join(output_dir, prefix + "confusion_matrix.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_prediction_samples(X_test, y_true, y_pred, class_names, n, output_dir, dpi=150, prefix=""):
    """Grid of N samples: input image | ground truth mask | predicted mask."""
    n_classes = len(class_names)
    n = min(n, len(X_test))

    # Pick evenly-spaced indices
    indices = np.linspace(0, len(X_test) - 1, n, dtype=int)

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        # Input image
        ax = axes[row, 0]
        if X_test[idx].ndim == 3 and X_test[idx].shape[-1] == 1:
            ax.imshow(X_test[idx].squeeze(), cmap="gray")
        else:
            ax.imshow(X_test[idx])
        ax.set_title(f"Input (#{idx})")
        ax.axis("off")

        # Ground truth
        ax = axes[row, 1]
        ax.imshow(y_true[idx], cmap="viridis", vmin=0, vmax=n_classes - 1)
        ax.set_title("Ground Truth")
        ax.axis("off")

        # Prediction
        ax = axes[row, 2]
        ax.imshow(y_pred[idx], cmap="viridis", vmin=0, vmax=n_classes - 1)
        ax.set_title("Prediction")
        ax.axis("off")

    fig.suptitle("Sample Predictions", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(output_dir, prefix + "prediction_samples.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_metrics_table(eval_results, per_class, class_names, output_dir, dpi=150, prefix=""):
    """Render Table IV as a matplotlib table image and save per-class JSON."""
    # Build table data
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    rows = []
    for i, name in enumerate(class_names):
        rows.append([
            name,
            f"{per_class['precision'][i]:.4f}",
            f"{per_class['recall'][i]:.4f}",
            f"{per_class['f1-score'][i]:.4f}",
            str(per_class['support'][i]),
        ])

    # Overall row
    rows.append([
        "Overall",
        f"{eval_results.get('precision', 0):.4f}",
        f"{eval_results.get('recall', 0):.4f}",
        f"{eval_results.get('f1_score', 0):.4f}",
        str(sum(per_class['support'])),
    ])

    fig, ax = plt.subplots(figsize=(10, 2 + 0.5 * len(rows)))
    ax.axis("off")
    ax.set_title("Classification Metrics (Table IV)", fontsize=14, fontweight="bold", pad=20)

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    # Style header row
    for j in range(len(headers)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    out_path = os.path.join(output_dir, prefix + "metrics_table.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")

    # Save per-class JSON
    json_path = os.path.join(output_dir, prefix + "per_class_metrics.json")
    with open(json_path, "w") as f:
        json.dump(per_class, f, indent=2)
    logger.info(f"Saved {json_path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication figures and tables from workflow outputs",
    )
    parser.add_argument("--training-history", required=True,
                        help="Path to training_history.json")
    parser.add_argument("--evaluation-results", required=True,
                        help="Path to evaluation_results.json")
    parser.add_argument("--model", required=True,
                        help="Path to model.hdf5")
    parser.add_argument("--test-data", required=True,
                        help="Path to X_test.npy")
    parser.add_argument("--test-labels", required=True,
                        help="Path to y_test_cat.npy")
    parser.add_argument("--metadata", default=None,
                        help="Path to preprocess_metadata.json (optional)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for output files")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of prediction samples to visualize (default: 5)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Plot resolution (default: 150)")
    parser.add_argument("--class-names", type=str, default="Thin Ice,Thick Ice,Open Water",
                        help="Comma-separated class names in label-encoder order "
                             "(sorted mask gray values 29=thin, 76=thick, 149=water; "
                             "default: 'Thin Ice,Thick Ice,Open Water')")
    parser.add_argument("--prefix", type=str, default="",
                        help="Prefix prepended to every output filename "
                             "(e.g. 'orig_' / 'filtered_') to disambiguate "
                             "multiple branches writing to the same directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    class_names = [c.strip() for c in args.class_names.split(",")]

    # ── 1. Training curves ──
    logger.info("Loading training history...")
    with open(args.training_history) as f:
        history = json.load(f)
    plot_training_curves(history, args.output_dir, dpi=args.dpi, prefix=args.prefix)

    # ── 2. Load model and test data for predictions ──
    logger.info("Loading model and test data...")
    import tensorflow as tf

    custom_objects = {
        "recall_m": recall_m,
        "precision_m": precision_m,
        "f1_m": f1_m,
    }
    model = tf.keras.models.load_model(args.model, custom_objects=custom_objects)

    X_test = np.load(args.test_data)
    y_test_cat = np.load(args.test_labels)
    logger.info(f"Test data: {X_test.shape}, Labels: {y_test_cat.shape}")

    # Determine n_classes from the actual data (last dimension of one-hot labels)
    n_classes = y_test_cat.shape[-1]
    logger.info(f"Detected n_classes={n_classes} from label shape {y_test_cat.shape}")

    # If class_names don't match the actual number of classes, auto-generate them
    if len(class_names) != n_classes:
        logger.warning(
            f"--class-names has {len(class_names)} entries but data has "
            f"{n_classes} classes. Auto-generating class names."
        )
        class_names = [f"Class {i}" for i in range(n_classes)]

    # Get predictions
    logger.info("Running predictions on test set...")
    y_pred_probs = model.predict(X_test)

    # Convert from one-hot to class indices
    y_true_flat = np.argmax(y_test_cat.reshape(-1, n_classes), axis=1)
    y_pred_flat = np.argmax(y_pred_probs.reshape(-1, n_classes), axis=1)

    # Spatial masks for visualization
    if y_test_cat.ndim == 4:
        # (N, H, W, C) → (N, H, W)
        y_true_spatial = np.argmax(y_test_cat, axis=-1)
        y_pred_spatial = np.argmax(y_pred_probs, axis=-1)
    else:
        y_true_spatial = np.argmax(y_test_cat, axis=-1)
        y_pred_spatial = np.argmax(y_pred_probs, axis=-1)

    # Use explicit label list so sklearn functions stay in sync with class_names
    labels = list(range(n_classes))

    # ── 3. Confusion matrix ──
    logger.info("Generating confusion matrix...")
    plot_confusion_matrix(y_true_flat, y_pred_flat, class_names,
                          output_dir=args.output_dir, dpi=args.dpi,
                          labels=labels, prefix=args.prefix)

    # ── 4. Prediction samples ──
    logger.info("Generating prediction samples...")
    plot_prediction_samples(X_test, y_true_spatial, y_pred_spatial,
                            class_names, args.num_samples,
                            args.output_dir, dpi=args.dpi, prefix=args.prefix)

    # ── 5. Per-class metrics and table ──
    logger.info("Computing per-class metrics...")
    from sklearn.metrics import classification_report

    with open(args.evaluation_results) as f:
        eval_results = json.load(f)

    report = classification_report(
        y_true_flat, y_pred_flat,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class = {
        "precision": [report[c]["precision"] for c in class_names],
        "recall": [report[c]["recall"] for c in class_names],
        "f1-score": [report[c]["f1-score"] for c in class_names],
        "support": [int(report[c]["support"]) for c in class_names],
    }

    plot_metrics_table(eval_results, per_class, class_names,
                       args.output_dir, dpi=args.dpi, prefix=args.prefix)

    logger.info("All plots generated successfully.")


if __name__ == "__main__":
    main()
