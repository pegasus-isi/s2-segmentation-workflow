#!/usr/bin/env python3
"""Compute row-normalized 3x3 confusion matrices for a trained model.

Emits the full confusion matrices (not just the diagonal) for the overall
test set and the high-/low-cloud strata, matching the paper's Fig 13
(auto-labeled row). Row-normalized so each row sums to 100% — the diagonal
is per-class recall; off-diagonals show the cloud-shadow-induced confusion
(e.g. thick ice classified as thin).

Class order = LabelEncoder order = sorted mask gray values:
    0 = thin ice (29), 1 = thick ice (76), 2 = open water (149).
"""

import argparse
import json
import sys

import numpy as np
from keras import backend as K
import tensorflow as tf
from sklearn.metrics import confusion_matrix


def recall_m(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pp = K.sum(K.round(K.clip(y_true, 0, 1)))
    return tp / (pp + K.epsilon())


def precision_m(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pp = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return tp / (pp + K.epsilon())


def f1_m(y_true, y_pred):
    p = precision_m(y_true, y_pred)
    r = recall_m(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))


def row_norm_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).astype(float)
    rs = cm.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return (cm / rs * 100.0).round(2).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--test-labels", required=True)
    ap.add_argument("--test-cloud-fractions", required=True)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    X = np.load(args.test_data)
    y_cat = np.load(args.test_labels)
    fracs = np.load(args.test_cloud_fractions)
    y_true = np.argmax(y_cat, axis=-1).reshape(-1)

    model = tf.keras.models.load_model(
        args.model,
        custom_objects={"recall_m": recall_m, "precision_m": precision_m,
                        "f1_m": f1_m})
    pred = model.predict(X, batch_size=args.batch_size, verbose=0)
    y_pred = np.argmax(pred, axis=-1).reshape(-1)

    # per-pixel stratum mask broadcast from per-tile cloud fraction
    px_per_tile = y_cat.shape[1] * y_cat.shape[2]
    valid = fracs >= 0
    high = valid & (fracs >= args.threshold)
    low = valid & (fracs < args.threshold)

    def expand(mask):
        return np.repeat(mask, px_per_tile)

    out = {
        "class_order": ["thin ice", "thick ice", "open water"],
        "overall": row_norm_cm(y_true, y_pred),
        "n_tiles": {"total": int(len(fracs)), "high": int(high.sum()),
                    "low": int(low.sum()), "dropped": int((~valid).sum())},
    }
    mh, ml = expand(high), expand(low)
    out["high_cloud"] = row_norm_cm(y_true[mh], y_pred[mh])
    out["low_cloud"] = row_norm_cm(y_true[ml], y_pred[ml])

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}")
    for k in ("overall", "high_cloud", "low_cloud"):
        print(k, out[k])


if __name__ == "__main__":
    main()
