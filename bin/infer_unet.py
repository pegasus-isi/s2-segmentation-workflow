#!/usr/bin/env python3

"""Run a trained U-Net over a full Sentinel-2 scene end-to-end.

Reproduces the inference pipeline shown in Fig. 9 of Iqrah et al.
(IPDPSW 2024): original scene → split into 256×256 tiles → (optional)
thin-cloud/shadow filter → U-Net.predict → merge predicted tiles into
a colour-coded sea-ice segmentation map covering the whole scene.

Outputs an RGB PNG using the paper's legend:
    red   = thick ice
    blue  = thin ice
    green = open water

The class→colour mapping is derived from preprocess_metadata.json
(``label_classes`` is the sorted list of grayscale values the encoder
saw — by construction, the OpenCV-grayscale values of {red, blue,
green}: roughly 29, 76, 149). When the metadata file is unavailable,
``--class-colors`` lets the user override the mapping explicitly.
"""

import argparse
import json
import logging
import os
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K

# filter_image lives in the same bin/ directory; import its core function so we
# don't duplicate the byte-faithful only_shadow_cloud_removal implementation.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filter_image import only_shadow_cloud_removal  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Custom Keras metrics (must match train_unet / evaluate_model exactly) ──

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


# ── Pipeline helpers ────────────────────────────────────────────────────────


def tile_image(img, tile_size):
    """Split a HxW grayscale image into a list of (row, col, tile) entries.

    Edge tiles are zero-padded to the full tile_size so the U-Net always
    sees a 256×256 input. Returns the original shape so the caller can
    crop back after merging.
    """
    h, w = img.shape[:2]
    tiles = []
    for r in range(0, h, tile_size):
        for c in range(0, w, tile_size):
            tile = img[r:r + tile_size, c:c + tile_size]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size), dtype=tile.dtype)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            tiles.append((r, c, tile))
    return tiles, (h, w)


def normalize_float32(tiles):
    """L2-normalize tiles along axis=1 in float32 — same as preprocess_data."""
    x = np.expand_dims(tiles, axis=3).astype(np.float32)
    norms = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
    np.maximum(norms, 1e-12, out=norms)
    x /= norms
    return x


def load_class_colors(metadata_path, override):
    """Return an (n_classes, 3) uint8 lookup table mapping class index → RGB.

    Order of resolution:
      1. ``override`` (list of "R,G,B" strings, one per class) — wins if set.
      2. Auto-mapping from preprocess_metadata.json's ``label_classes``.
      3. Default {red, blue, green} legend in encoder order.
    """
    if override:
        colors = []
        for s in override:
            parts = [int(p) for p in s.split(",")]
            if len(parts) != 3:
                raise ValueError(f"--class-colors entry must be 'R,G,B', got {s!r}")
            colors.append(parts)
        return np.array(colors, dtype=np.uint8)

    # Encoder sees the grayscale values of red, blue, green written by
    # color_segment.py. OpenCV's BGR-grayscale of those colours is:
    #     red   (PNG 255,0,0) → 29
    #     blue  (PNG   0,0,255) → 76
    #     green (PNG   0,255,0) → 149
    # LabelEncoder sorts these → 0=red, 1=blue, 2=green.
    gray_to_rgb = {
        29:  (255, 0, 0),    # thick ice
        76:  (0, 0, 255),    # thin ice
        149: (0, 255, 0),    # open water
    }
    default_legend = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]

    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path) as f:
            meta = json.load(f)
        label_classes = meta.get("label_classes", [])
        if label_classes:
            colors = []
            for g in label_classes:
                colors.append(gray_to_rgb.get(int(g), (128, 128, 128)))
            logger.info(
                f"Class colours from {metadata_path}: "
                + ", ".join(f"{g}->{rgb}" for g, rgb in zip(label_classes, colors)))
            return np.array(colors, dtype=np.uint8)
        logger.warning(f"{metadata_path} has no 'label_classes'; using default legend.")

    logger.info("Using default class colours: 0=red (thick ice), 1=blue (thin ice), 2=green (open water).")
    return np.array(default_legend, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Apply a trained U-Net to a Sentinel-2 scene (paper Fig 9).")
    parser.add_argument("--model", required=True,
                        help="Trained model.hdf5 (e.g. model_filtered.hdf5).")
    parser.add_argument("--input", required=True,
                        help="Input Sentinel-2 scene PNG (colour).")
    parser.add_argument("--output", required=True,
                        help="Output colour-coded prediction PNG.")
    parser.add_argument("--filter", action="store_true",
                        help="Apply only_shadow_cloud_removal before tiling "
                             "(use when the trained model was trained on "
                             "thin-cloud/shadow-filtered tiles).")
    parser.add_argument("--tile-size", type=int, default=256,
                        help="Inference tile size (default: 256 — must match training).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="model.predict batch size (default: 32).")
    parser.add_argument("--metadata", default=None,
                        help="preprocess_metadata.json — used to recover the "
                             "exact class→colour mapping from training labels.")
    parser.add_argument("--class-colors", nargs="+", default=None,
                        help="Explicit class→colour overrides as 'R,G,B' "
                             "strings, one per class (in encoder order). "
                             "Bypasses --metadata.")
    parser.add_argument("--save-intermediate", default=None,
                        help="Optional path to write the (filtered) grayscale "
                             "scene that was actually fed to the U-Net "
                             "(for debugging).")
    args = parser.parse_args()

    logger.info(f"Model: {args.model}")
    logger.info(f"Input scene: {args.input}")
    logger.info(f"Filter: {args.filter}")

    # ── 1. Load scene ──
    raw = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if raw is None:
        logger.error(f"Failed to read scene: {args.input}")
        sys.exit(1)
    logger.info(f"Scene size: {raw.shape[1]}x{raw.shape[0]}")

    # ── 2. Optional thin-cloud/shadow filter ──
    if args.filter:
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        filt = only_shadow_cloud_removal(rgb)
        gray = filt[:, :, 0]
    else:
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    if args.save_intermediate:
        cv2.imwrite(args.save_intermediate, gray)
        logger.info(f"Saved intermediate grayscale: {args.save_intermediate}")

    orig_h, orig_w = gray.shape

    # ── 3. Tile ──
    tiles, (h, w) = tile_image(gray, args.tile_size)
    logger.info(f"Tiled {h}x{w} scene into {len(tiles)} patches of {args.tile_size}².")
    tile_arr = np.stack([t for _, _, t in tiles], axis=0)

    # ── 4. Normalize (same as preprocess_data) ──
    X = normalize_float32(tile_arr)
    logger.info(f"Normalized tile array shape: {X.shape}")

    # ── 5. Load model ──
    custom_objects = {"recall_m": recall_m, "precision_m": precision_m, "f1_m": f1_m}
    model = tf.keras.models.load_model(args.model, custom_objects=custom_objects)
    n_classes = int(model.output_shape[-1])
    logger.info(f"Loaded model — n_classes={n_classes}.")

    # ── 6. Predict ──
    t0 = time.time()
    preds = model.predict(X, batch_size=args.batch_size, verbose=1)
    logger.info(f"Inference: {time.time() - t0:.1f}s for {len(X)} tiles.")
    class_idx = np.argmax(preds, axis=-1).astype(np.uint8)  # (N, H, W)

    # ── 7. Class index → RGB ──
    colors = load_class_colors(args.metadata, args.class_colors)
    if colors.shape[0] < n_classes:
        logger.error(
            f"Need {n_classes} class colours but only got {colors.shape[0]}. "
            "Pass --class-colors or a complete --metadata file.")
        sys.exit(1)

    # ── 8. Reassemble ──
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for (r, c, _), pred_tile in zip(tiles, class_idx):
        # Pred tile is the size of the input tile (tile_size). When the
        # original scene didn't cover the full tile (edge zero-padding),
        # crop back to the real region.
        rh = min(args.tile_size, orig_h - r)
        rw = min(args.tile_size, orig_w - c)
        canvas[r:r + rh, c:c + rw, :] = colors[pred_tile[:rh, :rw]]

    canvas = canvas[:orig_h, :orig_w, :]

    # ── 9. Save (cv2 writes BGR; convert from our RGB canvas) ──
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(args.output, bgr)
    logger.info(f"Prediction saved: {args.output}")


if __name__ == "__main__":
    main()
