#!/usr/bin/env python3

"""Preprocess training images and masks for U-Net training.

Loads grayscale training images and segmentation masks, encodes labels,
normalizes pixel values, performs train/test split, and one-hot encodes
the masks. Outputs NumPy arrays for training and evaluation.

Memory-efficient: splits file indices first, then loads and processes
each split (train/test) separately so that the full dataset is never
held in memory at once. Uses float32 throughout to halve memory vs
the default float64.
"""

import argparse
import json
import logging
import os
import sys

import cv2
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_images(file_paths):
    """Load grayscale images from a list of file paths as uint8."""
    images = []
    for path in file_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
        else:
            logger.warning(f"Could not read: {path}")
    return np.array(images, dtype=np.uint8)


def normalize_float32(images):
    """L2-normalize images along axis=1 in float32 (no float64 intermediate).

    Equivalent to keras.utils.normalize(images, axis=1) but stays in
    float32 throughout, avoiding the ~2x memory spike from float64.
    """
    x = np.expand_dims(images, axis=3).astype(np.float32)
    del images
    # L2 norm per sample (axis=1) — same as keras normalize(x, axis=1)
    norms = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
    np.maximum(norms, 1e-12, out=norms)  # avoid div-by-zero, in-place
    x /= norms  # in-place division
    del norms
    return x


def encode_masks(masks, labelencoder, n_classes):
    """Label-encode and one-hot encode masks, returning float32 array."""
    n, h, w = masks.shape
    flat = masks.reshape(-1, 1)
    encoded = labelencoder.transform(flat).reshape(n, h, w).astype(np.uint8)
    del flat
    cat = to_categorical(
        np.expand_dims(encoded, axis=3), num_classes=n_classes,
    ).astype(np.float32).reshape(n, h, w, n_classes)
    del encoded
    return cat


def main():
    parser = argparse.ArgumentParser(description="Preprocess training data for U-Net")
    parser.add_argument("--image", action="append", required=True,
                        help="Training image PNG (can be specified multiple times)")
    parser.add_argument("--mask", action="append", required=True,
                        help="Training mask PNG (can be specified multiple times)")
    parser.add_argument("--x-train", required=True, help="Output X_train.npy")
    parser.add_argument("--x-test", required=True, help="Output X_test.npy")
    parser.add_argument("--y-train", required=True, help="Output y_train_cat.npy")
    parser.add_argument("--y-test", required=True, help="Output y_test_cat.npy")
    parser.add_argument("--test-size", type=float, default=0.20,
                        help="Test split ratio (default: 0.20)")
    parser.add_argument("--n-classes", type=int, default=0,
                        help="Number of segmentation classes (0 = auto-detect from data)")
    parser.add_argument("--metadata", default=None,
                        help="Output metadata JSON (n_classes, label mapping)")
    parser.add_argument("--cloud-fraction", action="append", default=None,
                        help="Per-scene cloud/shadow fraction JSON (one --cloud-fraction "
                             "per scene). When set, the test split's cloud fractions "
                             "are emitted alongside X_test as test_cloud_fractions.npy "
                             "so evaluate_stratified can bucket tiles by paper's 10% "
                             "threshold (Table V / Fig 13 stratified panels).")
    parser.add_argument("--test-cloud-fractions", default=None,
                        help="Output .npy aligned with X_test containing the test "
                             "tiles' cloud/shadow fractions (required when "
                             "--cloud-fraction is passed).")
    parser.add_argument("--random-state", type=int, default=0,
                        help="Random seed (default: 0)")
    args = parser.parse_args()

    image_paths = sorted(args.image)
    mask_paths = sorted(args.mask)

    if len(image_paths) != len(mask_paths):
        logger.error(
            f"Image/mask count mismatch: {len(image_paths)} images vs "
            f"{len(mask_paths)} masks. They must be equal."
        )
        sys.exit(1)

    logger.info(f"Dataset: {len(image_paths)} images, {len(mask_paths)} masks")

    # ── Fit LabelEncoder on ALL masks (need to see every class) ──
    # Load masks as uint8 — small memory footprint
    all_masks = load_images(mask_paths)
    if len(all_masks) == 0:
        logger.error("No masks loaded")
        sys.exit(1)

    labelencoder = LabelEncoder()
    labelencoder.fit(all_masks.reshape(-1, 1))
    detected_classes = len(labelencoder.classes_)
    logger.info(f"Label classes: {labelencoder.classes_} ({detected_classes} unique)")
    del all_masks  # free — we'll reload per-split below

    # Determine n_classes: auto-detect (0) or user-specified
    if args.n_classes <= 0:
        n_classes = detected_classes
        logger.info(f"Auto-detected n_classes={n_classes}")
    else:
        if detected_classes > args.n_classes:
            logger.warning(
                f"Data has {detected_classes} unique classes but --n-classes={args.n_classes}. "
                f"Using {detected_classes} to avoid index errors."
            )
            n_classes = detected_classes
        else:
            n_classes = args.n_classes

    # Save metadata for downstream jobs (train_unet, evaluate_model)
    if args.metadata:
        meta = {
            "n_classes": n_classes,
            "label_classes": labelencoder.classes_.tolist(),
        }
        with open(args.metadata, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Metadata saved to {args.metadata}")

    # ── Optional cloud/shadow fractions for stratified evaluation ──
    # Build a single lookup keyed by tile basename (sans ".png") from one or
    # more per-scene JSON files (see bin/compute_cloud_fraction.py).
    cloud_lookup = {}
    if args.cloud_fraction:
        for cf_path in args.cloud_fraction:
            with open(cf_path) as f:
                cloud_lookup.update(json.load(f))
        logger.info(f"Loaded cloud fractions for {len(cloud_lookup)} tiles "
                    f"from {len(args.cloud_fraction)} JSON file(s).")
        if not args.test_cloud_fractions:
            logger.error("--cloud-fraction requires --test-cloud-fractions "
                         "to know where to write the aligned test array.")
            sys.exit(1)

    def _tile_cloud_fraction(image_path):
        """Look up the cloud/shadow fraction for a training-image tile.

        Training image tiles are named ``train_img_<scene>_<row>_<col>.png``
        (image_split's ``--output-prefix train_img_<scene>``). Cloud
        fractions are keyed by the equivalent ``<scene>_<row>_<col>``.
        """
        base = os.path.splitext(os.path.basename(image_path))[0]
        # Strip the train_img_ / train_imgf_ prefix added by image_split when
        # auto-labeling. compute_cloud_fraction always keys by the SCENE
        # name (not the filtered-scene name), so for the filtered branch we
        # strip ``train_imgf_`` and look up the matching raw-scene fraction.
        for prefix in ("train_img_", "train_imgf_"):
            if base.startswith(prefix):
                base = base[len(prefix):]
                break
        return cloud_lookup.get(base)

    # ── Split file-path indices (no image data in memory) ──
    n = len(image_paths)
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=args.random_state,
    )
    logger.info(f"Split: {len(train_idx)} train, {len(test_idx)} test")

    if cloud_lookup:
        # Align test-split cloud fractions in the same order as X_test.
        # Missing entries default to -1 so downstream code can filter them out
        # (and so callers can spot the misalignment loudly).
        test_fracs = np.array(
            [_tile_cloud_fraction(image_paths[i]) or -1.0 for i in test_idx],
            dtype=np.float32,
        )
        missing = int((test_fracs < 0).sum())
        if missing:
            logger.warning(
                f"{missing}/{len(test_fracs)} test tiles had no cloud-fraction "
                "entry; they will be excluded from stratified evaluation.")
        out_dir = os.path.dirname(args.test_cloud_fractions)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.save(args.test_cloud_fractions, test_fracs)
        logger.info(f"Saved {args.test_cloud_fractions} "
                    f"(shape {test_fracs.shape}).")

    # ── Process and save TRAIN split ──
    logger.info("Processing train split...")
    train_imgs = load_images([image_paths[i] for i in train_idx])
    X_train = normalize_float32(train_imgs)  # consumes train_imgs
    np.save(args.x_train, X_train)
    logger.info(f"X_train shape: {X_train.shape}")
    del X_train

    train_msks = load_images([mask_paths[i] for i in train_idx])
    y_train = encode_masks(train_msks, labelencoder, n_classes)
    del train_msks
    np.save(args.y_train, y_train)
    logger.info(f"y_train shape: {y_train.shape}")
    del y_train

    # ── Process and save TEST split ──
    logger.info("Processing test split...")
    test_imgs = load_images([image_paths[i] for i in test_idx])
    X_test = normalize_float32(test_imgs)  # consumes test_imgs
    np.save(args.x_test, X_test)
    logger.info(f"X_test shape: {X_test.shape}")
    del X_test

    test_msks = load_images([mask_paths[i] for i in test_idx])
    y_test = encode_masks(test_msks, labelencoder, n_classes)
    del test_msks
    np.save(args.y_test, y_test)
    logger.info(f"y_test shape: {y_test.shape}")
    del y_test

    logger.info(f"Saved: {args.x_train}, {args.x_test}, {args.y_train}, {args.y_test}")


if __name__ == "__main__":
    main()
