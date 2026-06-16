#!/usr/bin/env python3
"""One-shot recovery for run0002's broken stratified-eval inputs.

run0002's compute_cloud_fraction keyed tiles by the resized-scene filename
(``resized_s2_vis_NN_r_c``) instead of the scene name (``s2_vis_NN_r_c``),
so preprocess_data could not match any test tile to its cloud fraction and
wrote an all-(-1) test_cloud_fractions array. The fraction VALUES are
correct — only the JSON keys carry a spurious ``resized_`` prefix.

This script rebuilds ``test_cloud_fractions_<branch>.npy`` by replaying
preprocess_data.py's exact logic: the same ordered --image list (parsed
from the job's .sh), the same train_test_split, and the same per-tile
lookup — but against a key-corrected cloud lookup. It does NOT recompute
any fractions or touch X_test/y_test.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from sklearn.model_selection import train_test_split


def parse_image_list(sh_path):
    """Return the ordered list of --image values from a preprocess .sh."""
    text = open(sh_path).read()
    # Matches both "--image foo.png" and "--image\nfoo.png"
    imgs = re.findall(r"--image\s+(\S+)", text)
    return imgs


def build_lookup(json_dir):
    """Cloud-fraction lookup keyed by <scene>_<row>_<col> (resized_ stripped)."""
    lookup = {}
    for cf in sorted(glob.glob(os.path.join(json_dir, "cloud_fraction_*.json"))):
        for k, v in json.load(open(cf)).items():
            if k.startswith("resized_"):
                k = k[len("resized_"):]
            lookup[k] = v
    return lookup


def tile_key(image_path):
    """Mirror preprocess_data._tile_cloud_fraction's key derivation."""
    base = os.path.splitext(os.path.basename(image_path))[0]
    for prefix in ("train_img_", "train_imgf_"):
        if base.startswith(prefix):
            return base[len(prefix):]
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sh", required=True, help="preprocess job .sh path")
    ap.add_argument("--json-dir", required=True, help="dir with cloud_fraction_*.json")
    ap.add_argument("--output", required=True, help="test_cloud_fractions_<branch>.npy")
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--random-state", type=int, default=0)
    args = ap.parse_args()

    image_paths = parse_image_list(args.sh)
    n = len(image_paths)
    if n == 0:
        sys.exit(f"No --image entries parsed from {args.sh}")

    lookup = build_lookup(args.json_dir)
    print(f"Parsed {n} image tiles; cloud lookup has {len(lookup)} keys.")

    _, test_idx = train_test_split(
        np.arange(n), test_size=args.test_size, random_state=args.random_state)

    test_fracs = np.array(
        [lookup.get(tile_key(image_paths[i]), -1.0) for i in test_idx],
        dtype=np.float32,
    )
    valid = int((test_fracs >= 0).sum())
    print(f"Test tiles: {len(test_fracs)}; valid fractions: {valid}; "
          f"missing: {len(test_fracs) - valid}")
    if valid == 0:
        sys.exit("ERROR: still 0 valid fractions — key mismatch not resolved.")

    np.save(args.output, test_fracs)
    print(f"Wrote {args.output} (shape {test_fracs.shape}, "
          f"high>=0.1: {int((test_fracs >= 0.1).sum())}, "
          f"low<0.1: {int(((test_fracs >= 0) & (test_fracs < 0.1)).sum())})")


if __name__ == "__main__":
    main()
