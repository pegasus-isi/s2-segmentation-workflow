#!/usr/bin/env python3

"""Compute per-tile cloud / shadow coverage fraction for a Sentinel-2 scene.

Reproduces the paper's cloud-coverage stratification (Table V and the
per-stratum panels of Fig 13) by reusing the same Otsu-thresholded
cloud/shadow mask that ``only_shadow_cloud_removal`` produces as an
intermediate step. The cloud mask is computed once on the full scene,
then averaged within each 256x256 tile (matching ``image_split --pad``)
to give a per-tile fraction in [0, 1].

The resulting JSON maps tile *basename* (without the extension) to its
cloud/shadow fraction:

    {
      "s2_vis_00_0000_0000": 0.034,
      "s2_vis_00_0000_0256": 0.211,
      ...
    }

Downstream, ``preprocess_data`` aligns these fractions to the test split
so ``evaluate_stratified`` can bucket tiles by paper's 10% threshold.
"""

import argparse
import json
import logging
import os
import sys

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def cloud_shadow_mask(rgb, kernel_size=155):
    """Return a boolean cloud/shadow mask using the same pipeline as filter_image.

    The mask is True where the pixel is considered cloud or shadow. We
    stop at the Otsu binary stage of ``only_shadow_cloud_removal`` — that
    output (``outs2``) already separates clear ground (255) from
    cloud/shadow regions (0). ``kernel_size`` mirrors filter_image's
    medianBlur kernel so the two pipelines stay in lockstep.
    """
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be odd and >= 3 (got {kernel_size}).")

    # Mask water first so it doesn't bias the cloud-detection.
    lower_water = (0, 0, 0)
    upper_water = (185, 255, 30)
    hsv_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask_water = cv2.inRange(hsv_img, lower_water, upper_water)

    without_water = rgb.copy()
    without_water[mask_water == 255] = [255, 255, 255]
    img = cv2.cvtColor(without_water, cv2.COLOR_RGB2GRAY)

    dilated = cv2.dilate(img, np.ones((7, 7), np.uint8))
    bg = cv2.medianBlur(dilated, kernel_size)
    diff = 255 - cv2.absdiff(img, bg)

    # Otsu binary threshold: high values = clear, low values = cloud/shadow.
    _, outs2 = cv2.threshold(
        src=diff, thresh=0, maxval=255,
        type=cv2.THRESH_OTSU + cv2.THRESH_BINARY,
    )
    # Cloud/shadow where outs2 is 0. Exclude water from the cloud mask so
    # the fraction stays interpretable on coastal/sea tiles.
    cloud = (outs2 == 0) & (mask_water != 255)
    return cloud


def per_tile_fractions(mask, tile_size, basename):
    """Average the cloud/shadow mask over each tile_size x tile_size region.

    Edge tiles that fall outside the scene are zero-padded (matching
    ``image_split --pad``), so the fraction is computed only over the
    real pixels in that tile.
    """
    h, w = mask.shape
    out = {}
    for r in range(0, h, tile_size):
        for c in range(0, w, tile_size):
            patch = mask[r:r + tile_size, c:c + tile_size]
            n = patch.size
            if n == 0:
                continue
            frac = float(np.count_nonzero(patch)) / float(n)
            key = f"{basename}_{str(r).zfill(4)}_{str(c).zfill(4)}"
            out[key] = round(frac, 6)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Per-tile cloud/shadow fractions for one Sentinel-2 scene")
    parser.add_argument("--input", required=True, help="Input scene PNG (colour)")
    parser.add_argument("--output", required=True,
                        help="Output JSON with per-tile cloud/shadow fractions")
    parser.add_argument("--tile-size", type=int, default=256,
                        help="Tile size to average over (default: 256, matches "
                             "training tiles).")
    parser.add_argument("--kernel-size", type=int, default=155,
                        help="medianBlur kernel for background estimation "
                             "(default: 155, matches filter_image's default).")
    parser.add_argument("--key-prefix", default=None,
                        help="Tile-key prefix in the output JSON (default: the "
                             "input filename stem). Pass the original SCENE name "
                             "when --input is a resized/derived file, so keys "
                             "match the training tiles' <scene>_<row>_<col>.")
    args = parser.parse_args()

    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    raw = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if raw is None:
        logger.error(f"Failed to read scene: {args.input}")
        sys.exit(1)

    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    mask = cloud_shadow_mask(rgb, kernel_size=args.kernel_size)
    logger.info(
        f"Cloud/shadow pixels: {int(mask.sum())} / {mask.size} "
        f"({mask.mean() * 100:.2f}% of scene)")

    basename = args.key_prefix or os.path.splitext(
        os.path.basename(args.input))[0]
    fractions = per_tile_fractions(mask, args.tile_size, basename)
    logger.info(
        f"Tiles: {len(fractions)} (max frac {max(fractions.values(), default=0):.3f}, "
        f"mean {np.mean(list(fractions.values())):.3f})")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(fractions, f, indent=2, sort_keys=True)
    logger.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
