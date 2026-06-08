#!/usr/bin/env python3

"""Thin-cloud and shadow filter for Sentinel-2 scenes.

Reproduces the paper's `only_shadow_cloud_removal` preprocessing
(Iqrah et al., IPDPSW 2024 — "A Parallel Workflow for Polar Sea-Ice
Classification using Auto-labeling of Sentinel-2 Imagery"). The filter
removes thin clouds and shadows so the downstream U-Net sees a cleaned
3-level (water / thin-ice / ice) grayscale image.

Pipeline (OpenCV), matching the paper's described techniques:
  RGB->HSV, water masking, RGB->GRAY, dilation + large median blur for
  background estimation, absolute difference, Otsu binary thresholding,
  bitwise-and, min-max normalization, truncated thresholding, and a
  final HSV re-thresholding into the three sea-ice classes.

Input is a colour Sentinel-2 scene; output is a single-channel grayscale
PNG (the cleaned image fed to the U-Net), so it can be tiled by
image_split --grayscale exactly like the unfiltered path.
"""

import argparse
import logging
import sys

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def only_shadow_cloud_removal(ori):
    """Filter thin clouds/shadows from an RGB image.

    Verbatim reproduction of the reference implementation used to
    produce the paper's thin-cloud/shadow-filtered results. Returns a
    3-channel image whose channels are equal (ternary 0/155/255 values),
    so channel 0 is the cleaned grayscale signal.
    """
    # --- separate open water ---
    lower_water = (0, 0, 0)
    upper_water = (185, 255, 30)
    hsv_img = cv2.cvtColor(ori, cv2.COLOR_RGB2HSV)
    mask_water = cv2.inRange(hsv_img, lower_water, upper_water)

    without_water_img = ori.copy()
    without_water_img[mask_water == 255] = [255, 255, 255]

    img = cv2.cvtColor(without_water_img, cv2.COLOR_RGB2GRAY)

    # --- background estimation + shadow/cloud subtraction ---
    dilated_img = cv2.dilate(img, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 155)
    diff_img = 255 - cv2.absdiff(img, bg_img)

    _, outs2 = cv2.threshold(
        src=diff_img, thresh=0, maxval=255,
        type=cv2.THRESH_OTSU + cv2.THRESH_BINARY,
    )
    diff_img2 = cv2.bitwise_and(diff_img, outs2)

    norm_img = cv2.normalize(
        diff_img2, None, alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1,
    )
    _, thr_img = cv2.threshold(norm_img, 235, 0, cv2.THRESH_TRUNC)
    thr_img = cv2.normalize(
        thr_img, None, alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1,
    )

    # --- separate thin and old ice ---
    old_thin_ice = cv2.cvtColor(thr_img, cv2.COLOR_GRAY2RGB)
    hsv_img = cv2.cvtColor(old_thin_ice, cv2.COLOR_RGB2HSV)

    lower_tice = (0, 0, 0)
    upper_tice = (185, 255, 204)
    mask_tice = cv2.inRange(hsv_img, lower_tice, upper_tice)

    lower_ice = (0, 0, 205)
    upper_ice = (185, 255, 255)
    mask_ice = cv2.inRange(hsv_img, lower_ice, upper_ice)
    mask_ice = cv2.bitwise_xor(mask_water, mask_ice)

    shadow_free = old_thin_ice.copy()
    shadow_free[mask_ice == 255] = [255, 255, 255]
    shadow_free[mask_tice == 255] = [155, 155, 155]
    shadow_free[mask_water == 255] = [0, 0, 0]
    shadow_free = cv2.cvtColor(shadow_free, cv2.COLOR_BGR2RGB)

    return shadow_free


def main():
    parser = argparse.ArgumentParser(
        description="Apply thin-cloud/shadow filter to a Sentinel-2 scene")
    parser.add_argument("--input", required=True, help="Input scene PNG (colour)")
    parser.add_argument("--output", required=True,
                        help="Output filtered grayscale PNG")
    args = parser.parse_args()

    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        logger.error(f"Failed to read image: {args.input}")
        sys.exit(1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    filtered = only_shadow_cloud_removal(img_rgb)

    # Channels are equal (ternary cleaned signal); persist channel 0 as
    # single-channel grayscale so image_split --grayscale tiles it directly.
    gray = filtered[:, :, 0]
    cv2.imwrite(args.output, gray)

    logger.info(f"Filtering complete: {args.output}")


if __name__ == "__main__":
    main()
