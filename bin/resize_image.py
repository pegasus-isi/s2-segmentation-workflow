#!/usr/bin/env python3

"""Resize a scene PNG to a square target size.

The reference paper uses 2048x2048 Sentinel-2 scenes (66 scenes -> 4224
tiles of 256x256, no padding). GEE exports of the same region come out
2000x2000, which does not tile evenly by 256 and forces zero-padding —
the padding then pollutes the auto-labels with a spurious 4th class.
Resizing each scene to 2048x2048 before any tiling reproduces the
paper's geometry exactly.

Masks are never resized: labels are always derived *after* resizing.
"""

import argparse
import logging
import sys

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Resize a scene to NxN")
    parser.add_argument("--input", required=True, help="Input scene PNG")
    parser.add_argument("--output", required=True, help="Output resized PNG")
    parser.add_argument("--size", type=int, required=True,
                        help="Target square size in pixels (e.g. 2048)")
    parser.add_argument("--grayscale", action="store_true",
                        help="Read image as grayscale (single channel)")
    args = parser.parse_args()

    read_flag = cv2.IMREAD_GRAYSCALE if args.grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(args.input, read_flag)
    if img is None:
        logger.error(f"Failed to read image: {args.input}")
        sys.exit(1)

    h, w = img.shape[:2]
    if (h, w) == (args.size, args.size):
        logger.info(f"{args.input} already {args.size}x{args.size}; copying")
        cv2.imwrite(args.output, img)
        return

    interp = cv2.INTER_CUBIC if args.size > max(h, w) else cv2.INTER_AREA
    resized = cv2.resize(img, (args.size, args.size), interpolation=interp)
    cv2.imwrite(args.output, resized)
    logger.info(f"Resized {args.input} {w}x{h} -> {args.size}x{args.size}")


if __name__ == "__main__":
    main()
