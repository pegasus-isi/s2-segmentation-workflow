#!/usr/bin/env python3

"""Download Sentinel-2 sea ice imagery from Google Earth Engine.

Downloads RGB (B4/B3/B2) scenes from the Antarctic Ross Sea region
for November 2019 and exports them as PNG tiles for the workflow.

Prerequisites:
    pip install earthengine-api
    earthengine authenticate

Reference:
    Iqrah et al., "A Parallel Workflow for Polar Sea-Ice Classification
    using Auto-Labeling of Sentinel-2 Imagery," IPDPSW 2024.

Data parameters from the paper:
    - Region: Ross Sea, Antarctica
    - Latitude: -70.00 to -78.00 (south)
    - Longitude: -140.00 to -180.00 (west)
    - Time: November 2019 (Antarctic summer)
    - Bands: B4 (red), B3 (green), B2 (blue) at 10m resolution
    - Cloud filter: <20% scene cloud coverage
    - 66 large scenes -> split into 256x256 tiles for training
"""

import argparse
import logging
import math
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    import ee
except ImportError:
    logger.error(
        "Google Earth Engine API not installed.\n"
        "Install with: pip install earthengine-api\n"
        "Then authenticate: earthengine authenticate"
    )
    sys.exit(1)


# Ross Sea region of interest (from the paper)
ROI_COORDS = [
    [-180.00, -70.00],
    [-140.00, -70.00],
    [-140.00, -78.00],
    [-180.00, -78.00],
]

# Sentinel-2 collection and parameters
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
START_DATE = "2019-11-01"
END_DATE = "2019-11-30"
CLOUD_FILTER = 20  # max cloud cover percentage
BANDS = ["B4", "B3", "B2"]  # RGB
SCALE = 10  # meters per pixel


def get_s2_collection(roi, start_date, end_date, cloud_pct):
    """Filter Sentinel-2 collection for the region and time period."""
    return (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .select(BANDS)
    )


def export_scenes_to_drive(collection, roi, output_folder, max_scenes=None):
    """Export individual scenes to Google Drive as GeoTIFF."""
    image_list = collection.toList(collection.size())
    count = collection.size().getInfo()

    if max_scenes:
        count = min(count, max_scenes)

    logger.info(f"Found {collection.size().getInfo()} scenes, exporting {count}")

    tasks = []
    for i in range(count):
        image = ee.Image(image_list.get(i))
        scene_id = image.get("system:index").getInfo()
        safe_name = f"s2_vis_{i:02d}"

        task = ee.batch.Export.image.toDrive(
            image=image.toUint8(),
            description=safe_name,
            folder=output_folder,
            region=roi,
            scale=SCALE,
            maxPixels=1e10,
            fileFormat="GeoTIFF",
        )
        task.start()
        tasks.append((safe_name, task))
        logger.info(f"  [{i+1}/{count}] Started export: {safe_name} ({scene_id})")

    return tasks


def export_scenes_to_local(collection, roi, output_dir, tile_size=2048,
                           max_scenes=None):
    """Download scenes directly as NumPy arrays and save as PNG tiles.

    This method downloads smaller tiles to avoid GEE memory limits.
    Each scene is split into tile_size x tile_size pixel tiles.
    """
    import numpy as np
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    image_list = collection.toList(collection.size())
    count = collection.size().getInfo()

    if max_scenes:
        count = min(count, max_scenes)

    logger.info(f"Found {collection.size().getInfo()} scenes, downloading {count}")

    roi_info = roi.getInfo()
    coords = roi_info["coordinates"][0]
    min_lon = min(c[0] for c in coords)
    max_lon = max(c[0] for c in coords)
    min_lat = min(c[1] for c in coords)
    max_lat = max(c[1] for c in coords)

    for i in range(count):
        image = ee.Image(image_list.get(i))
        scene_id = image.get("system:index").getInfo()
        safe_name = f"s2_vis_{i:02d}"

        logger.info(f"  [{i+1}/{count}] Downloading: {safe_name} ({scene_id})")

        try:
            # Get the scene footprint for tighter bounds
            footprint = image.geometry()

            # Download as numpy array via getThumbURL or computePixels
            url = image.getThumbURL({
                "bands": BANDS,
                "min": 0,
                "max": 3000,
                "region": footprint,
                "dimensions": f"{tile_size}x{tile_size}",
                "format": "png",
            })

            import urllib.request
            out_path = os.path.join(output_dir, f"{safe_name}.png")
            urllib.request.urlretrieve(url, out_path)
            logger.info(f"    Saved: {out_path}")

        except Exception as e:
            logger.warning(f"    Failed to download {safe_name}: {e}")
            continue

    logger.info(f"Downloaded scenes to {output_dir}")


def split_into_training_tiles(input_dir, output_images_dir, output_masks_dir,
                               tile_size=256):
    """Split large scene PNGs into 256x256 tiles for training.

    This creates the train_images/ directory expected by preprocess_data.py.
    Masks are not generated here — they come from the color segmentation
    stage of the workflow (Stage 1).
    """
    import numpy as np
    from PIL import Image

    os.makedirs(output_images_dir, exist_ok=True)

    tile_count = 0
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".png"):
            continue

        img_path = os.path.join(input_dir, fname)
        img = np.array(Image.open(img_path).convert("L"))  # grayscale
        h, w = img.shape

        basename = os.path.splitext(fname)[0]
        for r in range(0, h - tile_size + 1, tile_size):
            for c in range(0, w - tile_size + 1, tile_size):
                tile = img[r:r + tile_size, c:c + tile_size]
                if tile.shape == (tile_size, tile_size):
                    out_path = os.path.join(
                        output_images_dir,
                        f"{basename}_{r:04d}_{c:04d}.png"
                    )
                    Image.fromarray(tile).save(out_path)
                    tile_count += 1

    logger.info(f"Created {tile_count} training tiles in {output_images_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 sea ice data from Google Earth Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download scenes to Google Drive (recommended for large exports)
  python download_data.py --method drive --drive-folder s2_ross_sea

  # Download scenes directly as PNG files
  python download_data.py --method local --output-dir data/s2_scenes

  # Download and split into 256x256 training tiles
  python download_data.py --method local --output-dir data/s2_scenes --split-tiles

  # Limit to 10 scenes for testing
  python download_data.py --method local --output-dir data/s2_scenes --max-scenes 10

  # Custom region and date range
  python download_data.py --method local --output-dir data/custom \\
      --start-date 2020-01-01 --end-date 2020-01-31 --cloud-pct 10
""",
    )

    parser.add_argument(
        "--method", choices=["drive", "local"], default="local",
        help="Export method: 'drive' (Google Drive) or 'local' (direct download). "
             "Default: local",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/s2_scenes",
        help="Local output directory for downloaded scenes (default: data/s2_scenes)",
    )
    parser.add_argument(
        "--drive-folder", type=str, default="s2_ross_sea",
        help="Google Drive folder name for Drive exports (default: s2_ross_sea)",
    )
    parser.add_argument(
        "--start-date", type=str, default=START_DATE,
        help=f"Start date YYYY-MM-DD (default: {START_DATE})",
    )
    parser.add_argument(
        "--end-date", type=str, default=END_DATE,
        help=f"End date YYYY-MM-DD (default: {END_DATE})",
    )
    parser.add_argument(
        "--cloud-pct", type=int, default=CLOUD_FILTER,
        help=f"Max cloud cover percentage (default: {CLOUD_FILTER})",
    )
    parser.add_argument(
        "--max-scenes", type=int, default=None,
        help="Max number of scenes to download (default: all)",
    )
    parser.add_argument(
        "--scene-size", type=int, default=2048,
        help="Scene download size in pixels (default: 2048, the authors' "
             "native scene geometry — divides evenly by 256, so no in-DAG "
             "resize and no padded edge tiles)",
    )
    parser.add_argument(
        "--split-tiles", action="store_true",
        help="Also split downloaded scenes into 256x256 training tiles",
    )
    parser.add_argument(
        "--tile-size", type=int, default=256,
        help="Training tile size in pixels (default: 256)",
    )
    parser.add_argument(
        "--project", type=str, default=None,
        help="Google Cloud project ID registered for Earth Engine "
             "(e.g. 'ee-myproject'). Register at: "
             "https://console.cloud.google.com/earth-engine",
    )

    args = parser.parse_args()

    # Initialize Earth Engine
    logger.info("Initializing Google Earth Engine...")
    init_kwargs = {}
    if args.project:
        init_kwargs["project"] = args.project

    try:
        ee.Initialize(**init_kwargs)
    except Exception as init_err:
        if "not registered" in str(init_err):
            project = args.project or "(default)"
            logger.error(
                f"\nProject {project} is not registered for Earth Engine.\n"
                f"Register at: https://console.cloud.google.com/earth-engine\n"
                f"Then re-run with: --project YOUR_PROJECT_ID\n"
            )
            sys.exit(1)
        logger.info(
            "Earth Engine not authenticated. Opening browser for authentication...\n"
            "This uses a browser-based OAuth flow (no gcloud CLI required)."
        )
        ee.Authenticate(auth_mode="notebook")
        ee.Initialize(**init_kwargs)

    # Define region of interest
    roi = ee.Geometry.Polygon([ROI_COORDS])
    logger.info(f"Region: Ross Sea, Antarctica")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Cloud filter: <{args.cloud_pct}%")

    # Get filtered collection
    collection = get_s2_collection(roi, args.start_date, args.end_date, args.cloud_pct)
    total = collection.size().getInfo()
    logger.info(f"Total matching scenes: {total}")

    if total == 0:
        logger.error("No scenes found for the specified parameters")
        sys.exit(1)

    # Export
    if args.method == "drive":
        logger.info(f"Exporting to Google Drive folder: {args.drive_folder}")
        tasks = export_scenes_to_drive(
            collection, roi, args.drive_folder, args.max_scenes,
        )
        logger.info(
            f"\n{len(tasks)} export tasks started. Check progress at:\n"
            "  https://code.earthengine.google.com/tasks\n\n"
            "Once complete, download from Google Drive and place in data/s2_scenes/"
        )
    else:
        logger.info(f"Downloading to: {args.output_dir}")
        export_scenes_to_local(
            collection, roi, args.output_dir,
            tile_size=args.scene_size,
            max_scenes=args.max_scenes,
        )

    # Optionally split into training tiles
    if args.split_tiles and args.method == "local":
        train_images_dir = os.path.join(args.output_dir, "..", "train_images")
        train_masks_dir = os.path.join(args.output_dir, "..", "train_masks")
        logger.info(f"Splitting scenes into {args.tile_size}x{args.tile_size} tiles...")
        split_into_training_tiles(
            args.output_dir, train_images_dir, train_masks_dir, args.tile_size,
        )
        logger.info(
            f"\nTraining tiles saved to: {train_images_dir}\n"
            "Note: Training masks must be generated by running Stage 1 "
            "(color segmentation) of the workflow."
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
