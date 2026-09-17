#!/usr/bin/env bash
# Fetch and unpack the S2 sea-ice dataset into the layout the workflow expects.
#
# Usage:
#   ./prepare_author_data.sh                 # unpack archives already in data/
#   S2_DATA_URL=<base-url> ./prepare_author_data.sh   # download them first
#
# S2_URL_SUFFIX appends a query string to each file URL, for hosts that need one
# (e.g. Zenodo: S2_URL_SUFFIX='?download=1').
#   WITH_TIFF=1 ./prepare_author_data.sh     # also stage the 8.3 GB GeoTIFFs
#
# Archives:
#   s2_original_2048.zip   66 scenes, 2048x2048 RGB PNG  (333 MB)  -> workflow input
#   S2_data_training.zip   4032 image + 4032 mask tiles  (341 MB)  -> label-validation ref
#   S2_tiff.zip            52 source GeoTIFFs            (8.3 GB)  -> provenance only
#
# NOTE: these are ZIP64 archives. macOS's bundled `unzip` cannot read them
# ("start of central directory not found"); Python's zipfile can, so it is
# used throughout.
set -euo pipefail

cd "$(dirname "$0")"
DATA_DIR="${DATA_DIR:-data}"
mkdir -p "$DATA_DIR"

# Publish the archives somewhere with direct, unauthenticated HTTP downloads
# (Zenodo is recommended — see README "Getting the Dataset") and set
# S2_DATA_URL to the base URL that serves the three files above.
S2_DATA_URL="${S2_DATA_URL:-}"

stage() {  # stage <canonical zip name> <expected top-level dir>
    local name="$1" target="$DATA_DIR/$2" zip dest
    if [[ -d "$target" ]]; then
        echo "  have: $target ($(find "$target" -type f | wc -l | tr -d ' ') files)"
        return 0
    fi
    # Accept the canonical name or any Drive-mangled variant (foo-2026...-001.zip)
    zip=$(ls "$DATA_DIR/$name" "$DATA_DIR/${name%.zip}"-*.zip 2>/dev/null | head -1 || true)
    if [[ -z "$zip" ]]; then
        if [[ -z "$S2_DATA_URL" ]]; then
            echo "  MISSING: $DATA_DIR/$name"
            echo "           set S2_DATA_URL=<base-url> to download it, or place it there by hand"
            return 1
        fi
        dest="$DATA_DIR/$name"
        echo "  downloading $name ..."
        curl -fL --retry 3 --retry-delay 5 -o "$dest.part" \
            "${S2_DATA_URL%/}/${name}${S2_URL_SUFFIX:-}"
        mv "$dest.part" "$dest"
        zip="$dest"
    fi
    echo "  unpacking $(basename "$zip") -> $target"
    python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
        "$zip" "$DATA_DIR"
}

echo "Staging dataset in $DATA_DIR/"
missing=0
stage s2_original_2048.zip s2_original_2048 || missing=1
stage S2_data_training.zip S2_data_training || missing=1
if [[ "${WITH_TIFF:-0}" == "1" ]]; then
    stage S2_tiff.zip S2_tiff || missing=1
else
    echo "  skip: S2_tiff.zip (8.3 GB provenance only — WITH_TIFF=1 to stage it)"
fi

# --- verify ---
scenes=$(ls "$DATA_DIR"/s2_original_2048/s2_vis_*.png 2>/dev/null | wc -l | tr -d ' ')
echo
echo "Scenes staged: $scenes (expected 66)"
[[ "$scenes" == "66" ]] || { echo "WARNING: expected 66 scenes"; missing=1; }

if [[ "$missing" != "0" ]]; then
    echo "Dataset incomplete — see messages above."
    exit 1
fi

cat <<MSG

Ready. Reproduce the paper (66 scenes / 4,224 tiles) with:

  python workflow_generator.py \\
      --images $DATA_DIR/s2_original_2048/s2_vis_*.png \\
      --scene-size 0 \\
      --output workflow_A.yml

--scene-size 0 skips the resize: these scenes are natively 2048x2048, so
nothing is resampled.

The authors' own tiles/labels (63 of the 66 scenes) are a validation
reference, not training input:
  $DATA_DIR/S2_data_training/{train_images_4032,train_masks_4032}/
MSG
