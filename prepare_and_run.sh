#!/usr/bin/env bash
#
# prepare_and_run.sh — Prepare local Sentinel-2 scenes and run the
# s2-segmentation workflow.
#
# This script does NOT download anything. It assumes the Sentinel-2 scene
# PNGs have already been placed in data/s2_scenes/ (transferred manually,
# or produced by download_data.py). It only prepares the local data and
# launches the workflow.
#
# By default --auto-label trains BOTH the unfiltered and the
# thin-cloud/shadow-filtered U-Net (paper Table IV) from a single DAG.
#
# Usage:
#   ./prepare_and_run.sh
#
set -euo pipefail

WORKFLOW_DIR="$HOME/s2-segmentation-workflow"
DATA_DIR="${WORKFLOW_DIR}/data/s2_scenes"

info()  { echo -e "\033[1;34m[INFO]\033[0m  $*" >&2; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*" >&2; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; exit 1; }

check_prereqs() {
    command -v python3      >/dev/null || error "python3 not found"
    command -v pegasus-plan >/dev/null || error "pegasus-plan not found"
}

prepare_data() {
    info "Preparing local data in ${DATA_DIR}/ ..."
    [[ -d "${DATA_DIR}" ]] || error "Data directory not found: ${DATA_DIR} (place s2_vis_*.png there first)"

    # Flatten: move PNGs out of any subdirectories up to DATA_DIR
    find "${DATA_DIR}" -mindepth 2 -type f -name "*.png" -exec mv {} "${DATA_DIR}/" \; 2>/dev/null || true
    find "${DATA_DIR}" -mindepth 1 -type d -empty -delete 2>/dev/null || true

    local count
    count=$(find "${DATA_DIR}" -maxdepth 1 -name "s2_vis_*.png" | wc -l)
    [[ "${count}" -gt 0 ]] || error "No s2_vis_*.png files found in ${DATA_DIR}/"

    # Detect image dimensions — print only the number to stdout
    local dims
    dims=$(python3 -c "
from PIL import Image
import glob
imgs = sorted(glob.glob('${DATA_DIR}/s2_vis_*.png'))
print(Image.open(imgs[0]).size[0])
")
    info "Found ${count} scene images (${dims}x${dims} pixels)"
    echo "${dims}"
}

run_workflow() {
    local img_size="$1"
    cd "${WORKFLOW_DIR}"

    # The generator defaults ARE the canonical paper reproduction:
    # scenes resized in-DAG to 2048x2048, 256x256 tiles, auto-label,
    # both training branches, stratified eval, whole-scene inference.
    # Native scene size (${img_size}) only matters with --scene-size 0.
    info "Generating Pegasus workflow (paper-default configuration, native=${img_size})..."
    python3 workflow_generator.py \
        --images "${DATA_DIR}"/s2_vis_*.png \
        --original-size "${img_size}" \
        --output workflow.yml

    info "Planning and submitting workflow..."
    pegasus-plan --submit \
        --sites condorpool \
        --output-sites local \
        workflow.yml 2>&1 | tee /tmp/pegasus_plan_output.txt

    local run_dir
    run_dir=$(grep -oP 'pegasus-status\s+\K\S+' /tmp/pegasus_plan_output.txt 2>/dev/null | head -1 || true)
    if [[ -n "${run_dir}" ]]; then
        info "Submitted!"
        info "Monitor:  pegasus-status ${run_dir}"
        info "Analyze:  pegasus-analyzer ${run_dir}"
        echo ""
        pegasus-status "${run_dir}" || true
    else
        warn "Check /tmp/pegasus_plan_output.txt for details."
    fi
}

main() {
    info "S2 Segmentation Workflow — Prepare & Run"
    info "========================================"
    check_prereqs
    local img_size
    img_size=$(prepare_data)
    run_workflow "${img_size}"
    info "Done."
}

main "$@"
