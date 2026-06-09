#!/usr/bin/env python3

"""
Pegasus workflow generator for Sentinel-2 sea ice segmentation.

Two-stage pipeline:
  Stage 1 — Color segmentation: split → segment tiles (N×64 parallel) → merge
  Stage 2 — U-Net training:     preprocess → train → evaluate

Usage:
    # Stage 1 only — produces 2000x2000 merged masks, NOT 256x256 tiles
    ./workflow_generator.py --images data/s2_scenes/s2_vis_*.png

    # Auto-label (recommended) — single DAG: source scenes are split into
    # 256x256 grayscale tiles for both training images and masks (via Stage 1
    # color segmentation). No external data directories needed.
    ./workflow_generator.py --images data/s2_scenes/s2_vis_*.png --auto-label

    # Both stages with pre-existing 256x256 mask tiles (e.g. external labels)
    ./workflow_generator.py --images data/s2_scenes/s2_vis_*.png \\
        --train-images-dir data/train_images/ \\
        --train-masks-dir data/train_masks/

    # Horovod training (with pre-existing masks)
    ./workflow_generator.py --images data/s2_scenes/s2_vis_*.png \\
        --train-images-dir data/train_images/ \\
        --train-masks-dir data/train_masks/ \\
        --training-mode horovod
"""

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

from Pegasus.api import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Per-tool resource configuration
TOOL_CONFIGS = {
    "image_split":     {"memory": "512 MB", "cores": 1},
    "color_segment":   {"memory": "256 MB", "cores": 1},
    "filter_image":    {"memory": "2 GB",   "cores": 1},
    "image_merge":     {"memory": "1 GB",   "cores": 1},
    "preprocess_data": {"memory": "14 GB",   "cores": 2},
    "train_unet":      {"memory": "14 GB",   "cores": 4, "gpus": 1},
    "evaluate_model":  {"memory": "4 GB",   "cores": 2, "gpus": 1},
    "generate_plots":  {"memory": "14 GB",  "cores": 2, "gpus": 1},
    "infer_unet":      {"memory": "8 GB",   "cores": 2, "gpus": 1},
}


class S2SegmentationWorkflow:
    """Pegasus workflow for Sentinel-2 sea ice segmentation."""

    wf = None
    sc = None
    tc = None
    rc = None
    props = None

    dagfile = None
    wf_dir = None
    shared_scratch_dir = None
    local_storage_dir = None
    wf_name = "s2_segmentation"

    def __init__(self, dagfile="workflow.yml"):
        self.dagfile = dagfile
        self.wf_dir = str(Path(__file__).parent.resolve())
        self.shared_scratch_dir = os.path.join(self.wf_dir, "scratch")
        self.local_storage_dir = os.path.join(self.wf_dir, "output")

    def write(self):
        if self.sc is not None:
            self.sc.write()
        self.props.write()
        self.rc.write()
        self.tc.write()
        self.wf.write(file=self.dagfile)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    def create_pegasus_properties(self):
        self.props = Properties()
        self.props["pegasus.transfer.threads"] = "16"
        self.props["pegasus.transfer.worker.package.strict"] = "false"

    # ------------------------------------------------------------------
    # Site Catalog
    # ------------------------------------------------------------------
    def create_sites_catalog(self, exec_site_name="condorpool",
                              gpu_site_name="gpu-condorpool"):
        self.sc = SiteCatalog()

        local = Site("local").add_directories(
            Directory(
                Directory.SHARED_SCRATCH, self.shared_scratch_dir
            ).add_file_servers(
                FileServer("file://" + self.shared_scratch_dir, Operation.ALL)
            ),
            Directory(
                Directory.LOCAL_STORAGE, self.local_storage_dir
            ).add_file_servers(
                FileServer("file://" + self.local_storage_dir, Operation.ALL)
            ),
        )

        exec_site = (
            Site(exec_site_name)
            .add_condor_profile(universe="vanilla")
            .add_pegasus_profile(style="condor")
        )

        gpu_site = (
            Site(gpu_site_name)
            .add_condor_profile(universe="vanilla")
            .add_pegasus_profile(style="condor")
        )

        self.sc.add_sites(local, exec_site, gpu_site)

    # ------------------------------------------------------------------
    # Transformation Catalog
    # ------------------------------------------------------------------
    def create_transformation_catalog(self, exec_site_name="condorpool",
                                       gpu_site_name="gpu-condorpool",
                                       container_image="kthare10/s2-segmentation:latest"):
        self.tc = TransformationCatalog()

        container = Container(
            "s2_container",
            container_type=Container.SINGULARITY,
            image=f"docker://{container_image}",
            image_site="docker_hub",
        )

        # CPU-bound transformations (Stage 1 + preprocess)
        cpu_tools = ["image_split", "color_segment", "filter_image",
                     "image_merge", "preprocess_data"]
        for tool_name in cpu_tools:
            config = TOOL_CONFIGS[tool_name]
            tx = Transformation(
                tool_name,
                site=exec_site_name,
                pfn=os.path.join(self.wf_dir, f"bin/{tool_name}.py"),
                is_stageable=True,
                container=container,
            ).add_pegasus_profile(
                memory=config["memory"],
                cores=config.get("cores", 1),
            )
            self.tc.add_transformations(tx)

        # GPU-bound transformations (train + evaluate + infer)
        # Registered on both exec and GPU sites so pegasus-plan can resolve
        # the PFN regardless of which site is passed via -s. HTCondor uses
        # request_gpus for placement on GPU-equipped nodes.
        gpu_tools = ["train_unet", "evaluate_model", "generate_plots",
                     "infer_unet"]
        for tool_name in gpu_tools:
            config = TOOL_CONFIGS[tool_name]
            pfn = os.path.join(self.wf_dir, f"bin/{tool_name}.py")
            tx = Transformation(
                tool_name,
                site=exec_site_name,
                pfn=pfn,
                is_stageable=True,
                container=container,
            ).add_pegasus_profile(
                memory=config["memory"],
                cores=config.get("cores", 1),
                gpus=config.get("gpus", 0),
            )
            if gpu_site_name != exec_site_name:
                gpu_ts = TransformationSite(
                    gpu_site_name,
                    pfn=pfn,
                    is_stageable=True,
                    container=container,
                ).add_pegasus_profile(
                    memory=config["memory"],
                    cores=config.get("cores", 1),
                    gpus=config.get("gpus", 0),
                )
                tx.add_sites(gpu_ts)
            self.tc.add_transformations(tx)

        self.tc.add_containers(container)

    # ------------------------------------------------------------------
    # Replica Catalog
    # ------------------------------------------------------------------
    def create_replica_catalog(self, args):
        self.rc = ReplicaCatalog()

        # Register source images (and any inference-only scenes not in the
        # training corpus, so --infer can run on fresh data).
        registered = set()
        all_scenes = list(args.images)
        if getattr(args, "infer", False) and args.infer_images:
            all_scenes += list(args.infer_images)
        for img_path in all_scenes:
            lfn = os.path.basename(img_path)
            if lfn in registered:
                continue
            self.rc.add_replica(
                "local",
                lfn,
                "file://" + os.path.abspath(img_path),
            )
            registered.add(lfn)

        # Register model.py as a support file (used by train + evaluate)
        model_py = os.path.join(self.wf_dir, "bin/model.py")
        self.rc.add_replica("local", "model.py", "file://" + model_py)

        # Register training data if Stage 2 is enabled (non-auto-label mode)
        # In auto-label mode, training images and masks are produced by
        # split_images / split_masks jobs within the DAG — nothing to register.
        if not args.auto_label and args.train_images_dir:
            for img_path in sorted(glob.glob(os.path.join(args.train_images_dir, "*.png"))):
                lfn = f"train_img_{os.path.basename(img_path)}"
                self.rc.add_replica("local", lfn, "file://" + os.path.abspath(img_path))

        if not args.auto_label and args.train_masks_dir:
            for mask_path in sorted(glob.glob(os.path.join(args.train_masks_dir, "*.png"))):
                lfn = f"train_mask_{os.path.basename(mask_path)}"
                self.rc.add_replica("local", lfn, "file://" + os.path.abspath(mask_path))

    # ------------------------------------------------------------------
    # Workflow DAG
    # ------------------------------------------------------------------
    def create_workflow(self, args):
        self.wf = Workflow(self.wf_name, infer_dependencies=True)

        tile_size = args.tile_size
        original_size = args.original_size
        grid = original_size // tile_size  # tiles per dimension

        # ============================================================
        # Stage 1: Color Segmentation (per-image fan-out / fan-in)
        # ============================================================
        # Auto-label tiles. Masks (labels) are shared across branches; the
        # U-Net input images come in an "orig" (raw scene) set and a "filt"
        # (thin-cloud/shadow-filtered scene) set. Which sets are populated
        # depends on the mode (unfiltered / filtered / both-paths).
        auto_label_image_tiles_orig = []  # raw-scene 256x256 image tiles
        auto_label_image_tiles_filt = []  # filtered-scene 256x256 image tiles
        auto_label_mask_tiles_orig = []   # labels: color-seg of the raw scene
        auto_label_mask_tiles_filt = []   # labels: color-seg of the filtered scene
        mask_tile_size = 256  # U-Net training tile size

        # Both the unfiltered and filtered training-image paths are built by
        # default (the paper compares both in Table IV); --paths restricts.
        build_orig = args.paths in ("both", "orig")
        build_filt = args.paths in ("both", "filtered")

        for img_path in args.images:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            input_file = File(os.path.basename(img_path))

            # --- Job: image_split ---
            tile_files = []
            for r in range(0, original_size, tile_size):
                for c in range(0, original_size, tile_size):
                    tile_lfn = f"{basename}_{str(r).zfill(4)}_{str(c).zfill(4)}.png"
                    tile_files.append(File(tile_lfn))

            split_job = (
                Job("image_split", _id=f"split_{basename}",
                    node_label=f"split_{basename}")
                .add_args(
                    "--input", input_file,
                    "--output-prefix", basename,
                    "--tile-size", str(tile_size),
                )
                .add_inputs(input_file)
                .add_pegasus_profiles(label=basename)
            )
            for tf_obj in tile_files:
                split_job.add_outputs(tf_obj, stage_out=False, register_replica=False)
            self.wf.add_jobs(split_job)

            # --- Jobs: color_segment (one per tile) ---
            seg_tile_files = []
            for tile_file in tile_files:
                seg_lfn = tile_file.lfn.replace(basename, f"{basename}_seg")
                seg_file = File(seg_lfn)
                seg_tile_files.append(seg_file)

                seg_job = (
                    Job("color_segment",
                        _id=f"seg_{os.path.splitext(tile_file.lfn)[0]}",
                        node_label=f"seg_{os.path.splitext(tile_file.lfn)[0]}")
                    .add_args("--input", tile_file, "--output", seg_file)
                    .add_inputs(tile_file)
                    .add_outputs(seg_file, stage_out=False, register_replica=False)
                    .add_pegasus_profiles(label=basename)
                )
                self.wf.add_jobs(seg_job)

            # --- Job: image_merge (fan-in) ---
            merged_file = File(f"{basename}_seg.png")
            input_args = []
            for sf in seg_tile_files:
                input_args.extend(["--input", sf])

            merge_job = (
                Job("image_merge", _id=f"merge_{basename}",
                    node_label=f"merge_{basename}")
                .add_args(
                    *input_args,
                    "--output", merged_file,
                    "--tile-size", str(tile_size),
                    "--original-size", str(original_size),
                )
                .add_inputs(*seg_tile_files)
                .add_outputs(merged_file, stage_out=True, register_replica=False)
                .add_pegasus_profiles(label=basename)
            )
            self.wf.add_jobs(merge_job)

            # --- Auto-label jobs: build image tiles (orig and/or filtered)
            #     plus the shared color-seg mask tiles, all 256x256 ---
            if args.auto_label:
                def _add_image_split(src_file, img_prefix, job_id):
                    """Add a split_images job; return its 256x256 tile Files."""
                    tiles = []
                    for r in range(0, original_size, mask_tile_size):
                        for c in range(0, original_size, mask_tile_size):
                            tile_lfn = (f"{img_prefix}_{str(r).zfill(4)}_"
                                        f"{str(c).zfill(4)}.png")
                            tiles.append(File(tile_lfn))
                    job = (
                        Job("image_split", _id=job_id, node_label=job_id)
                        .add_args(
                            "--input", src_file,
                            "--output-prefix", img_prefix,
                            "--tile-size", str(mask_tile_size),
                            "--grayscale",
                            "--pad",
                        )
                        .add_inputs(src_file)
                        .add_pegasus_profiles(label=basename)
                    )
                    for t in tiles:
                        job.add_outputs(t, stage_out=False, register_replica=False)
                    self.wf.add_jobs(job)
                    return tiles

                # ORIG path: tile the raw scene directly.
                if build_orig:
                    orig_tiles = _add_image_split(
                        input_file, f"train_img_{basename}",
                        f"split_images_{basename}")
                    auto_label_image_tiles_orig.extend(orig_tiles)

                # FILTERED path: clean the scene (thin-cloud/shadow removal),
                # then tile into U-Net inputs.
                if build_filt:
                    filtered_file = File(f"filtered_{basename}.png")
                    filter_job = (
                        Job("filter_image", _id=f"filter_{basename}",
                            node_label=f"filter_{basename}")
                        .add_args("--input", input_file, "--output", filtered_file)
                        .add_inputs(input_file)
                        .add_outputs(filtered_file, stage_out=True,
                                     register_replica=False)
                        .add_pegasus_profiles(label=basename)
                    )
                    self.wf.add_jobs(filter_job)
                    filt_tiles = _add_image_split(
                        filtered_file, f"train_imgf_{basename}",
                        f"split_imagesf_{basename}")
                    auto_label_image_tiles_filt.extend(filt_tiles)

                    # Option A (default): derive the filtered branch's LABELS by
                    # color-segmenting the filtered tiles, so input and target are
                    # self-consistent (reproduces the paper's filtered condition).
                    # With --filtered-labels raw the branch instead reuses the
                    # raw-scene labels (the honest cross-comparison).
                    if args.filtered_labels == "filtered":
                        for it in filt_tiles:
                            base = os.path.splitext(it.lfn)[0]
                            lbl = File(base.replace("train_imgf_", "train_maskf_")
                                       + "_seg.png")
                            segf_job = (
                                Job("color_segment",
                                    _id=f"segf_{base}", node_label=f"segf_{base}")
                                .add_args("--input", it, "--output", lbl)
                                .add_inputs(it)
                                .add_outputs(lbl, stage_out=False,
                                             register_replica=False)
                                .add_pegasus_profiles(label=basename)
                            )
                            self.wf.add_jobs(segf_job)
                            auto_label_mask_tiles_filt.append(lbl)

                # split_masks: split merged seg mask → 256x256 grayscale tiles
                mask_prefix = f"train_mask_{basename}_seg"
                mask_tiles = []
                for r in range(0, original_size, mask_tile_size):
                    for c in range(0, original_size, mask_tile_size):
                        tile_lfn = f"{mask_prefix}_{str(r).zfill(4)}_{str(c).zfill(4)}.png"
                        mask_tiles.append(File(tile_lfn))

                split_mask_job = (
                    Job("image_split", _id=f"split_masks_{basename}",
                        node_label=f"split_masks_{basename}")
                    .add_args(
                        "--input", merged_file,
                        "--output-prefix", mask_prefix,
                        "--tile-size", str(mask_tile_size),
                        "--grayscale",
                        "--pad",
                    )
                    .add_inputs(merged_file)
                    .add_pegasus_profiles(label=basename)
                )
                for mt in mask_tiles:
                    split_mask_job.add_outputs(mt, stage_out=False, register_replica=False)
                self.wf.add_jobs(split_mask_job)
                auto_label_mask_tiles_orig.extend(mask_tiles)

        # ============================================================
        # Stage 2: U-Net Training & Evaluation (optional)
        # ============================================================
        if not args.auto_label and (not args.train_images_dir or not args.train_masks_dir):
            logger.info("Stage 2 skipped (no --train-images-dir / --train-masks-dir)")
            return

        # Shared model.py support file (read-only input to train/plots)
        model_py_file = File("model.py")

        # Assemble the (label, image_tiles) branches to train. In both-paths
        # mode we train two models — one on raw images, one on filtered — that
        # share the same color-seg mask labels and the same train/test split
        # (identical --random-state and file ordering), giving an apples-to-
        # apples original-vs-filtered comparison from a single DAG.
        if args.auto_label:
            branches = []
            if build_orig:
                branches.append(("orig", auto_label_image_tiles_orig,
                                 auto_label_mask_tiles_orig))
            if build_filt:
                # Self-consistent filtered labels (Option A) unless overridden.
                filt_masks = (auto_label_mask_tiles_filt
                              if args.filtered_labels == "filtered"
                              else auto_label_mask_tiles_orig)
                branches.append(("filtered", auto_label_image_tiles_filt,
                                 filt_masks))
        else:
            train_img_files = []
            for img_path in sorted(glob.glob(os.path.join(args.train_images_dir, "*.png"))):
                lfn = f"train_img_{os.path.basename(img_path)}"
                train_img_files.append(File(lfn))

            train_mask_files = []
            for mask_path in sorted(glob.glob(os.path.join(args.train_masks_dir, "*.png"))):
                lfn = f"train_mask_{os.path.basename(mask_path)}"
                train_mask_files.append(File(lfn))
            branches = [("", train_img_files, train_mask_files)]

        if not any(imgs and masks for _, imgs, masks in branches):
            logger.warning("No training images or masks found, skipping Stage 2")
            return

        for label, train_img_files, train_mask_files in branches:
            if not train_img_files or not train_mask_files:
                continue
            logger.info(
                f"Stage 2 [{label or 'default'}]: {len(train_img_files)} images, "
                f"{len(train_mask_files)} masks")
            self._add_stage2_branch(
                args, train_img_files, train_mask_files, model_py_file, label)

    # ------------------------------------------------------------------
    # Stage 2 branch: preprocess → train → evaluate → plots
    # ------------------------------------------------------------------
    def _add_stage2_branch(self, args, train_img_files, train_mask_files,
                           model_py_file, label=""):
        """Add one U-Net training branch.

        label="" yields the canonical unprefixed filenames (single-path
        modes, backward compatible). A non-empty label (e.g. "orig" /
        "filtered") suffixes every intermediate and output file so multiple
        branches can coexist in one workflow without LFN collisions.
        """
        suffix = f"_{label}" if label else ""
        prefix = f"{label}_" if label else ""

        x_train_file = File(f"X_train{suffix}.npy")
        x_test_file = File(f"X_test{suffix}.npy")
        y_train_file = File(f"y_train_cat{suffix}.npy")
        y_test_file = File(f"y_test_cat{suffix}.npy")
        metadata_file = File(f"preprocess_metadata{suffix}.json")

        # --- Job: preprocess_data ---
        img_args = []
        for f in train_img_files:
            img_args.extend(["--image", f])
        mask_args = []
        for f in train_mask_files:
            mask_args.extend(["--mask", f])

        preprocess_job = (
            Job("preprocess_data", _id=f"preprocess{suffix}",
                node_label=f"preprocess{suffix}")
            .add_args(
                *img_args,
                *mask_args,
                "--x-train", x_train_file,
                "--x-test", x_test_file,
                "--y-train", y_train_file,
                "--y-test", y_test_file,
                "--metadata", metadata_file,
                "--test-size", str(args.test_size),
                "--n-classes", str(args.n_classes),
                "--random-state", str(args.random_state),
            )
            .add_inputs(*train_img_files, *train_mask_files)
            .add_outputs(x_train_file, stage_out=False, register_replica=False)
            .add_outputs(x_test_file, stage_out=False, register_replica=False)
            .add_outputs(y_train_file, stage_out=False, register_replica=False)
            .add_outputs(y_test_file, stage_out=False, register_replica=False)
            .add_outputs(metadata_file, stage_out=False, register_replica=False)
        )
        self.wf.add_jobs(preprocess_job)

        # --- Job: train_unet ---
        model_file = File(f"model{suffix}.hdf5")
        history_file = File(f"training_history{suffix}.json")

        train_job = (
            Job("train_unet", _id=f"train{suffix}", node_label=f"train{suffix}")
            .add_args(
                "--train-data", x_train_file,
                "--train-labels", y_train_file,
                "--output-model", model_file,
                "--output-history", history_file,
                "--metadata", metadata_file,
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--mode", args.training_mode,
            )
            .add_inputs(x_train_file, y_train_file, model_py_file, metadata_file)
            .add_outputs(model_file, stage_out=True, register_replica=False)
            .add_outputs(history_file, stage_out=True, register_replica=False)
        )
        self.wf.add_jobs(train_job)

        # --- Job: evaluate_model ---
        eval_file = File(f"evaluation_results{suffix}.json")

        eval_job = (
            Job("evaluate_model", _id=f"evaluate{suffix}",
                node_label=f"evaluate{suffix}")
            .add_args(
                "--model", model_file,
                "--test-data", x_test_file,
                "--test-labels", y_test_file,
                "--output", eval_file,
            )
            .add_inputs(model_file, x_test_file, y_test_file)
            .add_outputs(eval_file, stage_out=True, register_replica=False)
        )
        self.wf.add_jobs(eval_job)

        # --- Job: generate_plots ---
        training_curves = File(f"{prefix}training_curves.png")
        confusion_matrix = File(f"{prefix}confusion_matrix.png")
        prediction_samples = File(f"{prefix}prediction_samples.png")
        metrics_table = File(f"{prefix}metrics_table.png")
        per_class_json = File(f"{prefix}per_class_metrics.json")

        plots_args = [
            "--training-history", history_file,
            "--evaluation-results", eval_file,
            "--model", model_file,
            "--test-data", x_test_file,
            "--test-labels", y_test_file,
            "--metadata", metadata_file,
            "--output-dir", ".",
        ]
        if prefix:
            plots_args += ["--prefix", prefix]

        plots_job = (
            Job("generate_plots", _id=f"plots{suffix}",
                node_label=f"generate_plots{suffix}")
            .add_args(*plots_args)
            .add_inputs(history_file, eval_file, model_file,
                        x_test_file, y_test_file, model_py_file, metadata_file)
            .add_outputs(training_curves, stage_out=True, register_replica=False)
            .add_outputs(confusion_matrix, stage_out=True, register_replica=False)
            .add_outputs(prediction_samples, stage_out=True, register_replica=False)
            .add_outputs(metrics_table, stage_out=True, register_replica=False)
            .add_outputs(per_class_json, stage_out=True, register_replica=False)
        )
        self.wf.add_jobs(plots_job)

        # --- Job: infer_unet (paper Fig 9) ---
        # Apply the freshly trained model to whole scenes end-to-end.
        # In the "filtered" branch we re-run the same only_shadow_cloud_removal
        # filter on each inference scene so the model sees the same input
        # distribution it was trained on. The orig branch skips --filter.
        if args.infer:
            infer_filter = (label == "filtered")
            for img_path in args.infer_images:
                scene_basename = os.path.splitext(os.path.basename(img_path))[0]
                input_file = File(os.path.basename(img_path))
                out_file = File(f"{prefix}infer_{scene_basename}.png")

                infer_args = [
                    "--model", model_file,
                    "--input", input_file,
                    "--output", out_file,
                    "--tile-size", "256",  # must match training tile size
                    "--metadata", metadata_file,
                ]
                if infer_filter:
                    infer_args.append("--filter")

                infer_job = (
                    Job("infer_unet",
                        _id=f"infer_{scene_basename}{suffix}",
                        node_label=f"infer_{scene_basename}{suffix}")
                    .add_args(*infer_args)
                    .add_inputs(model_file, input_file, model_py_file,
                                metadata_file)
                    .add_outputs(out_file, stage_out=True,
                                 register_replica=False)
                )
                self.wf.add_jobs(infer_job)


# ======================================================================
# main()
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate Pegasus workflow for S2 sea ice segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage 1 only — produces 2000x2000 merged masks, NOT 256x256 tiles
  %(prog)s --images data/s2_scenes/s2_vis_*.png

  # Auto-label (recommended) — single DAG, no external data dirs needed.
  # By DEFAULT trains BOTH an unfiltered and a thin-cloud/shadow-filtered
  # U-Net (paper Table IV), sharing color-seg labels and the train/test
  # split. Outputs are prefixed orig_/filtered_.
  %(prog)s --images data/s2_scenes/s2_vis_*.png --auto-label

  # Restrict to a single path if desired
  %(prog)s --images data/s2_scenes/s2_vis_*.png --auto-label --paths filtered

  # Both stages with pre-existing 256x256 mask tiles
  %(prog)s --images data/s2_scenes/s2_vis_*.png \\
      --train-images-dir data/train_images/ \\
      --train-masks-dir data/train_masks/

  # Horovod training (with pre-existing masks)
  %(prog)s --images data/s2_scenes/s2_vis_*.png \\
      --train-images-dir data/train_images/ \\
      --train-masks-dir data/train_masks/ \\
      --training-mode horovod
""",
    )

    # Standard Pegasus arguments
    parser.add_argument("-s", "--skip-sites-catalog", action="store_true",
                        help="Skip site catalog creation")
    parser.add_argument("-e", "--execution-site-name", type=str, default="condorpool",
                        help="CPU execution site (default: condorpool)")
    parser.add_argument("--gpu-site-name", type=str, default="gpu-condorpool",
                        help="GPU execution site (default: gpu-condorpool)")
    parser.add_argument("-o", "--output", type=str, default="workflow.yml",
                        help="Output file (default: workflow.yml)")
    parser.add_argument("--container-image", type=str,
                        default="kthare10/s2-segmentation:latest",
                        help="Docker container image")

    # Stage 1: Color segmentation
    parser.add_argument("--images", type=str, nargs="+", required=True,
                        help="Input Sentinel-2 PNG images")
    parser.add_argument("--tile-size", type=int, default=250,
                        help="Tile size in pixels (default: 250)")
    parser.add_argument("--original-size", type=int, default=2000,
                        help="Original image dimension (default: 2000)")

    # Stage 2: U-Net training (optional)
    parser.add_argument("--auto-label", action="store_true",
                        help="Single-DAG auto-label mode: splits source scenes "
                             "into 256x256 grayscale training tiles and Stage 1 "
                             "masks into matching tiles. No external data dirs needed.")
    parser.add_argument("--paths", choices=["both", "orig", "filtered"],
                        default="both",
                        help="Which auto-label training path(s) to run "
                             "(default: both). 'both' trains an unfiltered AND "
                             "a thin-cloud/shadow-filtered U-Net in one DAG "
                             "(paper Table IV columns), sharing the train/test "
                             "split; outputs are prefixed orig_/filtered_. "
                             "'orig'/'filtered' restrict to a single path.")
    parser.add_argument("--filtered-labels", choices=["filtered", "raw"],
                        default="filtered",
                        help="How the filtered path's LABELS are produced "
                             "(default: filtered). 'filtered' color-segments the "
                             "filtered tiles so input and label are self-"
                             "consistent (reproduces the paper's ~99%). 'raw' "
                             "reuses the raw-scene labels (filtered input vs "
                             "raw-derived target — the honest cross-comparison).")
    parser.add_argument("--train-images-dir", type=str, default=None,
                        help="Directory of 256x256 grayscale training images")
    parser.add_argument("--train-masks-dir", type=str, default=None,
                        help="Directory of 256x256 segmentation masks")
    parser.add_argument("--training-mode", type=str, default="single-gpu",
                        choices=["single-gpu", "mirrored", "horovod"],
                        help="Training mode (default: single-gpu)")
    parser.add_argument("--infer", action="store_true",
                        help="After training, run infer_unet on whole scenes "
                             "and emit colour-coded prediction PNGs "
                             "({orig,filtered}_infer_<scene>.png). "
                             "The filtered branch applies the same "
                             "only_shadow_cloud_removal filter to each scene "
                             "before predicting.")
    parser.add_argument("--infer-images", type=str, nargs="+", default=None,
                        help="Scenes to run inference on (default: same as "
                             "--images). Use this to predict on scenes that "
                             "weren't part of the training corpus.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--n-classes", type=int, default=3, help="Segmentation classes")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test split ratio")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    # Default inference scenes to the training corpus when --infer is on
    # but no separate --infer-images list was supplied.
    if args.infer_images is None:
        args.infer_images = args.images

    # Validate inputs
    for img_path in args.images:
        if not os.path.exists(img_path):
            logger.error(f"Image not found: {img_path}")
            sys.exit(1)

    if args.infer:
        for img_path in args.infer_images:
            if not os.path.exists(img_path):
                logger.error(f"Inference image not found: {img_path}")
                sys.exit(1)

    if not args.auto_label and args.train_images_dir and not os.path.isdir(args.train_images_dir):
        logger.error(f"Training images directory not found: {args.train_images_dir}")
        sys.exit(1)

    if not args.auto_label and args.train_masks_dir and not os.path.isdir(args.train_masks_dir):
        logger.error(f"Training masks directory not found: {args.train_masks_dir}")
        sys.exit(1)

    n_images = len(args.images)
    n_tiles = n_images * (args.original_size // args.tile_size) ** 2

    logger.info("=" * 70)
    logger.info("S2 SEGMENTATION WORKFLOW GENERATOR")
    logger.info("=" * 70)
    logger.info(f"Source images: {n_images}")
    logger.info(f"Tiles per image: {(args.original_size // args.tile_size) ** 2}")
    logger.info(f"Total parallel segment jobs: {n_tiles}")
    stage2_enabled = args.auto_label or (args.train_images_dir and args.train_masks_dir)
    logger.info(f"Stage 2 (U-Net): {'enabled' if stage2_enabled else 'disabled'}")
    if args.auto_label:
        logger.info("Auto-label: ON (Stage 1 masks → split → Stage 2)")
    if stage2_enabled:
        logger.info(f"Training mode: {args.training_mode}")
    logger.info("=" * 70)

    try:
        workflow = S2SegmentationWorkflow(dagfile=args.output)

        workflow.create_pegasus_properties()

        if not args.skip_sites_catalog:
            workflow.create_sites_catalog(
                exec_site_name=args.execution_site_name,
                gpu_site_name=args.gpu_site_name,
            )

        workflow.create_transformation_catalog(
            exec_site_name=args.execution_site_name,
            gpu_site_name=args.gpu_site_name,
            container_image=args.container_image,
        )
        workflow.create_replica_catalog(args)
        workflow.create_workflow(args)
        workflow.write()

        logger.info(f"\nWorkflow written to {args.output}")
        logger.info(
            f"Submit: pegasus-plan --submit "
            f"-s {args.execution_site_name} -o local {args.output}"
        )

    except Exception as e:
        logger.error(f"Failed to generate workflow: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
