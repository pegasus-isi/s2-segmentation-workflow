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
    "resize_image":           {"memory": "1 GB",   "cores": 1},
    "image_split":            {"memory": "512 MB", "cores": 1},
    "color_segment":          {"memory": "256 MB", "cores": 1},
    "filter_image":           {"memory": "2 GB",   "cores": 1},
    "image_merge":            {"memory": "1 GB",   "cores": 1},
    "compute_cloud_fraction": {"memory": "2 GB",   "cores": 1},
    "preprocess_data":        {"memory": "14 GB",  "cores": 2},
    "train_unet":             {"memory": "14 GB",  "cores": 4, "gpus": 1},
    "evaluate_model":         {"memory": "8 GB",   "cores": 2, "gpus": 1},
    "evaluate_stratified":    {"memory": "8 GB",   "cores": 2, "gpus": 1},
    "generate_plots":         {"memory": "14 GB",  "cores": 2, "gpus": 1},
    "infer_unet":             {"memory": "8 GB",   "cores": 2, "gpus": 1},
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
        cpu_tools = ["resize_image", "image_split", "color_segment",
                     "filter_image", "image_merge", "compute_cloud_fraction",
                     "preprocess_data"]
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
        gpu_tools = ["train_unet", "evaluate_model", "evaluate_stratified",
                     "generate_plots", "infer_unet"]
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

        # Register filter_image.py as a support file — infer_unet imports
        # only_shadow_cloud_removal from it at module level, so every infer
        # job needs it staged alongside the executable.
        filter_image_py = os.path.join(self.wf_dir, "bin/filter_image.py")
        self.rc.add_replica(
            "local", "filter_image.py", "file://" + filter_image_py
        )

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

        # Paper geometry: 2048x2048 scenes tile evenly into 256x256 (66
        # scenes -> 4224 tiles, §IV-A). --scene-size resizes every scene
        # in-DAG before any tiling; 0 keeps the native size, in which
        # case edge tiles are padded.
        scene_size = args.scene_size if args.scene_size else args.original_size

        # --- Jobs: resize_image (one per unique scene, incl. infer-only) ---
        # scene_files maps basename -> the File every downstream job consumes
        # (the resized scene, or the raw scene when --scene-size 0).
        scene_files = self.scene_files = {}
        all_scene_paths = list(args.images)
        if args.infer and args.infer_images:
            all_scene_paths += list(args.infer_images)
        for img_path in dict.fromkeys(all_scene_paths):
            basename = os.path.splitext(os.path.basename(img_path))[0]
            if basename in scene_files:
                continue
            raw_file = File(os.path.basename(img_path))
            if args.scene_size:
                resized_file = File(f"resized_{basename}.png")
                resize_job = (
                    Job("resize_image", _id=f"resize_{basename}",
                        node_label=f"resize_{basename}")
                    .add_args("--input", raw_file,
                              "--output", resized_file,
                              "--size", str(args.scene_size))
                    .add_inputs(raw_file)
                    .add_outputs(resized_file, stage_out=False,
                                 register_replica=False)
                    .add_pegasus_profiles(label=basename)
                )
                self.wf.add_jobs(resize_job)
                scene_files[basename] = resized_file
            else:
                scene_files[basename] = raw_file

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
        cloud_fraction_files = []         # one --cloud-fraction JSON per scene
        mask_tile_size = 256  # U-Net training tile size

        # Both the unfiltered and filtered training-image paths are built by
        # default (the paper compares both in Table IV); --paths restricts.
        build_orig = args.paths in ("both", "orig")
        build_filt = args.paths in ("both", "filtered")

        for img_path in args.images:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            input_file = scene_files[basename]

            # --- Job: image_split ---
            tile_files = []
            for r in range(0, scene_size, tile_size):
                for c in range(0, scene_size, tile_size):
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

            # --- Job: compute_cloud_fraction (Table V / Fig 13 stratification)
            if args.stratified_eval:
                cf_file = File(f"cloud_fraction_{basename}.json")
                cf_job = (
                    Job("compute_cloud_fraction",
                        _id=f"cloudfrac_{basename}",
                        node_label=f"cloudfrac_{basename}")
                    .add_args(
                        "--input", input_file,
                        "--output", cf_file,
                        "--tile-size", str(mask_tile_size),
                        # input_file may be the resized scene; key by the
                        # original scene name so the JSON keys match the
                        # training tiles' <scene>_<row>_<col>.
                        "--key-prefix", basename,
                    )
                    .add_inputs(input_file)
                    .add_outputs(cf_file, stage_out=True,
                                 register_replica=False)
                    .add_pegasus_profiles(label=basename)
                )
                self.wf.add_jobs(cf_job)
                cloud_fraction_files.append(cf_file)

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
                    "--original-size", str(scene_size),
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
                    for r in range(0, scene_size, mask_tile_size):
                        for c in range(0, scene_size, mask_tile_size):
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

                # ORIG path: tile the raw scene directly. (Also produced as a
                # prerequisite to per-tile filtering when build_filt is True
                # and --filter-scale tile.)
                need_orig_tiles_for_filter = (
                    build_filt and args.filter_scale == "tile")
                orig_tiles = None
                if build_orig or need_orig_tiles_for_filter:
                    orig_tiles = _add_image_split(
                        input_file, f"train_img_{basename}",
                        f"split_images_{basename}")
                    if build_orig:
                        auto_label_image_tiles_orig.extend(orig_tiles)

                # FILTERED path: clean the scene (thin-cloud/shadow removal),
                # then tile into U-Net inputs.
                if build_filt:
                    if args.filter_scale == "scene":
                        # Apply only_shadow_cloud_removal once to the full
                        # scene, then tile the result.
                        filtered_file = File(f"filtered_{basename}.png")
                        filter_job = (
                            Job("filter_image", _id=f"filter_{basename}",
                                node_label=f"filter_{basename}")
                            .add_args("--input", input_file,
                                      "--output", filtered_file,
                                      "--kernel-size", str(args.filter_kernel_size))
                            .add_inputs(input_file)
                            .add_outputs(filtered_file, stage_out=True,
                                         register_replica=False)
                            .add_pegasus_profiles(label=basename)
                        )
                        self.wf.add_jobs(filter_job)
                        filt_tiles = _add_image_split(
                            filtered_file, f"train_imgf_{basename}",
                            f"split_imagesf_{basename}")
                    else:
                        # Apply only_shadow_cloud_removal per 256x256 tile so
                        # the medianBlur kernel sees only the tile's own
                        # local background (matches the Spark inference path).
                        filt_tiles = []
                        for ot in orig_tiles:
                            ft_lfn = ot.lfn.replace(
                                "train_img_", "train_imgf_", 1)
                            ft = File(ft_lfn)
                            base = os.path.splitext(ot.lfn)[0]
                            filter_tile_job = (
                                Job("filter_image",
                                    _id=f"filter_tile_{base}",
                                    node_label=f"filter_tile_{base}")
                                .add_args("--input", ot, "--output", ft,
                                          "--kernel-size",
                                          str(args.filter_kernel_size))
                                .add_inputs(ot)
                                .add_outputs(ft, stage_out=False,
                                             register_replica=False)
                                .add_pegasus_profiles(label=basename)
                            )
                            self.wf.add_jobs(filter_tile_job)
                            filt_tiles.append(ft)
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
                for r in range(0, scene_size, mask_tile_size):
                    for c in range(0, scene_size, mask_tile_size):
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
                        # Pad mask edges with the open-water gray value so
                        # padding cannot become a phantom 4th label class.
                        "--pad-value", "149",
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
                args, train_img_files, train_mask_files, model_py_file,
                label, cloud_fraction_files)

    # ------------------------------------------------------------------
    # Stage 2 branch: preprocess → train → evaluate → plots
    # ------------------------------------------------------------------
    def _add_stage2_branch(self, args, train_img_files, train_mask_files,
                           model_py_file, label="", cloud_fraction_files=None):
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
        do_stratified = bool(args.stratified_eval and cloud_fraction_files)
        test_cf_file = (File(f"test_cloud_fractions{suffix}.npy")
                        if do_stratified else None)

        # --- Job: preprocess_data ---
        img_args = []
        for f in train_img_files:
            img_args.extend(["--image", f])
        mask_args = []
        for f in train_mask_files:
            mask_args.extend(["--mask", f])

        extra_args = []
        if do_stratified:
            for cf in cloud_fraction_files:
                extra_args.extend(["--cloud-fraction", cf])
            extra_args.extend(["--test-cloud-fractions", test_cf_file])

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
                *extra_args,
            )
            .add_inputs(*train_img_files, *train_mask_files)
            .add_outputs(x_train_file, stage_out=False, register_replica=False)
            .add_outputs(x_test_file, stage_out=False, register_replica=False)
            .add_outputs(y_train_file, stage_out=False, register_replica=False)
            .add_outputs(y_test_file, stage_out=False, register_replica=False)
            .add_outputs(metadata_file, stage_out=False, register_replica=False)
        )
        if do_stratified:
            preprocess_job.add_inputs(*cloud_fraction_files)
            preprocess_job.add_outputs(test_cf_file, stage_out=True,
                                       register_replica=False)
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

        # --- Job: evaluate_stratified (paper Table V / Fig 13 panels) ---
        if do_stratified:
            strat_summary = File(f"{prefix}stratified_summary.json")
            strat_high_eval = File(f"{prefix}evaluation_results_high_cloud.json")
            strat_low_eval = File(f"{prefix}evaluation_results_low_cloud.json")
            strat_high_cm = File(f"{prefix}high_cloud_confusion_matrix.png")
            strat_low_cm = File(f"{prefix}low_cloud_confusion_matrix.png")
            strat_high_pc = File(f"{prefix}high_cloud_per_class_metrics.json")
            strat_low_pc = File(f"{prefix}low_cloud_per_class_metrics.json")
            strat_high_mt = File(f"{prefix}high_cloud_metrics_table.png")
            strat_low_mt = File(f"{prefix}low_cloud_metrics_table.png")

            strat_job = (
                Job("evaluate_stratified",
                    _id=f"strat_eval{suffix}",
                    node_label=f"strat_eval{suffix}")
                .add_args(
                    "--model", model_file,
                    "--test-data", x_test_file,
                    "--test-labels", y_test_file,
                    "--test-cloud-fractions", test_cf_file,
                    "--threshold", str(args.cloud_threshold),
                    "--output-dir", ".",
                    "--prefix", prefix,
                )
                .add_inputs(model_file, x_test_file, y_test_file, test_cf_file)
                .add_outputs(strat_summary, stage_out=True, register_replica=False)
                .add_outputs(strat_high_eval, stage_out=True, register_replica=False)
                .add_outputs(strat_low_eval, stage_out=True, register_replica=False)
                .add_outputs(strat_high_cm, stage_out=True, register_replica=False)
                .add_outputs(strat_low_cm, stage_out=True, register_replica=False)
                .add_outputs(strat_high_pc, stage_out=True, register_replica=False)
                .add_outputs(strat_low_pc, stage_out=True, register_replica=False)
                .add_outputs(strat_high_mt, stage_out=True, register_replica=False)
                .add_outputs(strat_low_mt, stage_out=True, register_replica=False)
            )
            self.wf.add_jobs(strat_job)

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
                input_file = self.scene_files[scene_basename]
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
                                metadata_file, File("filter_image.py"))
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
  # CANONICAL PAPER REPRODUCTION — the defaults do everything: resize
  # scenes to 2048x2048, auto-label (Fig 6), train BOTH the unfiltered
  # and thin-cloud/shadow-filtered U-Nets (Table IV), stratified
  # high/low-cloud evaluation (Table V, Fig 13), and whole-scene
  # inference (Fig 9/14). Outputs are prefixed orig_/filtered_.
  %(prog)s --images data/s2_scenes/s2_vis_*.png

  # Stage 1 color segmentation only (no training)
  %(prog)s --images data/s2_scenes/s2_vis_*.png --no-auto-label

  # Restrict to a single training path / skip optional stages
  %(prog)s --images data/s2_scenes/s2_vis_*.png --paths filtered
  %(prog)s --images data/s2_scenes/s2_vis_*.png --no-infer --no-stratified-eval

  # Variant scenarios (non-default, for comparison runs)
  %(prog)s --images data/s2_scenes/s2_vis_*.png --filter-scale tile
  %(prog)s --images data/s2_scenes/s2_vis_*.png --filtered-labels raw
  %(prog)s --images data/s2_scenes/s2_vis_*.png --scene-size 0   # native 2000², padded

  # Horovod distributed training (paper Fig 12)
  %(prog)s --images data/s2_scenes/s2_vis_*.png --training-mode horovod
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
    parser.add_argument("--tile-size", type=int, default=256,
                        help="Stage 1 color-segmentation tile size in pixels "
                             "(default: 256, matching the paper; the legacy "
                             "parallel demo used 250).")
    parser.add_argument("--original-size", type=int, default=2000,
                        help="Native input scene dimension (default: 2000, "
                             "the GEE export size). Only used when "
                             "--scene-size 0 disables in-DAG resizing.")
    parser.add_argument("--scene-size", type=int, default=2048,
                        help="Resize every scene to this square size before "
                             "any tiling (default: 2048, the paper's scene "
                             "geometry — 2048/256 tiles evenly, so no edge "
                             "padding ever enters the labels). Pass 0 to keep "
                             "the native size; edge tiles are then padded.")

    # Stage 2: U-Net training (optional)
    parser.add_argument("--auto-label", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Single-DAG auto-label mode: splits source scenes "
                             "into 256x256 grayscale training tiles and Stage 1 "
                             "masks into matching tiles. No external data dirs "
                             "needed. ON by default (the paper's pipeline); "
                             "--no-auto-label restricts the DAG to Stage 1 "
                             "color segmentation only (or external data via "
                             "--train-images-dir/--train-masks-dir).")
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
                             "consistent (reproduces the paper's ~99%%). 'raw' "
                             "reuses the raw-scene labels (filtered input vs "
                             "raw-derived target — the honest cross-comparison).")
    parser.add_argument("--train-images-dir", type=str, default=None,
                        help="Directory of 256x256 grayscale training images")
    parser.add_argument("--train-masks-dir", type=str, default=None,
                        help="Directory of 256x256 segmentation masks")
    parser.add_argument("--training-mode", type=str, default="single-gpu",
                        choices=["single-gpu", "mirrored", "horovod"],
                        help="Training mode (default: single-gpu)")
    parser.add_argument("--filter-scale", choices=["scene", "tile"],
                        default="scene",
                        help="Apply only_shadow_cloud_removal to the whole "
                             "scene (default: 'scene', paper's described "
                             "configuration) or per 256x256 training tile "
                             "('tile', matches the Spark map-reduce inference "
                             "path in the reference notebooks). At tile scale "
                             "the medianBlur kernel is auto-shrunk so it stays "
                             "the same fraction of the input dimension.")
    parser.add_argument("--filter-kernel-size", type=int, default=None,
                        help="medianBlur kernel for background estimation. "
                             "Defaults: 155 at --filter-scale scene, 19 at "
                             "--filter-scale tile. Must be odd and >= 3.")
    parser.add_argument("--stratified-eval",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Compute per-tile cloud/shadow fractions and "
                             "evaluate each trained branch separately on the "
                             "high-cloud (≥10%%) and low-cloud (<10%%) test "
                             "subsets, reproducing the paper's Table V split "
                             "and the per-stratum panels of Fig 13. ON by "
                             "default; --no-stratified-eval skips it.")
    parser.add_argument("--cloud-threshold", type=float, default=0.10,
                        help="Cloud-fraction cutoff between strata "
                             "(default: 0.10, matching the paper).")
    parser.add_argument("--infer", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="After training, run infer_unet on whole scenes "
                             "and emit colour-coded prediction PNGs "
                             "({orig,filtered}_infer_<scene>.png, paper Fig 9/14). "
                             "The filtered branch applies the same "
                             "only_shadow_cloud_removal filter to each scene "
                             "before predicting. ON by default; --no-infer "
                             "skips it.")
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

    # Resolve filter kernel default based on scale.
    if args.filter_kernel_size is None:
        args.filter_kernel_size = 155 if args.filter_scale == "scene" else 19
    if args.filter_kernel_size < 3 or args.filter_kernel_size % 2 == 0:
        logger.error(
            f"--filter-kernel-size must be odd and >= 3 "
            f"(got {args.filter_kernel_size}).")
        sys.exit(1)

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
