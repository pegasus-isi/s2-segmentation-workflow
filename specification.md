# S2 Parallel Workflow — Pegasus Workflow Specification

## 1. Overview

> See also: [`README.md`](README.md) for quick-start usage, project structure, and CLI reference.

This document specifies a Pegasus WMS workflow for **Sentinel-2 satellite sea ice segmentation**, executed on an **HTCondor** pool. The workflow combines two existing pipelines into an end-to-end DAG:

- **Stage 1 — Color Segmentation**: Split large Sentinel-2 images into tiles, apply HSV-based color segmentation in parallel to generate labeled training masks, then merge tiles back into full images.
- **Stage 2 — U-Net Training & Evaluation**: Preprocess the labeled data, train a U-Net semantic segmentation model (3-class: ice, thin-ice, water), and evaluate it.

### Design Goals

1. **Maximize parallelism** — Pegasus + HTCondor replace Python `multiprocessing`. Every independent unit of work becomes its own HTCondor job, enabling distribution across a cluster rather than a single machine.
2. **HTCondor as the execution backend** — Pegasus plans the DAG; HTCondor schedules and executes jobs across the pool. GPU jobs use HTCondor's `request_gpus` mechanism for placement on GPU-equipped nodes.

## 2. Project Structure

```
s2-segmentation-workflow/
├── workflow_generator.py       # Pegasus DAG generator
├── bin/                        # Wrapper scripts (one per pipeline step)
│   ├── model.py                # Shared U-Net model definition
│   ├── image_split.py          # Stage 1: tile splitting
│   ├── color_segment.py        # Stage 1: HSV segmentation
│   ├── image_merge.py          # Stage 1: tile reassembly (fan-in)
│   ├── preprocess_data.py      # Stage 2: data loading & encoding
│   ├── train_unet.py           # Stage 2: U-Net training (3 modes)
│   ├── evaluate_model.py       # Stage 2: model evaluation
│   └── generate_plots.py       # Stage 2: publication figures & tables
├── Docker/
│   └── S2_Dockerfile           # Container image definition
├── tests/                      # pytest test suite (37 tests)
├── download_data.py            # Sentinel-2 data download script (GEE)
├── run_manual.sh               # Bash-based local integration test
├── requirements.txt            # Python runtime dependencies
├── README.md                   # Quick-start guide and CLI reference
└── specification.md            # This file
```

## 3. Source Code Inventory

| File | Role |
|---|---|
| `parallel_segmentation.py` | Multiprocessing split → segment → merge pipeline |
| `s2_u_net_tf.py` | U-Net training with TensorFlow MirroredStrategy |
| `s2_u_net_horovod.py` | U-Net training with Horovod distributed optimizer |

## 4. Pipeline Stages and Jobs

### Stage 1: Color Segmentation (Label Generation)

#### Job 1 — `image_split`

- **Source function**: `image_split()` in `parallel_segmentation.py`
- **Input**: One 2000×2000 PNG image from `s2_par/s2_vis/`
- **Output**: 64 tiles of 250×250 PNG (8×8 grid), named `{basename}_{row}_{col}.png`
- **Parallelism**: One `image_split` job **per source image**. All N split jobs run **concurrently** on HTCondor — they have no inter-dependencies.
- **HTCondor profile**: CPU-only, ~512 MB RAM, short runtime.
- **Parameters**:
  - `--input_image <path>` — path to the source image
  - `--output_dir <path>` — directory for output tiles
  - `--tile_size 250` — tile dimension (default 250)
- **Dependencies**: None (entry point)

#### Job 2 — `color_segment`

- **Source function**: `color_segmentation()` + `col_seg()` in `parallel_segmentation.py`
- **Input**: One 250×250 tile PNG
- **Output**: One 250×250 segmented tile PNG (ice=red, thin-ice=blue, water=green)
- **Parallelism**: One job **per tile** — this is the embarrassingly parallel stage. For N source images, there are **N×64 independent segment jobs** all eligible to run concurrently on HTCondor. This is the primary parallelism exploit — the original code used `multiprocessing.Pool` on a single machine; HTCondor distributes these across the entire cluster.
- **HTCondor profile**: CPU-only, ~256 MB RAM, very short runtime (~seconds per tile).
- **Parameters**:
  - `--input_tile <path>` — input tile path
  - `--output_tile <path>` — output segmented tile path
- **Logic**: Convert RGB→HSV, apply three threshold ranges (ice/thin-ice/water), color-mask the output.
- **Dependencies**: Depends on the corresponding `image_split` job that produced its input tile.

#### Job 3 — `image_merge`

- **Source function**: `image_merge()` in `parallel_segmentation.py`
- **Input**: All 64 segmented tiles for one source image
- **Output**: One 2000×2000 reassembled segmentation mask PNG in `s2_par/s2_seg/`
- **Parallelism**: One merge job **per source image** (fan-in). All N merge jobs are independent and run **concurrently** once their respective tile dependencies are met.
- **HTCondor profile**: CPU-only, ~1 GB RAM (loads 64 tiles into memory).
- **Parameters**:
  - `--input_dir <path>` — directory containing the 64 segmented tiles for this image
  - `--output_image <path>` — path for the merged output
  - `--tile_size 250`
  - `--original_size 2000`
- **Dependencies**: All 64 `color_segment` jobs for this source image must complete.

### Auto-label Bridge (with `--auto-label`)

#### Job 3b — `split_images` (auto-label mode only)

- **Source**: `image_split.py` (same script as Job 1, invoked with `--grayscale --pad --tile-size 256`)
- **When**: Only present when `--auto-label` is passed to the workflow generator.
- **Input**: One 2000×2000 source scene PNG (same input as Job 1)
- **Output**: 256×256 grayscale tile PNGs, named `train_img_{basename}_{row}_{col}.png`
- **Parallelism**: One job per source image, all run concurrently. Runs in parallel with Stage 1 (no dependency on segmentation).
- **HTCondor profile**: CPU-only, ~512 MB RAM.
- **Parameters**:
  - `--input <source_scene.png>`
  - `--output-prefix train_img_{basename}`
  - `--tile-size 256`
  - `--grayscale`
  - `--pad`
- **Dependencies**: None (reads original source scene directly).

#### Job 3c — `split_masks` (auto-label mode only)

- **Source**: `image_split.py` (same script as Job 1, invoked with `--grayscale --pad --tile-size 256`)
- **When**: Only present when `--auto-label` is passed to the workflow generator.
- **Input**: One 2000×2000 merged segmentation mask PNG from `image_merge`
- **Output**: 256×256 grayscale tile PNGs, named `train_mask_{basename}_seg_{row}_{col}.png`
- **Parallelism**: One job per source image, all run concurrently.
- **HTCondor profile**: CPU-only, ~512 MB RAM.
- **Parameters**:
  - `--input <merged_mask.png>`
  - `--output-prefix train_mask_{basename}_seg`
  - `--tile-size 256`
  - `--grayscale`
  - `--pad`
- **Dependencies**: `image_merge` for the same source image.

> **Tile count matching**: Both `split_images` and `split_masks` use `--pad` to zero-pad edge tiles to full 256×256 when the image dimension (2000) is not evenly divisible by 256. This guarantees both produce exactly `ceil(2000/256)^2 = 64` tiles per scene, so image and mask counts always match for `preprocess_data`.

#### Job 3d — `filter_image` (auto-label mode, filtered path)

- **Source**: `bin/filter_image.py` — direct port of `only_shadow_cloud_removal()` from the
  paper's reference notebooks (dilate → medianBlur(155) → absdiff → Otsu → bitwise → min-max
  normalize → truncated threshold).
- **When**: Present when `--paths` is `both` or `filtered`.
- **Input**: One source scene PNG (same input as `split_images`).
- **Output**: One thin-cloud/shadow-filtered grayscale PNG, named `filtered_{basename}.png`.
- **Parallelism**: One job per source image, all run concurrently. Runs in parallel with
  Stage 1.
- **HTCondor profile**: CPU-only, ~1 GB RAM (`medianBlur(155)` on a 2048² scene).
- **Parameters**: `--input <source_scene.png>`, `--output <filtered.png>`.
- **Dependencies**: None.

The filtered scene then feeds a *second* `split_images` job (producing
`train_imgf_{basename}_*.png`) and, in the default Option A configuration
(`--filtered-labels filtered`), a *second* `color_segment` → `image_merge` → `split_masks`
chain that re-derives labels from the filtered tiles. With `--filtered-labels raw` the
filtered branch instead reuses the orig-branch mask tiles, giving filtered input vs raw
labels (the honest cross-comparison that exposes ~90% accuracy and motivates the paper's
~99% headline).

### Stage 2: U-Net Model Training

When `--paths both` is in effect (default), Stage 2 is **instantiated twice** — once on the
raw image/raw label branch (suffix `_orig`) and once on the filtered image / filtered-or-raw
label branch (suffix `_filtered`). Both branches share the same train/test split
(`random_state=0` and identical file ordering), so the comparison is apples-to-apples.
Job IDs and output filenames are suffixed accordingly (e.g. `train_orig`, `train_filtered`,
`model_orig.hdf5`, `model_filtered.hdf5`, `evaluation_results_{orig,filtered}.json`, and
`{orig prefix=none, filtered}_{training_curves,confusion_matrix,...}.png`).

#### Job 4 — `preprocess_data`

- **Source**: Data loading and encoding logic from `s2_u_net_tf.py` (lines 141–213)
- **Input**:
  - 256×256 grayscale training image tiles — from `split_images` jobs (auto-label) or `--train-images-dir`
  - 256×256 grayscale mask tiles — from `split_masks` jobs (auto-label) or `--train-masks-dir`
- **Output**:
  - `X_train.npy`, `X_test.npy` — normalized image arrays (80/20 split)
  - `y_train_cat.npy`, `y_test_cat.npy` — one-hot encoded 3-class label arrays
- **Parameters**:
  - `--images_dir <path>`
  - `--masks_dir <path>`
  - `--output_dir <path>`
  - `--test_size 0.20`
  - `--n_classes 3`
  - `--random_state 0`
- **Logic**: Split file indices into train/test first (no data loaded), then load and process each split separately to limit peak memory. Normalize images in float32 (manual L2 norm, avoids float64 intermediate from keras). LabelEncoder on masks, one-hot encode with `to_categorical`.
- **Memory optimization**: Processes train and test splits independently — never holds the full dataset in memory at once. Uses float32 throughout (~half the memory of default float64).
- **Dependencies**: Depends on `split_images` + `split_masks` jobs when `--auto-label` is used; otherwise reads from Replica Catalog entries.

#### Job 5 — `train_unet`

- **Source**: Model definition and training from `s2_u_net_tf.py` (lines 32–298)
- **Input**: `X_train.npy`, `y_train_cat.npy` from `preprocess_data`
- **Output**:
  - `model.hdf5` — saved trained model weights
  - `training_history.json` — loss/accuracy/F1 per epoch
  - `logs/` — TensorBoard log directory
- **Parameters**:
  - `--train_data <path>` — path to X_train.npy
  - `--train_labels <path>` — path to y_train_cat.npy
  - `--output_model <path>`
  - `--epochs 50`
  - `--batch_size 32`
  - `--n_classes 3`
- **Model architecture**: 6-level U-Net (16→32→64→128→256→512 filters), Conv2DTranspose upsampling, dropout (0.1–0.3), softmax output.
- **Compilation**: Adam optimizer, categorical crossentropy, custom F1/precision/recall metrics.
- **Resource requirements**: GPU node (requires `tensorflow-gpu`).
- **Training modes** (selectable via parameter):
  - **Single-GPU**: Default. One HTCondor job with `request_gpus = 1`.
  - **Multi-GPU (MirroredStrategy)**: Single HTCondor job with `request_gpus = N` on one node. Uses `tf.distribute.MirroredStrategy` from `s2_u_net_tf.py`.
  - **Multi-node (Horovod)**: Submitted as an HTCondor **parallel universe** or MPI job. Uses `horovodrun -np <N>` with `s2_u_net_horovod.py`. Requires HTCondor MPI support and Horovod in the container.
- **HTCondor profile**: `request_gpus >= 1`, `request_memory = 8 GB`, `request_cpus = 4`, long runtime.
- **Dependencies**: `preprocess_data`

#### Job 6 — `evaluate_model`

- **Source**: Evaluation logic from `s2_u_net_tf.py` (lines 303–313)
- **Input**: `model.hdf5`, `X_test.npy`, `y_test_cat.npy`
- **Output**:
  - `evaluation_results.json` — test loss, accuracy, F1, precision, recall
- **Parameters**:
  - `--model <path>`
  - `--test_data <path>`
  - `--test_labels <path>`
  - `--output <path>`
- **Dependencies**: `train_unet`, `preprocess_data`

#### Job 7 — `generate_plots`

- **Source**: New standalone script `bin/generate_plots.py`
- **Input**: `training_history.json`, `evaluation_results.json`, `model.hdf5`, `X_test.npy`, `y_test_cat.npy`, `preprocess_metadata.json`, `model.py`
- **Output**:
  - `training_curves.png` — 2×2 subplot grid: loss, accuracy, F1, precision/recall vs epoch
  - `confusion_matrix.png` — Normalized confusion matrix with counts (paper Fig 13)
  - `prediction_samples.png` — Grid of N samples: input image | ground truth | prediction (paper Fig 14)
  - `metrics_table.png` — Per-class classification metrics rendered as a table image (paper Table IV)
  - `per_class_metrics.json` — Per-class precision, recall, F1-score, and support
- **Parameters**:
  - `--training-history <path>` — path to training_history.json
  - `--evaluation-results <path>` — path to evaluation_results.json
  - `--model <path>` — path to model.hdf5
  - `--test-data <path>` — path to X_test.npy
  - `--test-labels <path>` — path to y_test_cat.npy
  - `--metadata <path>` — path to preprocess_metadata.json (optional, for n_classes)
  - `--output-dir <path>` — output directory for all plot files
  - `--num-samples 5` — number of prediction samples to visualize
  - `--dpi 150` — plot resolution
  - `--class-names "Ice,Thin Ice,Water"` — comma-separated class names
- **Logic**: Loads the trained model (with custom metric functions f1_m, precision_m, recall_m), runs predictions on the test set, computes per-class metrics via sklearn's `classification_report`, and generates four matplotlib figures plus a JSON summary. Uses `matplotlib.use("Agg")` for headless rendering.
- **HTCondor profile**: GPU node (requires TF/Keras to load model and run predictions), 4 GB RAM, 2 cores.
- **Dependencies**: `evaluate_model` (consumes eval_file), `train_unet` (consumes model_file, history_file), `preprocess_data` (consumes test data arrays, metadata)

#### Job 8 — `compute_cloud_fraction` (optional, paper Table V)

- **Source**: `bin/compute_cloud_fraction.py`.
- **When**: Emitted only when `--stratified-eval` is passed to `workflow_generator.py`. One job per source scene; runs in parallel with the Stage 1 colour-segmentation chain.
- **Input**: One source scene PNG.
- **Output**: `cloud_fraction_{basename}.json` — `{tile_basename: fraction, ...}`. Reuses the same Otsu intermediate mask produced inside `only_shadow_cloud_removal` (so the cloud/shadow definition matches the filter exactly), averaged over each 256×256 tile.
- **Parameters**: `--input`, `--output`, `--tile-size 256`.
- **HTCondor profile**: CPU-only, ~2 GB RAM.
- **Dependencies**: None.

#### Job 9 — `evaluate_stratified` (optional, paper Table V + Fig 13 panels)

- **Source**: `bin/evaluate_stratified.py`.
- **When**: Emitted per branch (`orig` / `filtered`) when `--stratified-eval` is passed. Consumes the branch's trained model + test split + the per-test-tile cloud fractions emitted by `preprocess_data` (which receives them via repeated `--cloud-fraction` args).
- **Inputs**: `model{_branch}.hdf5`, `X_test{_branch}.npy`, `y_test_cat{_branch}.npy`, `test_cloud_fractions{_branch}.npy`.
- **Outputs** (all prefixed `{branch_prefix}`):
  - `evaluation_results_high_cloud.json`, `evaluation_results_low_cloud.json` (loss / accuracy / F1 / precision / recall per stratum)
  - `high_cloud_confusion_matrix.png`, `low_cloud_confusion_matrix.png` (Fig 13-style)
  - `high_cloud_per_class_metrics.json`, `low_cloud_per_class_metrics.json`
  - `high_cloud_metrics_table.png`, `low_cloud_metrics_table.png`
  - `stratified_summary.json` (Table V row for this branch — accuracy / F1 / n_tiles for each stratum and the threshold used)
- **Parameters**: `--model`, `--test-data`, `--test-labels`, `--test-cloud-fractions`, `--threshold 0.10`, `--output-dir`, `--prefix`.
- **HTCondor profile**: GPU node, 8 GB RAM, 2 cores.
- **Dependencies**: `train_unet` (model), `preprocess_data` (test arrays + test_cloud_fractions).

#### Job 10 — `infer_unet` (optional, paper Fig 9)

- **Source**: `bin/infer_unet.py`.
- **When**: Emitted only when `--infer` is passed to `workflow_generator.py`. One inference job per `(branch, scene)` pair — i.e. when `--paths both --infer` is in effect, the DAG contains `2 × N` inference jobs (N = number of `--infer-images`, defaulting to `--images`).
- **Input**: Trained `model{_branch}.hdf5`, a full Sentinel-2 scene PNG, the branch's `preprocess_metadata{_branch}.json`, and `model.py`.
- **Output**: Colour-coded prediction PNG `{branch_prefix}infer_{scene_basename}.png` at the original scene resolution (red = thick ice, blue = thin ice, green = open water).
- **Parameters**:
  - `--model <path>`, `--input <scene.png>`, `--output <prediction.png>`
  - `--tile-size 256` (must match training tile size)
  - `--metadata <preprocess_metadata.json>` (recovers the exact class→colour mapping from the encoder's grayscale-class table)
  - `--filter` (added by the filtered branch only — re-runs `only_shadow_cloud_removal` on each inference scene so the model sees the same input distribution it was trained on)
  - `--batch-size 32`, `--class-colors "R,G,B"` (optional overrides)
- **Logic**: tile → optional filter → L2-normalize (same `normalize(axis=1)` used in training) → `model.predict` → argmax → recolour via metadata → reassemble → write RGB PNG. Edge tiles are zero-padded to the full `tile_size` and the predicted canvas is cropped back to the original scene dimensions before saving.
- **HTCondor profile**: GPU node, 8 GB RAM, 2 cores.
- **Dependencies**: `train_unet` (model), `preprocess_data` (metadata), and the source-scene replica in the Replica Catalog.

## 5. DAG Structure

```
  Image 0                    Image 1                    Image N-1
  ────────                   ────────                   ─────────
  image_split_0              image_split_1      ...     image_split_N-1
  ┌──┬──┬─...─┐              ┌──┬──┬─...─┐              ┌──┬──┬─...─┐
  │  │  │     │  ◄── all     │  │  │     │              │  │  │     │
  seg seg seg seg     run    seg seg seg seg            seg seg seg seg
  (0) (1)(2) (63)    on     (0) (1)(2) (63)            (0) (1)(2) (63)
  └──┴──┴─...─┘    HTCondor  └──┴──┴─...─┘              └──┴──┴─...─┘
       │           in             │                          │
  image_merge_0   parallel   image_merge_1             image_merge_N-1
       │                          │                          │
  [split_masks_0]           [split_masks_1]           [split_masks_N-1]
  (256x256 mask tiles)      (256x256 mask tiles)      (256x256 mask tiles)
       │                          │                          │
  [split_images_0]          [split_images_1]          [split_images_N-1]
  (256x256 img tiles)       (256x256 img tiles)       (256x256 img tiles)
       │                          │                          │
       └──────────────────────────┴──────────────────────────┘
                              │
                    (--auto-label: matched image + mask tiles)
                              │
                              ▼
                       preprocess_data
                              │
                              ▼
                    train_unet (GPU, Horovod/MirroredStrategy)
                              │
                              ▼
                       evaluate_model
                              │
                              ▼
                       generate_plots
```

> **Note**: The `split_images_*` and `split_masks_*` jobs (in brackets) only appear in `--auto-label` mode. They produce matched 256×256 training tile pairs from the source scenes. Without `--auto-label`, Stage 2 reads pre-existing data from `--train-images-dir` and `--train-masks-dir`.

**Parallelism in the DAG:**
- **Stage 1**: All N `image_split` jobs launch concurrently (no dependencies between images). As each completes, its 64 `color_segment` children launch immediately — yielding up to **N×64 concurrent HTCondor jobs**. Each image's `image_merge` waits only for its own 64 segments (not other images), so merges also overlap.
- **Auto-label bridge**: When `--auto-label` is set, two additional job types run per source image: `split_images` tiles the original scene into 256×256 grayscale training images (runs immediately, no dependency on segmentation), and `split_masks` tiles the merged mask into matching 256×256 grayscale labels (depends on `image_merge`). Both use `--pad` to zero-pad edge tiles, guaranteeing equal tile counts. All split jobs run concurrently. The resulting tiles are wired as `--image` and `--mask` inputs to `preprocess_data`.
- **Stage 2**: Sequential (preprocess → train → evaluate → generate_plots), but `train_unet` can use intra-job parallelism via multi-GPU MirroredStrategy or multi-node Horovod.
- The connection between stages is optional: if pre-labeled training data already exists, Stage 2 can run independently without `--auto-label` by providing `--train-images-dir` and `--train-masks-dir`.

## 6. Data Acquisition

The input data is **Sentinel-2 optical imagery** from ESA's Copernicus program, collected via [Google Earth Engine](https://earthengine.google.com/) (GEE).

### Reference Dataset (from the paper)

| Parameter | Value |
|---|---|
| Region | Ross Sea, Antarctica |
| Latitude | -70.00 to -78.00 (south) |
| Longitude | -140.00 to -180.00 (west) |
| Time period | November 2019 (Antarctic summer) |
| Bands | B4 (red), B3 (green), B2 (blue) |
| Resolution | 10m per pixel |
| Scenes | 66 large scenes |
| Training tiles | 4,224 images of 256×256 pixels |

> Iqrah et al., *"A Parallel Workflow for Polar Sea-Ice Classification using Auto-Labeling of Sentinel-2 Imagery,"* IEEE IPDPSW 2024. DOI: [10.1109/IPDPSW63119.2024.00172](https://doi.org/10.1109/IPDPSW63119.2024.00172)

### Download Script

The `download_data.py` script automates data acquisition from GEE:

```bash
pip install earthengine-api
earthengine authenticate --auth_mode=notebook

# Download scenes as PNGs
python download_data.py --method local --output-dir data/s2_scenes

# Download and split into 256x256 training tiles
python download_data.py --method local --output-dir data/s2_scenes --split-tiles

# Export to Google Drive (for large downloads)
python download_data.py --method drive --drive-folder s2_ross_sea
```

Training masks are **not downloaded** — they are produced by Stage 1 (color segmentation) of the workflow itself. This is the auto-labeling approach described in the paper.

## 7. Data Catalog

| Logical Name | Type | Format | Typical Size |
|---|---|---|---|
| `s2_vis_{id}.png` | Input | RGB PNG | 2000×2000, ~150 KB |
| `s2_vis_{id}_{row}_{col}.png` | Intermediate | RGB PNG tile | 250×250, ~3 KB |
| `s2_seg_{id}_{row}_{col}.png` | Intermediate | RGB PNG tile | 250×250, ~3 KB |
| `s2_seg_{id}.png` | Output | RGB PNG | 2000×2000, ~200 KB |
| `train_img_{id}_{row}_{col}.png` | Intermediate | Grayscale PNG (auto-label) | 256×256, zero-padded |
| `train_mask_{id}_seg_{row}_{col}.png` | Intermediate | Grayscale PNG (auto-label) | 256×256, zero-padded |
| `train_images_dir/*.png` | Input | Grayscale PNG (non-auto-label) | 256×256 |
| `train_masks_dir/*.png` | Input | Grayscale PNG (non-auto-label) | 256×256 |
| `X_train.npy` / `X_test.npy` | Intermediate | NumPy array | Varies with dataset |
| `y_train_cat.npy` / `y_test_cat.npy` | Intermediate | NumPy array | Varies with dataset |
| `model.hdf5` | Output | Keras model | ~25 MB |
| `evaluation_results.json` | Output | JSON | <1 KB |
| `training_curves.png` | Output | PNG plot | ~200 KB |
| `confusion_matrix.png` | Output | PNG plot | ~150 KB |
| `prediction_samples.png` | Output | PNG plot | ~500 KB |
| `metrics_table.png` | Output | PNG plot | ~50 KB |
| `per_class_metrics.json` | Output | JSON | <1 KB |

## 8. Dependencies and Container

### Python Packages

All runtime dependencies are listed in `requirements.txt`:

```
tensorflow>=2.10
opencv-python-headless
scikit-learn
Pillow
numpy
matplotlib
```

Install with:

```bash
pip install -r requirements.txt
```

### Optional (for Horovod distributed training)

```
horovod[tensorflow]
```

### Container Image

A single Docker image should include all dependencies. The `train_unet` job requires GPU support (NVIDIA runtime).

```
Base: tensorflow/tensorflow:2.13.0-gpu
+ opencv-python-headless, scikit-learn, Pillow
```

## 9. HTCondor Execution Configuration

Pegasus translates the DAG into an HTCondor DAGMan workflow. Each job type maps to an HTCondor submit description with appropriate resource requests.

### Site Catalog

| Site | Type | Purpose |
|---|---|---|
| `local` | Local | Pegasus planning, data staging |
| `condorpool` | HTCondor pool | CPU jobs (split, segment, merge, preprocess) |
| `gpu-condorpool` | HTCondor pool | GPU jobs (train, evaluate) |

### Job-to-HTCondor Mapping

| Job | HTCondor Universe | `request_cpus` | `request_memory` | `request_gpus` | `request_disk` | Concurrency |
|---|---|---|---|---|---|---|
| `image_split` | vanilla | 1 | 512 MB | 0 | 500 MB | N (one per image) |
| `color_segment` | vanilla | 1 | 256 MB | 0 | 10 MB | N×64 (all tiles) |
| `image_merge` | vanilla | 1 | 1 GB | 0 | 500 MB | N (one per image) |
| `preprocess_data` | vanilla | 2 | 8 GB | 0 | 2 GB | 1 |
| `train_unet` | vanilla (or parallel for Horovod) | 4 | 8 GB | 1+ | 5 GB | 1 |
| `evaluate_model` | vanilla | 2 | 4 GB | 1 | 2 GB | 1 |
| `generate_plots` | vanilla | 2 | 4 GB | 1 | 2 GB | 1 |

### Pegasus Profiles for HTCondor

```
# Example: color_segment jobs
condor.request_cpus = 1
condor.request_memory = 256
condor.request_disk = 10240
condor.+WantGPU = false

# Example: train_unet job
condor.request_cpus = 4
condor.request_memory = 8192
condor.request_gpus = 1
condor.+WantGPU = true
```

## 10. Parallelism Summary

| Level | What runs in parallel | Managed by | Max concurrent jobs |
|---|---|---|---|
| **Image-level** | All N `image_split` jobs | HTCondor DAGMan | N |
| **Tile-level** | All N×64 `color_segment` jobs | HTCondor DAGMan | N×64 (limited by pool slots) |
| **Merge-level** | All N `image_merge` jobs (independent per image) | HTCondor DAGMan | N |
| **Auto-label split** | All N `split_images` + N `split_masks` jobs | HTCondor DAGMan | 2N |
| **Intra-training** | Multi-GPU via MirroredStrategy or multi-node via Horovod | TensorFlow / Horovod | 1 job, multiple GPUs |

**Key insight**: The original `parallel_segmentation.py` uses `multiprocessing.Pool` to parallelize on a single machine. In the Pegasus+HTCondor version, each tile becomes an independent HTCondor job. For 10 source images, this yields **640 concurrent segment jobs** distributed across the cluster — far exceeding what a single-node multiprocessing pool can achieve.

## 11. Configurable Parameters

| Parameter | Default | Description |
|---|---|---|
| `tile_size` | 250 | Tile dimension for splitting |
| `original_size` | 2000 | Original image dimension |
| `n_classes` | 3 | Segmentation classes (ice, thin-ice, water) |
| `test_size` | 0.20 | Train/test split ratio |
| `epochs` | 50 | Training epochs |
| `batch_size` | 32 | Training batch size |
| `random_state` | 0 | Random seed for reproducibility |
| `--paths` | both | Auto-label training paths: `both` (paper Table IV), `orig`, or `filtered` |
| `--filtered-labels` | filtered | Filtered branch's label source: `filtered` (self-consistent, paper ~99%) or `raw` (filtered input vs raw labels, ~90%) |

## 12. Refactoring Notes

The existing code needs to be decomposed into standalone CLI scripts for Pegasus to invoke as individual jobs. Each script listed in Section 3 should:

1. Accept all inputs/outputs as command-line arguments (no hardcoded paths).
2. Read from stdin or files, write to stdout or files — no shared global state.
3. Exit with code 0 on success, non-zero on failure.
4. Log timing and metrics to stderr.

The `multi_unet_model()` function (currently duplicated in both `s2_u_net_tf.py` and `s2_u_net_horovod.py`) should be extracted into a shared module (`model.py`).

## 13. Testing

### Prerequisites

```bash
pip install pytest tensorflow opencv-python-headless scikit-learn Pillow numpy
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_image_split.py -v        # Stage 1: split
pytest tests/test_color_segment.py -v      # Stage 1: segment
pytest tests/test_image_merge.py -v        # Stage 1: merge
pytest tests/test_preprocess_data.py -v    # Stage 2: preprocess
pytest tests/test_model.py -v              # U-Net model definition
pytest tests/test_train_unet.py -v         # Stage 2: training (slower)
pytest tests/test_evaluate_model.py -v     # Stage 2: evaluation (slower)
pytest tests/test_workflow_generator.py -v # Pegasus DAG generation
pytest tests/test_integration.py -v        # End-to-end Stage 1 pipeline

# Run only fast tests (skip training/evaluation)
pytest tests/ -v -k "not train and not evaluate"

# Run the bash-based manual integration test
bash run_manual.sh
```

### Test Structure

All tests use **synthetic data** generated via pytest fixtures (no real Sentinel-2 imagery required). Fixtures are defined in `tests/conftest.py`.

| Test File | What It Tests | Synthetic Data |
|---|---|---|
| `test_image_split.py` | Tile count, dimensions, filename pattern, pixel content preservation, error handling | 500×500 RGB image → 4 tiles of 250×250 |
| `test_color_segment.py` | Output creation, dimension preservation, HSV classification (ice/water), error handling | 250×250 solid-color tiles |
| `test_image_merge.py` | Roundtrip split→merge reconstruction, output dimensions, invalid tile handling, partial merges | 500×500 → split → merge |
| `test_preprocess_data.py` | Output file creation, train/test split ratios, array shapes (4D), normalization, one-hot encoding | 4× 256×256 grayscale images + masks |
| `test_model.py` | Input/output shapes, custom classes, custom dimensions, forward pass validity, uncompiled state | NumPy random arrays |
| `test_train_unet.py` | Model + history file creation, history JSON format (keys, epoch count) | Preprocessed .npy arrays, 1 epoch |
| `test_evaluate_model.py` | Result file creation, JSON keys (loss/accuracy/F1/precision/recall), numeric values | Trained model + test data |
| `test_workflow_generator.py` | CLI help, Stage 1 only DAG, Stage 1+2 DAG, job counts, job ID uniqueness, error handling | 500×500 dummy images |
| `test_integration.py` | Full Stage 1 pipeline: split→segment→merge, output dimensions, segmentation changes pixels | 500×500 image with color regions |

### Test Design Principles

1. **Synthetic data only** — No dependency on real satellite imagery. All fixtures generate small images deterministically using fixed random seeds.
2. **Subprocess isolation** — Wrapper scripts are tested via `subprocess.run()`, exactly as Pegasus/HTCondor would invoke them. This validates argument parsing, file I/O, and exit codes.
3. **Fast by default** — Stage 1 tests (split/segment/merge) run in seconds. Training tests use 1 epoch with batch_size=2 on tiny 256×256 data.
4. **Roundtrip verification** — The split→merge test verifies pixel-perfect reconstruction. The integration test verifies the full pipeline changes pixel values (segmentation was applied).
5. **Workflow DAG validation** — `test_workflow_generator.py` verifies the correct number of jobs, unique IDs, and presence of all job types without requiring Pegasus to be installed (parses the output YAML).

### Manual Integration Test

The `run_manual.sh` script provides a bash-based end-to-end test that:
1. Generates a 250×250 synthetic image and 4 synthetic training samples
2. Runs all 7 pipeline steps sequentially with small parameters (tile_size=125, 2 epochs)
3. Verifies all output files are created (including plot PNGs and per-class metrics JSON)

This is useful for validating the full pipeline including Stage 2 training and plot generation on a local machine before Pegasus submission.
