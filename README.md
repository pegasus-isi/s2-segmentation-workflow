# S2 Segmentation Workflow

A [Pegasus WMS](https://pegasus.isi.edu/) workflow for **Sentinel-2 satellite sea ice segmentation**, executed on an [HTCondor](https://htcondor.org/) pool.

## Pipeline Overview

The workflow combines two stages into an end-to-end DAG:

**Stage 1 — Color Segmentation (Label Generation)**

1. **image_split** — Splits each 2000×2000 Sentinel-2 PNG into 64 tiles of 250×250. One job per source image, all run concurrently.
2. **color_segment** — HSV-based color segmentation on each tile (ice/thin-ice/water classification). One job per tile — N×64 embarrassingly parallel HTCondor jobs.
3. **image_merge** — Reassembles 64 segmented tiles back into a full 2000×2000 image. One merge per source image (fan-in).

**Auto-label Bridge (with `--auto-label`)**

3b. **split_images** — Splits each 2000×2000 source scene into 256×256 grayscale training image tiles (zero-padded at edges since 2000 is not divisible by 256). Reuses `image_split` with `--grayscale --pad`. One job per source image.
3c. **split_masks** — Splits each 2000×2000 merged segmentation mask into 256×256 grayscale mask tiles (same grid as split_images). One job per source image. Together with split_images, these produce matched image/mask tile pairs for Stage 2.

**Stage 2 — U-Net Training & Evaluation (optional)**

4. **preprocess_data** — Loads 256×256 grayscale training images and masks (from auto-label tiles or `--train-images-dir`/`--train-masks-dir`), encodes labels, normalizes (L2, float32), performs 80/20 train/test split. Processes each split separately for memory efficiency. Outputs `.npy` arrays.
5. **train_unet** — Trains a 6-level U-Net (16→512 filters, 3-class softmax, categorical crossentropy, Adam optimizer). Supports single-GPU, multi-GPU (MirroredStrategy), and multi-node (Horovod) training modes.
6. **evaluate_model** — Evaluates the trained model on the test set. Outputs loss, accuracy, F1, precision, and recall.
7. **generate_plots** — Produces publication figures and tables: training curves, confusion matrix (Fig 13), prediction samples (Fig 14), metrics table (Table IV), and per-class metrics JSON.

```
  Image 0                    Image 1                    Image N-1
  ────────                   ────────                   ─────────
  image_split_0              image_split_1      ...     image_split_N-1
  ┌──┬──┬─...─┐              ┌──┬──┬─...─┐              ┌──┬──┬─...─┐
  seg seg seg seg            seg seg seg seg            seg seg seg seg
  (0) (1)(2) (63)            (0) (1)(2) (63)            (0) (1)(2) (63)
  └──┴──┴─...─┘              └──┴──┴─...─┘              └──┴──┴─...─┘
       │                          │                          │
  image_merge_0              image_merge_1             image_merge_N-1
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
                         train_unet
                              │
                              ▼
                       evaluate_model
                              │
                              ▼
                       generate_plots
```

![Workflow DAG](images/workflow.png)

> **Note**: The `split_images_*` and `split_masks_*` jobs (shown in brackets) only appear when `--auto-label` is used. They produce matched training image/mask tile pairs directly from the source scenes. Without `--auto-label`, Stage 2 reads pre-existing files from `--train-images-dir` and `--train-masks-dir`.

## Project Structure

```
s2-segmentation-workflow/
├── workflow_generator.py       # Pegasus DAG generator
├── bin/
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
├── tests/                      # pytest test suite
│   ├── conftest.py             # Shared fixtures (synthetic data)
│   ├── test_image_split.py
│   ├── test_color_segment.py
│   ├── test_image_merge.py
│   ├── test_preprocess_data.py
│   ├── test_model.py
│   ├── test_train_unet.py
│   ├── test_evaluate_model.py
│   ├── test_workflow_generator.py
│   └── test_integration.py
├── download_data.py            # Sentinel-2 data download script (GEE)
├── run_manual.sh               # Bash-based local integration test
├── specification.md            # Detailed workflow specification
├── requirements.txt            # Python dependencies
└── README.md
```

## Prerequisites

- Python 3.8+
- [Pegasus WMS](https://pegasus.isi.edu/) (for workflow generation and submission)
- [HTCondor](https://htcondor.org/) (execution backend)

```bash
pip install -r requirements.txt
```

For Horovod multi-node training (optional):

```bash
pip install horovod[tensorflow]
```

## Data

The workflow uses **Sentinel-2 optical imagery** from ESA's Copernicus program, collected via [Google Earth Engine](https://earthengine.google.com/). The reference dataset covers the **Antarctic Ross Sea** during the summer season (November 2019):

| Parameter | Value |
|---|---|
| Region | Ross Sea, Antarctica |
| Latitude | -70.00 to -78.00 (south) |
| Longitude | -140.00 to -180.00 (west) |
| Time period | November 2019 |
| Bands | B4 (red), B3 (green), B2 (blue) |
| Resolution | 10m per pixel |
| Scenes | 66 large scenes |
| Training tiles | 4,224 images of 256×256 pixels |

> Source: Iqrah et al., *"A Parallel Workflow for Polar Sea-Ice Classification using Auto-Labeling of Sentinel-2 Imagery,"* IEEE IPDPSW 2024. DOI: [10.1109/IPDPSW63119.2024.00172](https://doi.org/10.1109/IPDPSW63119.2024.00172)

### Downloading the Data

A download script is provided that uses the Google Earth Engine Python API:

```bash
# 1. Install the GEE API
pip install earthengine-api
pip install -r requirements.txt

# 2. Download and split into 256x256 training tiles
python download_data.py --method local --output-dir data/s2_scenes --split-tiles
    
python download_data.py --method local --output-dir data/s2_scenes --split-tiles --max-scenes 10

# Export to Google Drive (recommended for large downloads)
python download_data.py --project ee-yourproject \
    --method drive --drive-folder s2_ross_sea
```

After downloading, your data directory should look like:

```
data/
└── s2_scenes/          # Full 2000×2000 scene PNGs (workflow input)
    ├── s2_vis_00.png
    ├── s2_vis_01.png
    └── ...
```

> **Note**: With `--auto-label` (recommended), **no separate training data directories are needed**. The workflow produces everything within the DAG: `split_images` jobs tile each source scene into 256×256 grayscale training images, and `split_masks` jobs tile the Stage 1 segmentation masks into matching 256×256 grayscale labels (zero-padded at edges). Both use the same grid so image/mask counts always match. This is the auto-labeling approach described in the paper. If you have external ground-truth data, you can skip `--auto-label` and provide pre-existing tiles via `--train-images-dir` and `--train-masks-dir`.

### Using Synthetic Test Data

For local testing **without real Sentinel-2 data**, the test suite and `run_manual.sh` generate synthetic images automatically:

```bash
# Bash-based integration test with synthetic data
bash run_manual.sh

# pytest suite with synthetic fixtures
pytest tests/ -v
```

## Usage

### 1. Build the Container Image

The workflow runs inside a Singularity/Docker container. Build and push the image before submitting:

```bash
docker build -t kthare10/s2-segmentation:latest -f Docker/S2_Dockerfile .
docker push kthare10/s2-segmentation:latest
```

### 2. Generate and Submit the Workflow

**Quick start — 2 images (for testing / small runs):**

```bash
# Auto-label with just 2 images (fast — 128 segment jobs + training)
python workflow_generator.py \
    --images data/s2_scenes/s2_vis_00.png data/s2_scenes/s2_vis_01.png \
    --auto-label \
    --output workflow.yml

# Submit
pegasus-plan --submit -s condorpool -o local workflow.yml
```

**Full dataset — all 66 images (production run):**

```bash
# Auto-label (recommended) — single DAG that runs Stage 1, then splits
# both source scenes and merged masks into matched 256×256 grayscale
# tile pairs, and feeds them into Stage 2 training. No external data
# directories needed — everything is produced within the workflow.
# With 66 images this creates 66×64 = 4,224 parallel segment jobs.
#
# Defaults: --paths both --filtered-labels filtered
#   → trains BOTH the unfiltered and the thin-cloud/shadow-filtered
#     U-Net in one DAG (paper Table IV columns) with self-consistent
#     filtered labels (Option A — reproduces the paper's ~99%).
python workflow_generator.py \
    --images data/s2_scenes/s2_vis_*.png \
    --auto-label \
    --output workflow.yml

pegasus-plan --submit -s condorpool -o local workflow.yml
```

**Only one path, or different label-derivation strategy:**

```bash
# Unfiltered branch only
python workflow_generator.py --images data/s2_scenes/s2_vis_*.png \
    --auto-label --paths orig --output workflow.yml

# Filtered branch only, with the honest cross-comparison
# (filtered input vs raw-scene labels — yields ~90%, exposing that the
# paper's 98.97% requires label-consistency)
python workflow_generator.py --images data/s2_scenes/s2_vis_*.png \
    --auto-label --paths filtered --filtered-labels raw \
    --output workflow.yml
```

See `comparison_report.md` for a side-by-side of every run mode against
the paper's reported numbers (U-Net-Auto: 90.18% original, 98.97% filtered).

**Horovod distributed training (multi-node GPU):**

```bash
# Auto-label + Horovod — uses multiple GPUs across nodes for training.
# Requires the container image built with Horovod support (see step 1).
python workflow_generator.py \
    --images data/s2_scenes/s2_vis_*.png \
    --auto-label \
    --training-mode horovod \
    --output workflow.yml

# With pre-existing masks + Horovod
python workflow_generator.py \
    --images data/s2_scenes/s2_vis_*.png \
    --train-images-dir data/train_images/ \
    --train-masks-dir data/train_masks/ \
    --training-mode horovod \
    --output workflow.yml

pegasus-plan --submit -s condorpool -o local workflow.yml
```

**Stage 1 only (no training):**

```bash
# Color segmentation only — produces one 2000×2000 merged mask per
# input image (e.g. s2_vis_00_seg.png). Does NOT produce 256×256
# training tiles; use --auto-label for that.
python workflow_generator.py \
    --images data/s2_scenes/s2_vis_*.png \
    --output workflow.yml

pegasus-plan --submit -s condorpool -o local workflow.yml
```

**With pre-existing masks (no auto-label):**

```bash
# Use this only when you already have a directory of 256×256 mask
# tiles (e.g. from external ground-truth labels)
python workflow_generator.py \
    --images data/s2_scenes/s2_vis_*.png \
    --train-images-dir data/train_images/ \
    --train-masks-dir data/train_masks/ \
    --output workflow.yml
```

### 3. Workflow Generator Options

| Option | Default | Description |
|---|---|---|
| `--images` | (required) | Input Sentinel-2 PNG images |
| `--tile-size` | 250 | Tile dimension in pixels |
| `--original-size` | 2000 | Original image dimension |
| `--auto-label` | off | Single-DAG mode: splits source scenes + masks into matched 256×256 tiles for Stage 2 (no external dirs needed) |
| `--paths` | both | Which auto-label training path(s) to run: `both` (orig + thin-cloud/shadow-filtered, paper Table IV), `orig`, or `filtered`. With `both`, outputs are suffixed `_orig` / `_filtered`. |
| `--filtered-labels` | filtered | How the filtered branch's labels are produced. `filtered` color-segments the *filtered* tiles so input and target are self-consistent (reproduces the paper's ~99%). `raw` reuses raw-scene labels (filtered input vs raw target — the honest cross-comparison, ~90%). |
| `--infer` | off | After training, run `infer_unet` end-to-end on every scene (paper Fig 9): tile → optional filter → predict → merge → colour-coded prediction PNG. The filtered branch passes `--filter` so inference matches its training distribution. Outputs are named `{orig,filtered}_infer_<scene>.png`. |
| `--infer-images` | (same as `--images`) | Override the scenes used for inference — useful for predicting on fresh scenes that weren't part of the training corpus. |
| `--train-images-dir` | None | Training images directory (enables Stage 2; not needed with `--auto-label`) |
| `--train-masks-dir` | None | Training masks directory (enables Stage 2; not needed with `--auto-label`) |
| `--training-mode` | single-gpu | Training mode: `single-gpu`, `mirrored`, or `horovod` |
| `--epochs` | 50 | Training epochs |
| `--batch-size` | 32 | Training batch size |
| `--n-classes` | 3 | Segmentation classes |
| `--container-image` | kthare10/s2-segmentation:latest | Docker container image |
| `--execution-site-name` | condorpool | CPU execution site |
| `--gpu-site-name` | gpu-condorpool | GPU execution site |

### Local Testing

Run the bash-based manual test (requires TensorFlow):

```bash
bash run_manual.sh
```

Run the pytest suite:

```bash
# All tests (skips TF/Pegasus tests if not installed)
pytest tests/ -v

# Fast tests only (Stage 1 — no TensorFlow required)
pytest tests/ -v -k "not train and not evaluate and not preprocess and not model and not workflow"

# Stage 2 tests (requires TensorFlow)
pytest tests/test_preprocess_data.py tests/test_model.py tests/test_train_unet.py tests/test_evaluate_model.py -v

# Workflow generator tests (requires Pegasus)
pytest tests/test_workflow_generator.py -v
```

## Outputs

With `--paths both` (default), Stage 2 artifacts are produced for **both** branches and
suffixed/prefixed as shown below. With `--paths orig` or `--paths filtered` alone, only
the corresponding suffix is emitted (no suffix when no auto-label is used).

| File | Description |
|---|---|
| `{basename}_seg.png` | Stage 1 merged segmentation mask (2000×2000, per source image) |
| `filtered_{basename}.png` | Thin-cloud/shadow-filtered source scene (when `--paths both` or `filtered`) |
| `model_orig.hdf5`, `model_filtered.hdf5` | Trained U-Net weights (one per branch) |
| `training_history_{orig,filtered}.json` | Loss/accuracy/F1 per epoch + training time |
| `evaluation_results_{orig,filtered}.json` | Test loss, accuracy, F1, precision, recall |
| `training_curves.png`, `filtered_training_curves.png` | Loss/accuracy/F1/precision-recall curves |
| `confusion_matrix.png`, `filtered_confusion_matrix.png` | Normalized confusion matrix (paper Fig 13) |
| `prediction_samples.png`, `filtered_prediction_samples.png` | Side-by-side input/truth/prediction grid (paper Fig 14) |
| `metrics_table.png`, `filtered_metrics_table.png` | Classification metrics table (paper Table IV) |
| `per_class_metrics.json`, `filtered_per_class_metrics.json` | Per-class precision, recall, F1-score, support |

## Reproducing the Paper's Table IV

| Condition | Paper | This workflow (run0009, default settings) |
|---|---|---|
| U-Net-Auto, original | 90.18% accuracy | **94.72%** accuracy |
| U-Net-Auto, filtered | 98.97% accuracy | **99.82%** accuracy |

See `comparison_report.md` / `comparison_report.html` for the full reproduction study,
per-class metrics, and a discussion of the (previously undocumented) requirement that
the filtered condition re-derive labels from the filtered images.
