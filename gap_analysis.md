# Gap Analysis — Workflow vs. Paper vs. Reference Code

**Date:** 2026-06-08 (updated 2026-06-17)
**Workflow:** `s2-segmentation-workflow/`
**Latest results:** see `comparison_report.md` / `comparison_report.html` for the consolidated
Run A (scene filter) vs Run B (tile filter) vs paper comparison, including full Fig 13 confusion
matrices and the variance-vs-filter-scale analysis.
**Paper:** Iqrah, Wang, Xie, Prasad — *"A Parallel Workflow for Polar Sea-Ice Classification
using Auto-labeling of Sentinel-2 Imagery,"* IEEE IPDPSW 2024.
**Reference code:** `../S2_Parallel_Workflow/` (`parallel_segmentation.py`,
`s2_u_net_tf.py`, `s2_u_net_horovod.py`, two Jupyter notebooks).

This file enumerates what the paper and/or reference code contain that our Pegasus
workflow does **not** yet implement, ranked by importance. Items marked "fully matched"
are not listed here in detail — they are summarised in §4.

---

## 1. Critical / High priority

These gaps block full reproducibility of the paper's claims.

### 1.1 No inference pipeline (paper Fig 9) — ✅ resolved

- **Paper:** Fig 9 (page 1021) describes the production path: original scene →
  split into 256×256 tiles → thin-cloud/shadow filter → U-Net model → merge
  predictions into a per-scene sea-ice classification map.
- **Reference code:** `s2_u_net_tf.py` lines 303–312 only evaluates on the
  held-out test set; no standalone inference script.
- **Implemented in:** `bin/infer_unet.py` + `--infer` / `--infer-images`
  flags in `workflow_generator.py`. The script loads a trained model,
  optionally re-applies `only_shadow_cloud_removal` (matching the trained
  branch), tiles the scene into 256² patches, runs `model.predict`,
  recolours predictions using the metadata's class→colour mapping, and
  merges back into a full RGB segmentation PNG. With `--paths both
  --infer`, one inference job is emitted per (branch, scene) pair —
  filtered-branch jobs carry `--filter`, orig-branch jobs do not.
- **Smoke-tested** on pegasus2 with the run0009 `model_filtered.hdf5` over
  `s2_vis_00.png`: 2.9 s for 64 tiles end-to-end; output predictions match
  the qualitative style of paper Fig 14.

### 1.2 No cloud-coverage stratified validation (paper Table V, Fig 13) — ✅ resolved

- **Paper:** Table V and Fig 13 split the test set into "≥10% cloud and
  shadow" vs "<10% cloud and shadow" and report 4 confusion matrices per
  model (U-Net-Man and U-Net-Auto × 2 strata × original/filtered).
- **Reference code:** No stratification logic.
- **Implemented in:**
  - `bin/compute_cloud_fraction.py` — per-scene cloud/shadow fraction
    JSON, computed from the same Otsu intermediate mask used inside
    `only_shadow_cloud_removal` (so the "cloud/shadow" definition matches
    the filter exactly).
  - `bin/preprocess_data.py` extension — accepts repeated
    `--cloud-fraction <json>` and emits `test_cloud_fractions.npy` aligned
    with `X_test`. The filtered branch reuses the raw-scene fractions by
    stripping the `train_imgf_` prefix, so both branches share an
    apples-to-apples cloud-coverage estimate.
  - `bin/evaluate_stratified.py` — splits the test set at the configured
    threshold (default `0.10`), evaluates each stratum, emits Fig 13-style
    confusion matrices + metrics tables + per-class JSON + a single
    `stratified_summary.json` per branch (Table V row).
  - `workflow_generator.py` — new `--stratified-eval` and
    `--cloud-threshold` flags wire one `compute_cloud_fraction` job per
    scene and one `evaluate_stratified` job per branch.
- **Smoke-tested** on pegasus2 in the workflow container: 7.6% scene-wide
  cloud/shadow on `s2_vis_00`, 12/64 tiles in the ≥10% bucket; `evaluate_stratified`
  ran end-to-end on a synthetic 20-tile test set and produced both strata's
  outputs in ~7 s.

### 1.3 No PySpark Map-Reduce auto-labeling (paper §B, Table II)

- **Paper:** §B and Table II report a 16.25× speedup from a 4-node
  Google-Cloud Dataproc PySpark cluster doing per-tile color segmentation.
- **Reference code:** `spark_unet_parallel.ipynb` is a notebook only.
- **Our workflow:** Zero Spark support. Auto-labeling is parallelised by
  HTCondor (one job per tile), which is arguably superior on a cluster but
  does not let us reproduce the paper's Spark-specific numbers.
- **Effort to add:** **Large, low ROI.** Would require packaging PySpark
  into the container and submitting it through Pegasus. Probably not worth
  the cost — the paper's main result uses the multiprocessing path, which
  our HTCondor parallelism subsumes.

---

## 2. Medium priority

### 2.1 No SSIM metric for auto-label quality

- **Paper:** §B (page 1022): *"we achieved 89% and 99.64% Structural
  Similarity Index (SSIM) precision over the manually labeled data."*
- **Reference code:** Not present.
- **Our workflow:** Not computed. We report classification accuracy and
  per-class F1 against the auto-labels, but no image-level similarity
  against a manually-labeled reference.
- **Effort to add:** **Small.** `bin/compute_ssim.py` using
  `skimage.metrics.structural_similarity` over (auto-label tile,
  manual-label tile) pairs. Requires a manually-labeled directory as input.

### 2.2 No separate manually-labeled validation set (U-Net-Man path)

- **Paper:** Trains *two* U-Nets — U-Net-Man on manually-labeled data and
  U-Net-Auto on color-segmentation auto-labels — then compares accuracy on
  a shared validation set. Table IV's U-Net-Man column needs this.
- **Reference code:** Reads training images/masks from fixed directories;
  manual vs auto labels are interchangeable as long as the dir contents
  differ.
- **Our workflow:** Only trains U-Net-Auto. There is no
  `--manual-validation-dir` parameter or job for the U-Net-Man baseline.
- **Effort to add:** **Small–medium.** Plumb an optional manually-labeled
  directory through `preprocess_data.py` and add a parallel training
  branch (suffix `_man`).

### 2.3 Dataset size: 63 vs 66 scenes

- **Paper:** 66 scenes / 4,224 tiles.
- **Our workflow:** 63 scenes / 4,032 tiles — missing
  `s2_vis_56/57/64.png`. Now **documented** in README and SPEC §6: the
  reference scripts themselves load `train_images_4032/` (63 scenes), so 63/4032
  matches the reference code even though the paper text says 66/4224.
- **Effort to close fully:** **Small.** Recover the missing three scenes from GEE
  (`download_data.py`) if exact paper-text parity is required.

### 2.4 Filter applied at full-scene scale, not per tile — ✅ resolved

- **Paper / reference code:** Ambiguous, but the Spark path appears to
  apply `only_shadow_cloud_removal` per 256×256 tile.
- **Implemented in:** new `--filter-scale {scene,tile}` and
  `--filter-kernel-size N` flags in `workflow_generator.py`.
  - `filter_image.py` and `compute_cloud_fraction.py` both now expose
    `--kernel-size` and route it to the `medianBlur` call so the same
    code path works at either scale.
  - At `--filter-scale tile`, the per-scene `filter_image` job is
    replaced with one `filter_image` job per 256×256 training tile
    (downstream of `split_images`), and `train_imgf_*.png` outputs feed
    directly into the filtered branch's auto-label / Stage 2 chain.
  - The kernel auto-defaults to 155 at scene scale (paper's value) and
    19 at tile scale (scaled to stay the same fraction of the input
    dimension).
- **Smoke-tested** on pegasus2 in the workflow container: 64 per-tile
  filter jobs emitted for one scene under `--filter-scale tile`, and a
  single 256×256 tile filters successfully with `--kernel-size 19`.

---

## 3. Low priority

### 3.1 No auto-labeling parallel-speedup benchmarks (paper Table I, Fig 10)

- **Paper:** Table I / Fig 10 report the **multiprocessing** speedup of the
  color-segmentation auto-labeling step (4.5× at 8 processes on one machine);
  Table II reports the PySpark variant (see §1.3).
- **Our workflow:** Auto-labeling is parallelised by HTCondor (one job per
  tile), not Python `multiprocessing`, so these single-machine speedup numbers
  are not reproduced. We rely on `pegasus-statistics` for runtime accounting.
- **Note:** `bin/generate_speedup_plot.py` already exists, but it covers the
  **distributed-training** speedup (Fig 12, §3.2) — it aggregates per-epoch
  `training_history` files, *not* the auto-labeling multiprocessing speedup.
- **Effort to add:** **Small, low ROI.** Time the multiprocessing path from
  `parallel_segmentation.py` at varying worker counts; not a Pegasus-DAG concern.

### 3.2 No reporting of training-throughput per epoch (paper Fig 12) — ✅ resolved

- **Paper:** Fig 12 plots distributed-training speedup, data/sec, total
  time, and time-per-epoch over 1/2/4/6/8 GPUs.
- **Implemented in:**
  - `bin/train_unet.py` — new `EpochTimer` Keras callback records
    `epoch_time_seconds`, `samples_per_second`, and a `training_meta`
    block (mode / replicas / batch_size / samples_per_epoch / epochs)
    into `training_history{_branch}.json`. Works for all three modes
    (single-gpu / MirroredStrategy / Horovod) with the correct replica
    count and Horovod's `steps_per_epoch` semantics.
  - `bin/generate_speedup_plot.py` — aggregates N history files (one per
    GPU count) into a 4-subplot PNG matching paper Fig 12 (speedup vs
    ideal, samples/sec bars, total time, time/epoch) plus a
    `speedup_summary.csv` for direct paper inclusion. Backward-compatible
    with older history files that lack the new fields (degrades to
    total-time-only).
- **Smoke-tested:** `train_unet --epochs 2 --mode single-gpu` emits the
  new fields cleanly (e.g. epoch 1 7.62 s, epoch 2 3.91 s — TF warm-up
  visible as expected); speedup script renders correctly with both new
  and legacy histories.

---

## 4. Fully matched (verified)

These were checked line-by-line against the reference; **no gap**:

| Component | Status |
|---|---|
| U-Net architecture (6 levels, 16→512 filters, Conv2DTranspose, softmax) | ✅ identical |
| Dropout rates (0.1 / 0.1 / 0.2 / 0.2 / 0.2 / 0.3) | ✅ identical |
| HSV thresholds (thick ice / thin ice / open water) | ✅ identical |
| `only_shadow_cloud_removal` — all 9 steps (dilate → medianBlur(155) → absdiff → Otsu → bitwise → min-max norm → truncated threshold → HSV re-threshold) | ✅ identical |
| Custom Keras metrics (`f1_m`, `precision_m`, `recall_m`) | ✅ identical |
| Optimizer / loss / epochs / batch / shuffle / split | ✅ Adam, categorical_crossentropy, 50, 32, `shuffle=False`, `random_state=0` |
| Normalization | ✅ `keras normalize(axis=1)` (L2 per-sample) |
| sklearn `classification_report` + confusion matrix + paper-style plots | ✅ via `generate_plots.py` |
| Multiple training modes | ✅ single-GPU + MirroredStrategy + Horovod (exceeds the reference, which only has one of each) |

---

## 5. Recommended next step

§1.1, §1.2, §2.4, and §3.2 are now closed. The next-highest-value addition is
**§2.1 (SSIM)** — small (`skimage.metrics.structural_similarity` over auto-label vs
manual-label tile pairs) and would add a new headline number for the
auto-labeling-quality claim ("89% / 99.64% SSIM"). After that, §2.2
(manually-labeled validation set / U-Net-Man baseline) is the only remaining
gap that blocks full reproducibility of Table IV's *other* column.

---

_Generated as a one-off audit; re-run by hand if the workflow changes significantly._
