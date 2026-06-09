# Gap Analysis — Workflow vs. Paper vs. Reference Code

**Date:** 2026-06-08
**Workflow:** `s2-segmentation-workflow/`
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

### 1.2 No cloud-coverage stratified validation (paper Table V, Fig 13)

- **Paper:** Table V and Fig 13 split the test set into ">10% cloud and
  shadow" vs "<10% cloud and shadow" and report 4 confusion matrices per
  model (U-Net-Man and U-Net-Auto × 2 strata × original/filtered).
- **Reference code:** No stratification logic — labels are taken as-is.
- **Our workflow:** `preprocess_data.py` does an 80/20 random split with
  `random_state=0`; there is no per-tile cloud-fraction estimate and no
  stratified evaluation.
- **Effort to add:** **Medium.** Compute a per-tile cloud fraction (reuse
  the water/HSV masks or threshold the filter's intermediate Otsu output),
  emit `cloud_fraction.npy` alongside test labels, and add a stratified
  reporting pass in `evaluate_model.py` / `generate_plots.py`.

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
  `s2_vis_56/57/64.png`. There is no assertion that exactly 66 are present
  and no documentation of the missing three.
- **Effort to add:** **Small.** A log line + soft warning in
  `preprocess_data.py`; recover the missing scenes from GEE
  (`download_data.py`).

### 2.4 Filter applied at full-scene scale, not per tile

- **Paper / reference code:** Ambiguous, but the Spark path appears to
  apply `only_shadow_cloud_removal` per 256×256 tile.
- **Our workflow:** `filter_image.py` runs on the full 2048×2048 scene
  before `split_images` tiles it. `medianBlur(155)` behaves very
  differently at the two scales.
- **Already acknowledged** in `comparison_report.md` §6 (difference #2).
- **Effort to add:** **Medium.** Move `filter_image` to a per-tile job
  downstream of `split_images`, or add `--filter-scale {scene,tile}` as a
  switch in `workflow_generator.py`.

---

## 3. Low priority

### 3.1 No Spark scaling benchmarks (paper Tables I/II, Fig 10)

- **Paper:** Reports parallel-execution speedup figures (Fig 10) and
  scalability on the GCD cluster (Table II).
- **Our workflow:** We rely on Pegasus's own statistics (`pegasus-statistics`)
  but don't produce paper-style speedup plots.
- **Effort to add:** **Small.** A `bin/generate_speedup_plot.py` consuming
  `pegasus-statistics` output.

### 3.2 No reporting of training-throughput per epoch (paper Fig 12)

- **Paper:** Fig 12 plots distributed-training speedup, data/sec, total
  time, and time-per-epoch over 1/2/4/6/8 GPUs.
- **Our workflow:** `training_history_*.json` records `loss`/`accuracy`
  per epoch and `training_time_seconds`, but not data/sec or per-GPU
  scaling.
- **Effort to add:** **Small** if you re-run the workflow at several GPU
  counts and aggregate.

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

§1.1 is now closed (`bin/infer_unet.py` + `--infer` flag). The next-highest-value
addition is **§1.2 (cloud-coverage stratification)** because it unlocks Table V and
the per-stratum panels of Fig 13. After that, §2.1 (SSIM) is small and would add a
new headline number for the auto-labeling quality claim.

---

_Generated as a one-off audit; re-run by hand if the workflow changes significantly._
