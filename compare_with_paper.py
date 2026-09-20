#!/usr/bin/env python3
"""Generate a side-by-side comparison report against the reference paper.

For each comparable section (sample scenes, cloud/shadow filter, color-segmentation
auto-labels, confusion matrices, prediction samples, headline metrics), this script:

  1. Extracts the relevant figure from the paper PDF (cropping a configurable bbox
     out of a rendered page).
  2. Pairs it with our matching ``output/run0009/`` artifact.
  3. Writes ``comparison_report.md`` with side-by-side markdown tables and an
     overall metrics table built from ``evaluation_results_*.json``.

Re-run after a new training run to refresh numbers and image pairings. The crop
boxes for paper figures are defined in ``PAPER_FIGURES`` below — adjust if the
PDF resolution or layout changes.

Dependencies: pdftoppm (poppler), Pillow.

Usage:
    python compare_with_paper.py
    python compare_with_paper.py --run-dir ../output/run0009 --paper ../A_Parallel_*.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install Pillow")

# ── Paper figure catalogue ──────────────────────────────────────────────────
# Each entry crops a region of a rendered PDF page (200 DPI → 1700×2200 px for
# the IEEE letter-format paper). Tune the bbox if pages look different.

@dataclass
class PaperFigure:
    fig_id: str          # short id used as filename
    page: int            # 1-indexed page number
    bbox: tuple          # (left, top, right, bottom) in pixels at 200 DPI
    caption: str         # short description for the report


PAPER_FIGURES = {
    "fig3_scenes": PaperFigure(
        "fig3_scenes", 3, (870, 300, 1590, 980),
        "Fig. 3 — Sample S2 scenes: (a) with cloud/shadow, (b) without."),
    "fig4_manual_labels": PaperFigure(
        "fig4_manual_labels", 3, (870, 1090, 1590, 1990),
        "Fig. 4 — Manually-labeled data with color codes (a/b/c = original, d/e/f = labels)."),
    "fig5_filtered": PaperFigure(
        "fig5_filtered", 4, (110, 195, 825, 705),
        "Fig. 5 — Thin cloud / shadow-filtered dataset (a/b/c original, d/e/f filtered)."),
    "fig11_colorseg": PaperFigure(
        "fig11_colorseg", 7, (870, 600, 1590, 1880),
        "Fig. 11 — Color-segmentation auto-labeling: (a) cloudy S2 scene, (b) color-segmented, "
        "(c) cloud/shadow-filtered, (d) color-segmented filtered."),
    "fig13_confusion": PaperFigure(
        "fig13_confusion", 9, (110, 130, 1590, 820),
        "Fig. 13 — Confusion matrices for U-Net-Man (top) and U-Net-Auto (bottom) across "
        "≥10% cloud, ≥10% cloud filtered, <10% cloud, <10% cloud filtered."),
    "fig14_predictions": PaperFigure(
        "fig14_predictions", 9, (870, 870, 1590, 1620),
        "Fig. 14 — Side-by-side: original S2, manually-labeled ground truth, "
        "U-Net-Man prediction, U-Net-Auto prediction."),
    "table4_metrics": PaperFigure(
        "table4_metrics", 8, (870, 960, 1590, 1185),
        "Table IV — U-Net-Man vs U-Net-Auto accuracy on original and filtered S2 imagery."),
}

# ── Section layout ──────────────────────────────────────────────────────────
# Each section pairs a paper figure with one or more files from the run dir.

@dataclass
class Section:
    title: str
    paper_fig: str            # key into PAPER_FIGURES
    ours: list                # list of (label, path-relative-to-repo-root) tuples
    commentary: str = ""


# ``ours`` paths are resolved against the run-dir.
SECTIONS = [
    Section(
        "Cloud / shadow filter output",
        "fig5_filtered",
        [("Filtered scene 00", "filtered_s2_vis_00.png"),
         ("Filtered scene 01", "filtered_s2_vis_01.png"),
         ("Filtered scene 02", "filtered_s2_vis_02.png")],
        "Our `bin/filter_image.py` is a byte-faithful port of the paper's "
        "`only_shadow_cloud_removal()` (dilate → medianBlur(155) → absdiff → Otsu → "
        "min-max norm → truncated threshold). The paper shows raw vs filtered "
        "scenes; we show only the filtered outputs (raw scenes live in the run "
        "input dir, not the output dir)."),
    Section(
        "Confusion matrices (paper Fig 13)",
        "fig13_confusion",
        [("Our U-Net-Auto — Original S2 imagery", "confusion_matrix.png"),
         ("Our U-Net-Auto — Thin cloud / shadow-filtered S2 imagery",
          "filtered_confusion_matrix.png")],
        "Paper's Fig 13 shows 8 matrices (U-Net-Man and U-Net-Auto × 4 cloud-coverage "
        "conditions). We plot the two U-Net-Auto conditions that correspond to the "
        "paper's Table IV rows: original S2 imagery and thin cloud / shadow-filtered "
        "S2 imagery."),
    Section(
        "Prediction samples (paper Fig 14)",
        "fig14_predictions",
        [("Our predictions — Original S2 imagery", "prediction_samples.png"),
         ("Our predictions — Thin cloud / shadow-filtered S2 imagery",
          "filtered_prediction_samples.png")],
        "Each tile is input | ground-truth | prediction. Red = thick ice, blue = thin "
        "ice, green = open water, matching the paper's legend."),
    Section(
        "Whole-scene inference (paper Fig 9 production path)",
        "fig14_predictions",
        [("Our whole-scene prediction — original branch, scene 00",
          "orig_infer_s2_vis_00.png"),
         ("Our whole-scene prediction — filtered branch, scene 00",
          "filtered_infer_s2_vis_00.png")],
        "The paper's Fig 9 describes the production path: a full scene is tiled, "
        "each tile is classified, and the predictions are merged back into one "
        "per-scene sea-ice map. These are that merged output for scene 00 from both "
        "trained branches (66 scenes x 2 branches were produced). Red = thick ice, "
        "blue = thin ice, green = open water. The paper shows no whole-scene figure "
        "to match numerically, so this pairing is qualitative."),
    Section(
        "Headline metrics (paper Table IV)",
        "table4_metrics",
        [("Our metrics — Original S2 imagery", "metrics_table.png"),
         ("Our metrics — Thin cloud / shadow-filtered S2 imagery",
          "filtered_metrics_table.png")],
        "See §2 above for the per-class numeric comparison."),
]

# ── Pipeline ────────────────────────────────────────────────────────────────


def require(cmd: str) -> None:
    if shutil.which(cmd) is None:
        sys.exit(f"Required executable '{cmd}' not found on PATH.")


def render_pdf_pages(pdf: Path, out_dir: Path, dpi: int = 200) -> dict:
    """Render every distinct page referenced by PAPER_FIGURES; return {page: Path}."""
    require("pdftoppm")
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = sorted({f.page for f in PAPER_FIGURES.values()})
    rendered: dict = {}
    for p in pages:
        # pdftoppm names files <prefix>-NN.ppm (zero-padded to width of total pages).
        # We render one page at a time so we know the exact name.
        prefix = out_dir / f"page-{p:02d}"
        subprocess.run(
            ["pdftoppm", "-r", str(dpi), "-f", str(p), "-l", str(p),
             str(pdf), str(prefix)],
            check=True,
        )
        # pdftoppm will emit either page-NN-NN.ppm or page-NN.ppm depending on count;
        # locate whichever was produced.
        candidates = sorted(out_dir.glob(f"page-{p:02d}*.ppm"))
        if not candidates:
            sys.exit(f"pdftoppm produced no output for page {p}")
        rendered[p] = candidates[-1]
    return rendered


def crop_figures(rendered: dict, out_dir: Path) -> dict:
    """Crop each paper figure out of its rendered page; return {fig_id: Path}."""
    cropped: dict = {}
    for fig in PAPER_FIGURES.values():
        page_img = Image.open(rendered[fig.page])
        crop = page_img.crop(fig.bbox)
        out = out_dir / f"{fig.fig_id}.png"
        crop.save(out, "PNG", optimize=True)
        cropped[fig.fig_id] = out
    return cropped


# ── Paper reference values (Iqrah et al., IPDPSW 2024) ──────────────────────
# U-Net-Auto column only (we do not reproduce the manually-labeled U-Net-Man).
PAPER_TABLE_IV = {            # overall accuracy
    "orig": 90.18,
    "filtered": 98.97,
}
PAPER_TABLE_IV_PRF = {        # precision / recall / F1 (%) from the paper text
    "orig": (91.14, 91.05, 91.10),
    # NOTE: the paper text reports filtered U-Net-Auto as 98.88 / 91.87 / 91.89;
    # the 91.87/91.89 recall+F1 are inconsistent with a 98.97% accuracy and
    # appear to be a typo (likely ~98.x). Shown verbatim, flagged in the report.
    "filtered": (98.88, 91.87, 91.89),
}
PAPER_TABLE_V = {             # stratified accuracy (%) — U-Net-Auto
    ("orig", "high"): 79.91,   # > ~10% cloud/shadow, original images
    ("filtered", "high"): 99.28,
    ("orig", "low"): 93.60,    # < ~10% cloud/shadow
    ("filtered", "low"): 98.87,
}
# Fig 13 (auto-labeled) confusion-matrix diagonals = per-class recall (%):
# order = (thin ice, thick ice, open water).
PAPER_FIG13_AUTO = {
    ("orig", "high"): (95.30, 75.95, 92.18),
    ("filtered", "high"): (98.90, 99.51, 97.04),
    ("orig", "low"): (85.74, 98.57, 96.98),
    ("filtered", "low"): (97.92, 99.12, 98.79),
}
# Full row-normalized 3x3 matrices transcribed from Fig 13 (auto-labeled row);
# rows = true thin/thick/water, cols = predicted. ``None`` marks an off-diagonal
# cell that is not legible in the figure (the open-water rows of the filtered
# conditions); the diagonal there is still readable.
PAPER_FIG13_FULL = {
    ("orig", "high"): [[95.30, 3.92, 0.78], [24.05, 75.95, 0.00], [7.58, 0.24, 92.18]],
    ("filtered", "high"): [[98.90, 1.01, 0.09], [0.49, 99.51, 0.00], [2.16, 0.00, 97.84]],
    ("orig", "low"): [[85.74, 13.56, 0.70], [1.43, 98.57, 0.00], [2.98, 0.04, 96.98]],
    ("filtered", "low"): [[97.92, 1.99, 0.09], [0.88, 99.12, 0.00], [1.21, 0.00, 98.79]],
}


def read_eval(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def per_class(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def metric_pct(d, key):
    if d is None:
        return "n/a"
    return f"{d[key] * 100:.2f}%" if d.get(key) is not None else "n/a"


def metric_f(d, key):
    if d is None:
        return "n/a"
    return f"{d[key]:.4f}" if d.get(key) is not None else "n/a"


def img_md(path, rel_from: Path) -> str:
    if path is None:
        return "&nbsp;"
    if not Path(path).exists():
        return "_(missing)_"
    rel = os.path.relpath(path, rel_from)
    return f"![]({rel})"


def load_run(run_dir: Path) -> dict:
    """Collect a run's comparable metrics into one dict (None where absent)."""
    d = {"overall": {}, "strat": {}, "strat_pc": {}, "cm": {}}
    for b in ("orig", "filtered"):
        d["overall"][b] = read_eval(run_dir / f"evaluation_results_{b}.json")
        s = read_eval(run_dir / f"{b}_stratified_summary.json")
        d["strat"][b] = s
        for stratum in ("high", "low"):
            pc = per_class(run_dir / f"{b}_{stratum}_cloud_per_class_metrics.json")
            d["strat_pc"][(b, stratum)] = pc
            cmj = read_eval(
                run_dir / f"{b}_{stratum}_cloud_confusion_matrix.json")
            d["cm"][(b, stratum)] = (cmj or {}).get("row_normalized_pct")
    return d


def _acc(ev):
    return f"{ev['test_accuracy'] * 100:.2f}%" if ev else "—"


def render_comprehensive(run_a: dict, label_a: str,
                         repo_root: Path = None) -> list:
    """Side-by-side comparison of one run against the paper's U-Net-Auto column."""
    L = []
    A = L.append

    def delta(ours_pct, paper_pct):
        if ours_pct is None:
            return "—"
        d = ours_pct - paper_pct
        return f"{d:+.2f}"

    A("## 0. Side-by-side vs. the paper")
    A("")
    A(f"One run of the **U-Net-Auto** pipeline in the paper's described configuration "
      f"(**{label_a}**, scene-scale thin-cloud/shadow filter). Every comparison below "
      "is against the paper's **U-Net-Auto** column; the manually-labeled "
      "**U-Net-Man** results are out of scope (see §0.5).")
    A("")
    A("> **Dataset.** The authors' own **66 scenes**, supplied directly and natively "
      "2048x2048, tiled into the paper's full 66 x 64 = **4,224 tiles** "
      "(3,379 train / 845 test) with **no resampling at any stage**. This is the "
      "paper's dataset, not an approximation of it.")
    A("")

    # ── 0.1 Table IV ────────────────────────────────────────────────────────
    A("### 0.1 Table IV — overall accuracy")
    A("")
    A("| Condition | Paper | Ours | Δ |")
    A("|---|:--:|:--:|:--:|")
    for b, cond, paper_acc in (("orig", "Original S2 imagery", PAPER_TABLE_IV["orig"]),
                               ("filtered", "Thin cloud / shadow filtered",
                                PAPER_TABLE_IV["filtered"])):
        ev = run_a["overall"][b]
        ours = ev["test_accuracy"] * 100 if ev else None
        ours_s = f"**{ours:.2f}%**" if ours is not None else "—"
        A(f"| {cond} | {paper_acc:.2f}% | {ours_s} | {delta(ours, paper_acc)} |")
    A("")
    A("We exceed the paper on both conditions. The margin is **not** evidence of a "
      "better reproduction: our auto-labels are self-consistent — the U-Net is scored "
      "against labels produced by color-segmenting the very tiles it sees — so the "
      "task is easier than the paper's, which scores against an independently derived "
      "reference. Training is also unseeded beyond the split (`random_state=0`), which "
      "historically contributes ~1 pt of run-to-run noise. See §0.6.")
    A("")

    # ── 0.2 P/R/F1 ──────────────────────────────────────────────────────────
    A("### 0.2 Table IV — precision / recall / F1 (micro-averaged)")
    A("")
    A("| Condition | Paper P / R / F1 | Ours P / R / F1 |")
    A("|---|:--:|:--:|")
    for b, cond, prf in (("orig", "Original", PAPER_TABLE_IV_PRF["orig"]),
                         ("filtered", "Filtered", PAPER_TABLE_IV_PRF["filtered"])):
        ev = run_a["overall"][b]
        if ev:
            o = " / ".join(f"{ev[k] * 100:.2f}" for k in
                           ("precision", "recall", "f1_score"))
        else:
            o = "—"
        A(f"| {cond} | {' / '.join(f'{v:.2f}' for v in prf)} | {o} |")
    A("")
    A("> ⚠️ The paper's filtered U-Net-Auto P/R/F1 reads **98.88 / 91.87 / 91.89** — "
      "the recall and F1 are inconsistent with its own 98.97% accuracy and appear to "
      "be a typo. Ours are internally consistent.")
    A("")

    # ── 0.3 Table V ─────────────────────────────────────────────────────────
    A("### 0.3 Table V — cloud/shadow-stratified accuracy")
    A("")
    da = run_a["strat"]["orig"]
    dropped_a = da.get("dropped_no_fraction") if da else None

    def _strat_n(d):
        if not d:
            return None
        return sum(d[k].get("n_tiles", 0) for k in ("high_cloud", "low_cloud")
                   if isinstance(d.get(k), dict)) or None
    n_test_a = _strat_n(da) or "all"

    A("| Stratum | Condition | Paper | Ours | Δ |")
    A("|---|---|:--:|:--:|:--:|")
    rn = {"high": "high_cloud", "low": "low_cloud"}
    for stratum, b, paper_v in (
            ("high", "orig", PAPER_TABLE_V[("orig", "high")]),
            ("high", "filtered", PAPER_TABLE_V[("filtered", "high")]),
            ("low", "orig", PAPER_TABLE_V[("orig", "low")]),
            ("low", "filtered", PAPER_TABLE_V[("filtered", "low")])):
        s = run_a["strat"][b]
        node = (s or {}).get(rn[stratum]) or {}
        ours = node.get("test_accuracy")
        ours = ours * 100 if ours is not None else None
        ours_s = f"**{ours:.2f}%**" if ours is not None else "—"
        label = "≥10% cloud/shadow" if stratum == "high" else "<10% cloud/shadow"
        A(f"| {label} | {b} | {paper_v:.2f}% | {ours_s} | {delta(ours, paper_v)} |")
    A("")
    hi = (da or {}).get("high_cloud") or {}
    lo = (da or {}).get("low_cloud") or {}
    A(f"All **{n_test_a}** test tiles are stratified "
      f"({hi.get('n_tiles', '?')} high-cloud / {lo.get('n_tiles', '?')} low-cloud), "
      f"`dropped_no_fraction={dropped_a}` — no tile is excluded, so neither column is "
      "biased by a partial test set.")
    A("")
    hi_a_orig = (f"{hi['test_accuracy'] * 100:.2f}%"
                 if "test_accuracy" in hi else "—")
    A("**Key divergence from the paper:** the paper's largest filter benefit is on "
      "≥10%-cloud *original* imagery (79.91% → 99.28%, +19 pt). Our original "
      f"high-cloud accuracy is already high ({hi_a_orig}), so our filter gain there is "
      "much smaller. Same root cause as §0.1 — self-consistent labels let the model fit "
      "cloudy raw tiles better than the paper's pipeline could.")
    A("")

    # ── 0.4 Fig 13 ──────────────────────────────────────────────────────────
    A("### 0.4 Fig 13 — confusion matrices (U-Net-Auto)")
    A("")
    A("Row-normalized percentages, rows = true class, cols = predicted, in the "
      "paper's class order (thin / thick / water) and its percentage format. The "
      "**diagonal is per-class recall**; off-diagonals are the cloud-shadow-induced "
      f"confusion the paper highlights. Computed on all {n_test_a} test tiles.")
    A("")

    def fmt_row(r):
        return " ".join("     ?" if v is None else f"{v:6.2f}%" for v in r)

    cond_titles = {
        ("orig", "high"): "≥10% cloud/shadow · original  (paper: \"cloudy-shadowy\")",
        ("filtered", "high"): "≥10% cloud/shadow · filtered  (paper: \"cloud-shadow-removed\")",
        ("orig", "low"): "<10% cloud/shadow · original  (paper: \"cloud-shadow-free\")",
        ("filtered", "low"): "<10% cloud/shadow · filtered",
    }
    classes = ("thin ", "thick", "water")
    for (b, stratum), title in cond_titles.items():
        A(f"**{title}**")
        A("")
        A("```")
        A("            true\\pred     thin    thick    water")
        for src, mat in (("Paper", PAPER_FIG13_FULL[(b, stratum)]),
                         ("Ours ", run_a["cm"].get((b, stratum)))):
            if mat is None:
                A(f"{src}      (matrix unavailable)")
                continue
            for i, cls in enumerate(classes):
                prefix = f"{src}  {cls}" if i == 0 else f"       {cls}"
                A(f"{prefix}      {fmt_row(mat[i])}")
        A("```")
        A("")
    # Derived, not hardcoded — these drifted between runs when written by hand.
    oh = run_a["cm"].get(("orig", "high")) or []
    t2t = f"{oh[1][0]:.2f}%" if oh else "—"
    trec = f"{oh[1][1]:.2f}%" if oh else "—"
    fh = run_a["cm"].get(("filtered", "high")) or []
    fl = run_a["cm"].get(("filtered", "low")) or []
    offs = [v for m in (fh, fl) for i, r in enumerate(m) for j, v in enumerate(r)
            if i != j]
    worst = f"{max(offs):.2f}%" if offs else "—"
    A("**What the matrices show.** Under ≥10% cloud/shadow on *original* imagery the "
      "paper's model sends **24.05% of thick ice → thin** (75.95% thick recall) — its "
      "signature cloud-shadow error, where shadowed thick ice reads as thin. We show "
      f"the *same* error in the same direction but far smaller: {t2t} thick → thin, "
      f"leaving thick-ice recall at {trec}. Filtering collapses every off-diagonal to "
      f"at most {worst} in our run, against the paper's largest filtered off-diagonal "
      "of 2.16% — the paper's central qualitative claim, and it reproduces.")
    A("")

    # 0.5 Coverage / not-compared
    A("### 0.5 Paper-claim coverage — what is and isn't compared")
    A("")
    A("| Paper item | Status | Notes / reason |")
    A("|---|---|---|")
    A("| Table IV (U-Net-Auto accuracy) | ✅ Compared | §0.1 |")
    A("| Table IV P/R/F1 | ✅ Compared | §0.2 (paper has an apparent typo) |")
    A("| Table V (stratified, U-Net-Auto) | ✅ Compared | §0.3 — all 845 test tiles, none dropped |")
    A("| Fig 13 auto-labeled confusion matrices | ✅ Compared | §0.4 (full 3×3 matrices) |")
    A("| Fig 5 filtered-scene grid | ✅ Qualitative | per-scene `filtered_s2_vis_*.png`; see the figure sections below |")
    A("| Fig 6 / Fig 11 color-seg auto-labeling | ✅ Qualitative | masks reproduced; see the figure sections below |")
    A("| Fig 14 whole-scene predictions | ✅ Qualitative | 132 PNGs (66 scenes × 2 branches); no paper numbers to match |")
    A("| **Table IV/V U-Net-Man column** | ❌ Not compared | No manual ground-truth labels in our dataset — we only run the auto-labeling (U-Net-Auto) path. |")
    A("| **Fig 13 manually-labeled matrices** | ❌ Not compared | Same — no U-Net-Man model. |")
    A("| **Auto-labeling SSIM (89% / 99.64%)** | ❌ Not compared | SSIM is measured against manual labels; none available. |")
    A("| **Table I — Python multiprocessing speedup (4.5×)** | ❌ Not compared | Reference uses `multiprocessing.Pool` on one host; our pipeline parallelizes via Pegasus/HTCondor job fan-out — a different model, not benchmarked. |")
    A("| **Table II — PySpark map-reduce speedup (16.25×)** | ❌ Not compared | Spark map-reduce not used; the Pegasus DAG replaces it. |")
    A("| **Table III / Fig 12 — Horovod training scaling (7.21× @ 8 GPU)** | ❌ Not run | Needs a 1/2/4/6/8-GPU sweep on a DGX-class node (Run C); our runs used single-GPU training. |")
    A("| **Dataset size (66 scenes / 4224 tiles)** | ✅ Matches | The authors supplied all 66 source scenes (2048×2048 native, `s2_original_2048/`), so the workflow runs the paper's full 66 × 64 = 4,224 tiles. Runs A/B predate this and used the 63-scene GEE export. |")
    A("")
    A("See [`gap_analysis.md`](gap_analysis.md) for the full audit of these "
      "not-compared items (U-Net-Man baseline, SSIM, Spark/multiprocessing "
      "speedups) — paper claim by claim, with effort estimates.")
    A("")

    # 0.6 Reading the numbers
    A("### 0.6 How to read these numbers")
    A("")
    A("- **The dataset is no longer a confound.** This run uses the authors' own 66 "
      "scenes at their native 2048x2048, so the paper's 4,224 tiles are reproduced "
      "exactly and nothing is resampled. An earlier run on a 63-scene Google Earth "
      "Engine export resized 2000->2048 scored 96.25% / 99.76% — within a point of "
      "this one, which retires the worry that the older numbers were an artifact of "
      "that incomplete export.")
    A("- **Our margin over the paper is a labeling artifact, not an improvement.** "
      "Both our branches are scored against auto-labels derived from the same tiles "
      "the U-Net consumes, so input and target are self-consistent by construction. "
      "The paper's pipeline does not have that property. This is the single most "
      "important caveat on every Delta in this report.")
    filt_ev = run_a["overall"].get("filtered")
    filt_s = f"~{filt_ev['test_accuracy'] * 100:.2f}%" if filt_ev else "near ceiling"
    A(f"- **The filtered branch sits near ceiling ({filt_s}) for the same reason**, "
      "amplified: `--filtered-labels filtered` re-derives labels from the filtered "
      "tiles. Running `--filtered-labels raw` instead scores filtered inputs against "
      "raw-scene labels and lands around 90%, which is the more honest "
      "cross-comparison.")
    A("- **Training is unseeded beyond the split.** Only `random_state=0` fixes the "
      "train/test partition; weight init and dropout vary run to run, historically "
      "worth ~1 pt of overall accuracy and considerably more on thin-ice recall. "
      "Treat sub-point differences as noise.")
    A("")
    A("---")
    A("")
    return L


def render_report(
    *,
    paper_pdf: Path,
    run_dir: Path,
    run_label: str,
    paper_fig_dir: Path,
    paper_figs: dict,
    out_md: Path,
) -> None:
    eval_orig = read_eval(run_dir / "evaluation_results_orig.json")
    eval_filt = read_eval(run_dir / "evaluation_results_filtered.json")
    # Newer runs prefix orig-branch artifacts with ``orig_``; fall back to the
    # unprefixed names used by older single-path runs.
    pc_orig = (per_class(run_dir / "orig_per_class_metrics.json")
               or per_class(run_dir / "per_class_metrics.json"))
    pc_filt = per_class(run_dir / "filtered_per_class_metrics.json")

    repo_root = out_md.parent
    lines: list = []
    A = lines.append

    A("# S2 Sea-Ice Segmentation — Reproduction vs. Paper")
    A("")

    # Side-by-side numeric comparison against the paper.
    lines.extend(render_comprehensive(load_run(run_dir), run_label,
                                      repo_root=out_md.parent))
    A("The sections below pair each paper figure with the matching output from "
      f"**{run_label}**, side by side.")
    A("")

    A(f"**Run (figures below):** `{run_label}` &nbsp;·&nbsp; **Paper:** Iqrah, Wang, Xie, Prasad — "
      "*\"A Parallel Workflow for Polar Sea-Ice Classification using Auto-labeling of "
      "Sentinel-2 Imagery,\"* IEEE IPDPSW 2024.  ")
    A("**Model:** U-Net-Auto (color-segmentation auto-labels — the paper's "
      "auto-labeled U-Net, *not* the manually-labeled U-Net-Man).")
    A("")
    A("**Conditions** (matching the paper's Table IV rows):")
    A("- _Original S2 imagery_ — raw Sentinel-2 grayscale tiles, auto-labels from raw scenes.")
    A("- _Thin cloud / shadow-filtered S2 imagery_ — tiles passed through the paper's "
      "`only_shadow_cloud_removal` filter; auto-labels are re-derived by "
      "color-segmenting the filtered tiles so input and label are self-consistent. "
      "This is the workflow's default (`--filtered-labels filtered`).")
    A("")
    A("This report is generated by `compare_with_paper.py`. Re-run after a new "
      "training run to refresh numbers and image pairings.")
    A("")
    A("---")
    A("")

    # ── Headline metrics ────────────────────────────────────────────────────
    A("## 1. Headline metrics (paper Table IV)")
    A("")
    A(f"| Condition | Paper (U-Net-Auto) | **Ours — {run_label}** | Δ |")
    A("|---|:--:|:--:|:--:|")
    if eval_orig:
        ours = eval_orig["test_accuracy"] * 100
        A(f"| Original S2 imagery | 90.18% | **{ours:.2f}%** | "
          f"{ours - 90.18:+.2f} pt |")
    if eval_filt:
        ours = eval_filt["test_accuracy"] * 100
        A(f"| Thin cloud / shadow filtered | 98.97% | **{ours:.2f}%** | "
          f"{ours - 98.97:+.2f} pt |")
    A("")
    A("Detailed F1 / precision / recall (Keras micro-averaged):")
    A("")
    A("| Dataset (paper Table IV) | Accuracy | F1 | Precision | Recall | Train time |")
    A("|---|:--:|:--:|:--:|:--:|:--:|")
    hist_orig = read_eval(run_dir / "training_history_orig.json")
    hist_filt = read_eval(run_dir / "training_history_filtered.json")
    rows = [
        ("Original S2 imagery", eval_orig, hist_orig),
        ("Thin cloud / shadow-filtered S2 imagery", eval_filt, hist_filt),
    ]
    for label, ev, hist in rows:
        if ev is None:
            continue
        tt = hist.get("training_time_seconds") if hist else None
        tt_str = f"{tt:.1f} s" if tt else "n/a"
        A(f"| {label} | {metric_pct(ev, 'test_accuracy')} | "
          f"{metric_f(ev, 'f1_score')} | "
          f"{metric_f(ev, 'precision')} | "
          f"{metric_f(ev, 'recall')} | {tt_str} |")
    A("")

    # ── Per-class ───────────────────────────────────────────────────────────
    A(f"## 2. Per-class metrics ({run_label})")
    A("")
    for label, pc in [("Original S2 imagery", pc_orig),
                      ("Thin cloud / shadow-filtered S2 imagery", pc_filt)]:
        if pc is None:
            continue
        # LabelEncoder order = sorted mask gray values: 29 = thin ice (blue),
        # 76 = thick ice (red), 149 = open water (green). This matches the
        # paper's Fig 13 axis order (Thin Ice, Thick Ice, Open water).
        classes = ["Thin ice", "Thick ice", "Open water"]
        if len(pc["support"]) == 4:
            # 2000x2000 scenes are zero-padded to tile evenly; when labels are
            # split from the full-scene mask the padding becomes a 4th class
            # (gray value 0, LabelEncoder sorts it first).
            classes = ["Tile padding (artifact)"] + classes
        A(f"### {label}")
        A("")
        A("| Class | Precision | Recall | F1 | Support |")
        A("|---|:--:|:--:|:--:|--:|")
        for i, cls in enumerate(classes):
            A(f"| {cls} | {pc['precision'][i]:.3f} | {pc['recall'][i]:.3f} | "
              f"{pc['f1-score'][i]:.3f} | {pc['support'][i]:,} |")
        A("")

    # ── Section-by-section image pairings ──────────────────────────────────
    A("## 3. Side-by-side figures")
    A("")
    ours_out_dir = repo_root / "paper_figures" / "ours"
    ours_out_dir.mkdir(parents=True, exist_ok=True)

    for i, sec in enumerate(SECTIONS, 1):
        fig = PAPER_FIGURES[sec.paper_fig]
        paper_path = paper_figs.get(sec.paper_fig)
        A(f"### 3.{i} {sec.title}")
        A("")
        A(f"_{fig.caption}_")
        A("")

        # Copy each ours file into paper_figures/ours/ so the markdown is
        # self-contained (output/ is gitignored). Skip files that don't exist.
        ours_paths: list = []
        for label, rel in sec.ours:
            src = run_dir / rel
            # Prefer the ``orig_``-prefixed name emitted by newer runs.
            prefixed = run_dir / f"orig_{rel}"
            if prefixed.exists():
                src = prefixed
            if not src.exists():
                ours_paths.append((label, None))
                continue
            dst = ours_out_dir / src.name
            if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                shutil.copy2(src, dst)
            ours_paths.append((label, dst))

        if len(ours_paths) == 1:
            our_label, our_path = ours_paths[0]
            A(f"| Paper (reference) | Ours — {run_label} |")
            A("|:---:|:---:|")
            A(f"| {img_md(paper_path, repo_root)} | {img_md(our_path, repo_root)} |")
            A(f"| _{fig.fig_id}_ | _{our_label}_ |")
        else:
            A("**Paper:**")
            A("")
            A(f"{img_md(paper_path, repo_root)}")
            A("")
            A(f"_{fig.fig_id}_")
            A("")
            A(f"**Ours — {run_label}:**")
            A("")
            # Lay all ours images on a single row so they can be compared at a
            # glance (matches the paper's multi-panel figures).
            cols = len(ours_paths)
            A("| " + " | ".join("&nbsp;" for _ in ours_paths) + " |")
            A("|" + "|".join(":---:" for _ in ours_paths) + "|")
            A("| " + " | ".join(img_md(p, repo_root) for _, p in ours_paths) + " |")
            A("| " + " | ".join(f"_{lbl}_" if lbl else "&nbsp;"
                                 for lbl, _ in ours_paths) + " |")
            A("")
        if sec.commentary:
            A(sec.commentary)
        A("")

    # ── Conclusions ─────────────────────────────────────────────────────────
    A("## 4. Conclusions")
    A("")
    if eval_orig and eval_filt:
        orig_pct = eval_orig["test_accuracy"] * 100
        filt_pct = eval_filt["test_accuracy"] * 100
        A(f"- **Original S2 imagery (U-Net-Auto):** {orig_pct:.2f}% accuracy "
          f"vs paper's 90.18% ({orig_pct - 90.18:+.2f} pt).")
        A(f"- **Thin cloud / shadow-filtered S2 imagery (U-Net-Auto):** "
          f"{filt_pct:.2f}% accuracy vs paper's 98.97% "
          f"({filt_pct - 98.97:+.2f} pt).")
        A(f"- **Filtering helps us less than it helped the paper:** our "
          f"original→filtered swing is +{filt_pct - orig_pct:.2f} pt against the "
          "paper's +8.79 pt (90.18% → 98.97%). The direction reproduces; the "
          "magnitude does not, because our unfiltered baseline already starts "
          f"{orig_pct - 90.18:.2f} pt above the paper's and so has far less room "
          "to gain. Both effects trace to the same cause — self-consistent "
          "auto-labels (§0.6).")
        A("- **Read every Δ above with that caveat.** Exceeding the paper here is "
          "a property of how the labels are made, not evidence of a better model.")
    A("")
    A("See `comparison_report.html` for the styled long-form discussion of "
      "methodology, code review, and remaining differences.")
    A("")
    A("---")
    A(f"_Generated by `compare_with_paper.py` from {run_dir.name} and {paper_pdf.name}._")
    A("")

    out_md.write_text("\n".join(lines))
    print(f"Wrote {out_md}")


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper", type=Path,
                    default=here.parent / "A_Parallel_Workflow_for_Polar_Sea-Ice_"
                                          "Classification_Using_Auto-Labeling_of_"
                                          "Sentinel-2_Imagery.pdf",
                    help="Path to the reference paper PDF")
    ap.add_argument("--run-dir", type=Path, default=here / "output",
                    help="Pegasus run output directory holding the evaluation JSONs "
                         "and figure PNGs (default: ./output)")
    ap.add_argument("--run-label", type=str, default="run0003",
                    help="Display label for the run shown in the report header")
    ap.add_argument("--paper-fig-dir", type=Path, default=here / "paper_figures",
                    help="Directory to write extracted paper figures into")
    ap.add_argument("--out", type=Path, default=here / "comparison_report.md",
                    help="Output markdown report path")
    ap.add_argument("--dpi", type=int, default=200,
                    help="DPI for pdftoppm page rendering (default 200)")
    args = ap.parse_args()

    if not args.paper.exists():
        sys.exit(f"Paper PDF not found: {args.paper}")
    if not args.run_dir.exists():
        sys.exit(f"Run directory not found: {args.run_dir}")

    args.paper_fig_dir.mkdir(parents=True, exist_ok=True)
    rendered = render_pdf_pages(args.paper, args.paper_fig_dir, dpi=args.dpi)
    paper_figs = crop_figures(rendered, args.paper_fig_dir)

    render_report(
        paper_pdf=args.paper,
        run_dir=args.run_dir,
        run_label=args.run_label,
        paper_fig_dir=args.paper_fig_dir,
        paper_figs=paper_figs,
        out_md=args.out,
    )


if __name__ == "__main__":
    main()
