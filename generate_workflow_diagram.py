#!/usr/bin/env python3

"""Generate a simplified workflow diagram for the README.

Produces a compact, publication-ready DAG image that represents the
workflow structure without expanding every parallel job. Reflects the
current defaults (``--scene-size 2048 --paths both --filtered-labels
filtered --stratified-eval --infer``): a Stage 0 scene resize, two U-Net
branches (raw scene vs thin-cloud/shadow-filtered scene) each with its
own auto-label tile chain, per-scene cloud/shadow fractions feeding a
stratified evaluation (Table V / Fig 13), and whole-scene inference
(Fig 9 / 14).

Cross-checked against the planned DAG of run0004
(``pegasus-graphviz workflow.yml``), whose node set is: resize_image*,
image_split, compute_cloud_fraction, color_segment, image_merge,
filter_image, split_images, split_masks, preprocess_data, train_unet,
evaluate_model, evaluate_stratified, generate_plots, infer_unet.
(*run0004 itself passed ``--scene-size 0``, so it has no resize jobs.)
"""

import argparse
import subprocess
import sys


def make_dot(n_images=2):
    """Build a Graphviz DOT string for the simplified workflow."""

    colors = {
        'resize': '#595959',
        'split': '#4472C4',
        'segment': '#ED7D31',
        'merge': '#70AD47',
        'autolabel': '#BF8F00',
        'filter': '#2E75B6',
        'cloudfrac': '#12869A',
        'stage2_orig': '#7030A0',
        'stage2_filt': '#A5468C',
        'plots': '#C00000',
        'output': '#F2F2F2',
    }
    fc = 'white'

    lines = []
    def L(s=''):
        lines.append(s)

    L('digraph S2_Segmentation {')
    L('  rankdir=TB;')
    L('  dpi=200;')
    L('  bgcolor=white;')
    L('  pad=0.4;')
    L('  nodesep=0.5;')
    L('  ranksep=0.75;')
    L('  compound=true;')
    L('  newrank=true;')
    L()
    L('  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12];')
    L('  edge [color="#555555", arrowsize=0.7];')
    L()

    # ── Title ──
    L('  labelloc=t;')
    L('  label="S2 Segmentation Workflow — defaults (--paths both, --filtered-labels filtered, --stratified-eval, --infer)\\n'
      'one Image column per scene (66 in the reference dataset) — Stage 0 resize is skipped with --scene-size 0";')
    L('  fontname="Helvetica-Bold"; fontsize=17; fontcolor="#333333";')
    L()

    # ── Stage 1 + auto-label: per-image columns ──
    for i in range(n_images):
        tag = f"Image {i}"
        L(f'  subgraph cluster_img{i} {{')
        L(f'    label="{tag}"; labeljust=c; fontname="Helvetica-Bold"; fontsize=13; fontcolor="#444444";')
        L(f'    style="dashed,rounded"; color="#AAAAAA";')
        L()

        # Stage 0 — scene normalization (skipped with --scene-size 0)
        L(f'    resize_{i}     [label="resize_image\\n(scene → 2048²)", fillcolor="{colors["resize"]}", fontcolor={fc}, style="rounded,filled,dashed"];')

        # Raw / orig chain
        L(f'    split_{i}      [label="image_split",                  fillcolor="{colors["split"]}",     fontcolor={fc}];')
        L(f'    seg_{i}        [label="color_segment\\n(×64 parallel)", fillcolor="{colors["segment"]}",  fontcolor={fc}];')
        L(f'    merge_{i}      [label="image_merge",                  fillcolor="{colors["merge"]}",     fontcolor={fc}];')
        L(f'    seg_out_{i}    [label="*_seg.png",                    shape=note, style=filled, fillcolor="{colors["output"]}", fontcolor="#333333", fontsize=10];')
        L(f'    split_img_{i}  [label="split_images\\n(raw 256² tiles)",  fillcolor="{colors["autolabel"]}", fontcolor={fc}];')
        L(f'    split_mask_{i} [label="split_masks\\n(raw 256² labels)",   fillcolor="{colors["autolabel"]}", fontcolor={fc}];')

        # Cloud/shadow fraction — the Table V / Fig 13 stratification key
        L(f'    cf_{i}         [label="compute_cloud_fraction\\n(per-tile cloud/shadow %)", fillcolor="{colors["cloudfrac"]}", fontcolor={fc}];')

        # Filtered branch. Note there is no merge/re-split here: color_segment
        # runs on each already-256² filtered tile, so it emits the filtered
        # branch's label tiles directly (Option A, --filtered-labels filtered).
        L(f'    filt_{i}       [label="filter_image\\n(cloud/shadow rm)", fillcolor="{colors["filter"]}",    fontcolor={fc}];')
        L(f'    split_imgf_{i} [label="split_images\\n(filt 256² tiles)", fillcolor="{colors["autolabel"]}", fontcolor={fc}];')
        L(f'    seg_filt_{i}   [label="color_segment\\n(×64 filtered tiles)\\n→ filt 256² labels", fillcolor="{colors["segment"]}",  fontcolor={fc}];')

        L()
        # stage 0 → everything that consumes the scene
        L(f'    resize_{i} -> split_{i};')
        L(f'    resize_{i} -> split_img_{i};')
        L(f'    resize_{i} -> filt_{i};')
        L(f'    resize_{i} -> cf_{i};')
        # orig edges: tiles are colour-segmented, merged, then re-split as
        # grayscale label tiles on the same grid as split_images.
        L(f'    split_{i} -> seg_{i};')
        L(f'    seg_{i} -> merge_{i};')
        L(f'    merge_{i} -> seg_out_{i} [style=dotted, color="#AAAAAA"];')
        L(f'    merge_{i} -> split_mask_{i};')
        # filtered edges
        L(f'    filt_{i} -> split_imgf_{i};')
        L(f'    split_imgf_{i} -> seg_filt_{i};')
        L(f'  }}')
        L()

    # ── Stage 2: two branches ──
    L('  subgraph cluster_stage2 {')
    L('    label="Stage 2 — dual U-Net training, stratified evaluation & whole-scene inference"; labeljust=c;')
    L('    fontname="Helvetica-Bold"; fontsize=13; fontcolor="#444444";')
    L('    style="dashed,rounded"; color="#AAAAAA";')
    L()

    # Orig branch
    L('    subgraph cluster_orig {')
    L('      label="orig branch (raw image + raw labels)"; labeljust=c; fontsize=11; fontcolor="#555555";')
    L('      style="rounded"; color="#CCCCCC";')
    L(f'      pre_o   [label="preprocess_data",       fillcolor="{colors["stage2_orig"]}", fontcolor={fc}];')
    L(f'      train_o [label="train_unet_orig\\n(GPU)",  fillcolor="{colors["stage2_orig"]}", fontcolor={fc}];')
    L(f'      eval_o  [label="evaluate_orig\\n(GPU)",    fillcolor="{colors["stage2_orig"]}", fontcolor={fc}];')
    L(f'      strat_o [label="evaluate_stratified\\n(≥10% / <10% cloud)", fillcolor="{colors["stage2_orig"]}", fontcolor={fc}];')
    L(f'      infer_o [label="infer_unet\\n(× N whole scenes)\\nre-reads the Stage 0 scenes", shape=box3d, style=filled, fillcolor="{colors["stage2_orig"]}", fontcolor={fc}];')
    L(f'      plots_o [label="generate_plots\\n(orig)",  fillcolor="{colors["plots"]}",       fontcolor={fc}];')
    L('      pre_o -> train_o -> eval_o -> plots_o;')
    L('      train_o -> strat_o;')
    L('      train_o -> infer_o;')
    L('    }')
    L()

    # Filtered branch
    L('    subgraph cluster_filt {')
    L('      label="filtered branch (filtered image + filtered labels — Option A)"; labeljust=c; fontsize=11; fontcolor="#555555";')
    L('      style="rounded"; color="#CCCCCC";')
    L(f'      pre_f   [label="preprocess_data",            fillcolor="{colors["stage2_filt"]}", fontcolor={fc}];')
    L(f'      train_f [label="train_unet_filtered\\n(GPU)",  fillcolor="{colors["stage2_filt"]}", fontcolor={fc}];')
    L(f'      eval_f  [label="evaluate_filtered\\n(GPU)",    fillcolor="{colors["stage2_filt"]}", fontcolor={fc}];')
    L(f'      strat_f [label="evaluate_stratified\\n(≥10% / <10% cloud)", fillcolor="{colors["stage2_filt"]}", fontcolor={fc}];')
    L(f'      infer_f [label="infer_unet --filter\\n(× N whole scenes)\\nre-reads the Stage 0 scenes", shape=box3d, style=filled, fillcolor="{colors["stage2_filt"]}", fontcolor={fc}];')
    L(f'      plots_f [label="generate_plots\\n(filtered)",  fillcolor="{colors["plots"]}",       fontcolor={fc}];')
    L('      pre_f -> train_f -> eval_f -> plots_f;')
    L('      train_f -> strat_f;')
    L('      train_f -> infer_f;')
    L('    }')
    L('  }')
    L()

    # ── Edges: auto-label → preprocess (per branch) ──
    for i in range(n_images):
        L(f'  split_img_{i}   -> pre_o;')
        L(f'  split_mask_{i}  -> pre_o;')
        L(f'  split_imgf_{i}  -> pre_f;')
        L(f'  seg_filt_{i}    -> pre_f;')
        # cloud fractions ride along with preprocess and come out aligned
        # with X_test, which is what evaluate_stratified splits on.
        L(f'  cf_{i} -> pre_o [style=dashed, color="#12869A"];')
        L(f'  cf_{i} -> pre_f [style=dashed, color="#12869A"];')
    L()

    # ── Stage 2 outputs (one note per branch, suffixed) ──
    L('  node [shape=note, style=filled, fillcolor="#F2F2F2", fontcolor="#333333", fontsize=10];')
    L('  out_o [label="model_orig.hdf5 · evaluation_results_orig.json\\n'
      '{training_curves,confusion_matrix,prediction_samples,metrics_table}.png\\n'
      'orig_{high,low}_cloud_* — Table V / Fig 13\\n'
      'orig_infer_&lt;scene&gt;.png × N — Fig 9 / 14"];')
    L('  out_f [label="model_filtered.hdf5 · evaluation_results_filtered.json\\n'
      'filtered_{training_curves,confusion_matrix,prediction_samples,metrics_table}.png\\n'
      'filtered_{high,low}_cloud_* — Table V / Fig 13\\n'
      'filtered_infer_&lt;scene&gt;.png × N — Fig 9 / 14"];')
    L()
    L('  eval_o  -> out_o [style=dotted, color="#AAAAAA"];')
    L('  strat_o -> out_o [style=dotted, color="#AAAAAA"];')
    L('  infer_o -> out_o [style=dotted, color="#AAAAAA"];')
    L('  plots_o -> out_o [style=dotted, color="#AAAAAA"];')
    L('  eval_f  -> out_f [style=dotted, color="#AAAAAA"];')
    L('  strat_f -> out_f [style=dotted, color="#AAAAAA"];')
    L('  infer_f -> out_f [style=dotted, color="#AAAAAA"];')
    L('  plots_f -> out_f [style=dotted, color="#AAAAAA"];')
    L()

    # ── Rank hints ──
    resizes = " ".join(f"resize_{i};" for i in range(n_images))
    splits = " ".join(f"split_{i};" for i in range(n_images))
    filts = " ".join(f"filt_{i};" for i in range(n_images))
    cfs = " ".join(f"cf_{i};" for i in range(n_images))
    segs = " ".join(f"seg_{i};" for i in range(n_images))
    merges = " ".join(f"merge_{i};" for i in range(n_images))
    imgfs = " ".join(f"split_imgf_{i};" for i in range(n_images))
    L(f'  {{ rank=same; {resizes} }}')
    L(f'  {{ rank=same; {splits} {filts} {cfs} }}')
    L(f'  {{ rank=same; {segs} {imgfs} }}')
    L(f'  {{ rank=same; {merges} }}')

    auto_nodes = " ".join(f"split_img_{i}; split_mask_{i}; seg_filt_{i};" for i in range(n_images))
    L(f'  {{ rank=same; {auto_nodes} }}')
    L('  { rank=same; pre_o; pre_f; }')
    L('  { rank=same; train_o; train_f; }')
    L('  { rank=same; eval_o; strat_o; eval_f; strat_f; }')
    L('  { rank=same; plots_o; infer_o; plots_f; infer_f; }')
    L('  { rank=same; out_o; out_f; }')

    L('}')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a simplified workflow diagram",
    )
    parser.add_argument("-n", "--n-images", type=int, default=2,
                        help="Number of representative images to show (default: 2)")
    parser.add_argument("-o", "--output", type=str, default="images/workflow.png",
                        help="Output file (png, pdf, svg, or dot)")
    parser.add_argument("--dot-only", action="store_true",
                        help="Print DOT to stdout instead of rendering")
    args = parser.parse_args()

    dot_str = make_dot(n_images=args.n_images)

    if args.dot_only:
        print(dot_str)
        return

    ext = args.output.rsplit(".", 1)[-1].lower()

    if ext == "dot":
        with open(args.output, "w") as f:
            f.write(dot_str)
        print(f"DOT file written to {args.output}")
        return

    try:
        subprocess.run(
            ["dot", f"-T{ext}", "-o", args.output],
            input=dot_str,
            text=True,
            capture_output=True,
            check=True,
        )
        print(f"Diagram written to {args.output}")
    except FileNotFoundError:
        print("Error: 'dot' command not found. Install graphviz:", file=sys.stderr)
        print("  brew install graphviz    # macOS", file=sys.stderr)
        print("  apt install graphviz     # Debian/Ubuntu", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running dot: {e.stderr}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
