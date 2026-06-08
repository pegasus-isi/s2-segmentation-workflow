#!/usr/bin/env python3

"""Generate a simplified workflow diagram for the README.

Produces a compact, publication-ready DAG image that represents the
workflow structure without expanding every parallel job. Reflects the
current ``--paths both --filtered-labels filtered`` default: two U-Net
branches (raw scene vs thin-cloud/shadow-filtered scene), each with its
own auto-label tile chain.
"""

import argparse
import subprocess
import sys


def make_dot(n_images=2):
    """Build a Graphviz DOT string for the simplified workflow."""

    colors = {
        'split': '#4472C4',
        'segment': '#ED7D31',
        'merge': '#70AD47',
        'autolabel': '#BF8F00',
        'filter': '#2E75B6',
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
    L('  label="S2 Segmentation Workflow (auto-label, --paths both, --filtered-labels filtered)";')
    L('  fontname="Helvetica-Bold"; fontsize=17; fontcolor="#333333";')
    L()

    # ── Stage 1 + auto-label: per-image columns ──
    for i in range(n_images):
        tag = f"Image {i}"
        L(f'  subgraph cluster_img{i} {{')
        L(f'    label="{tag}"; labeljust=c; fontname="Helvetica-Bold"; fontsize=13; fontcolor="#444444";')
        L(f'    style="dashed,rounded"; color="#AAAAAA";')
        L()

        # Raw / orig chain
        L(f'    split_{i}      [label="image_split",                  fillcolor="{colors["split"]}",     fontcolor={fc}];')
        L(f'    seg_{i}        [label="color_segment\\n(×64 parallel)", fillcolor="{colors["segment"]}",  fontcolor={fc}];')
        L(f'    merge_{i}      [label="image_merge",                  fillcolor="{colors["merge"]}",     fontcolor={fc}];')
        L(f'    seg_out_{i}    [label="*_seg.png",                    shape=note, style=filled, fillcolor="{colors["output"]}", fontcolor="#333333", fontsize=10];')
        L(f'    split_img_{i}  [label="split_images\\n(raw 256² tiles)",  fillcolor="{colors["autolabel"]}", fontcolor={fc}];')
        L(f'    split_mask_{i} [label="split_masks\\n(raw 256² labels)",   fillcolor="{colors["autolabel"]}", fontcolor={fc}];')

        # Filtered branch
        L(f'    filt_{i}       [label="filter_image\\n(cloud/shadow rm)", fillcolor="{colors["filter"]}",    fontcolor={fc}];')
        L(f'    split_imgf_{i} [label="split_images\\n(filt 256² tiles)", fillcolor="{colors["autolabel"]}", fontcolor={fc}];')
        L(f'    seg_filt_{i}   [label="color_segment\\n(filtered tiles)",  fillcolor="{colors["segment"]}",  fontcolor={fc}];')
        L(f'    merge_filt_{i} [label="image_merge\\n(filtered)",          fillcolor="{colors["merge"]}",    fontcolor={fc}];')
        L(f'    split_maskf_{i}[label="split_masks\\n(filt 256² labels)",  fillcolor="{colors["autolabel"]}", fontcolor={fc}];')

        L()
        # orig edges
        L(f'    split_{i} -> seg_{i};')
        L(f'    seg_{i} -> merge_{i};')
        L(f'    merge_{i} -> seg_out_{i} [style=dotted, color="#AAAAAA"];')
        L(f'    merge_{i} -> split_mask_{i};')
        L(f'    split_{i} -> split_img_{i} [style=dashed, color="#999999"];')
        # filtered edges
        L(f'    filt_{i} -> split_imgf_{i};')
        L(f'    filt_{i} -> seg_filt_{i};')
        L(f'    seg_filt_{i} -> merge_filt_{i};')
        L(f'    merge_filt_{i} -> split_maskf_{i};')
        L(f'  }}')
        L()

    # ── Ellipsis ──
    if n_images == 2:
        L('  ellipsis [label="  ...  \\n(× N images)", shape=plaintext, fontsize=12, fontname="Helvetica", fontcolor="#888888"];')
        L(f'  {{ rank=same; merge_0; ellipsis; merge_1; }}')
        L()

    # ── Stage 2: two branches ──
    L('  subgraph cluster_stage2 {')
    L('    label="Stage 2 — dual U-Net training & evaluation (--paths both)"; labeljust=c;')
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
    L(f'      plots_o [label="generate_plots\\n(orig)",  fillcolor="{colors["plots"]}",       fontcolor={fc}];')
    L('      pre_o -> train_o -> eval_o -> plots_o;')
    L('    }')
    L()

    # Filtered branch
    L('    subgraph cluster_filt {')
    L('      label="filtered branch (filtered image + filtered labels — Option A)"; labeljust=c; fontsize=11; fontcolor="#555555";')
    L('      style="rounded"; color="#CCCCCC";')
    L(f'      pre_f   [label="preprocess_data",            fillcolor="{colors["stage2_filt"]}", fontcolor={fc}];')
    L(f'      train_f [label="train_unet_filtered\\n(GPU)",  fillcolor="{colors["stage2_filt"]}", fontcolor={fc}];')
    L(f'      eval_f  [label="evaluate_filtered\\n(GPU)",    fillcolor="{colors["stage2_filt"]}", fontcolor={fc}];')
    L(f'      plots_f [label="generate_plots\\n(filtered)",  fillcolor="{colors["plots"]}",       fontcolor={fc}];')
    L('      pre_f -> train_f -> eval_f -> plots_f;')
    L('    }')
    L('  }')
    L()

    # ── Edges: auto-label → preprocess (per branch) ──
    for i in range(n_images):
        L(f'  split_img_{i}   -> pre_o;')
        L(f'  split_mask_{i}  -> pre_o;')
        L(f'  split_imgf_{i}  -> pre_f;')
        L(f'  split_maskf_{i} -> pre_f;')
    L()

    # ── Stage 2 outputs (suffixed) ──
    L('  node [shape=note, style=filled, fillcolor="#F2F2F2", fontcolor="#333333", fontsize=10];')
    L('  out_model_o [label="model_orig.hdf5"];')
    L('  out_eval_o  [label="evaluation_results_orig.json"];')
    L('  out_plot_o  [label="{training_curves,confusion_matrix,\\nprediction_samples,metrics_table}.png"];')
    L('  out_model_f [label="model_filtered.hdf5"];')
    L('  out_eval_f  [label="evaluation_results_filtered.json"];')
    L('  out_plot_f  [label="filtered_{training_curves,confusion_matrix,\\nprediction_samples,metrics_table}.png"];')
    L()
    L('  train_o -> out_model_o [style=dotted, color="#AAAAAA"];')
    L('  eval_o  -> out_eval_o  [style=dotted, color="#AAAAAA"];')
    L('  plots_o -> out_plot_o  [style=dotted, color="#AAAAAA"];')
    L('  train_f -> out_model_f [style=dotted, color="#AAAAAA"];')
    L('  eval_f  -> out_eval_f  [style=dotted, color="#AAAAAA"];')
    L('  plots_f -> out_plot_f  [style=dotted, color="#AAAAAA"];')
    L()

    # ── Rank hints ──
    splits = " ".join(f"split_{i};" for i in range(n_images))
    filts = " ".join(f"filt_{i};" for i in range(n_images))
    segs = " ".join(f"seg_{i};" for i in range(n_images))
    merges = " ".join(f"merge_{i};" for i in range(n_images))
    L(f'  {{ rank=same; {splits} {filts} }}')
    L(f'  {{ rank=same; {segs} }}')
    L(f'  {{ rank=same; {merges} }}')

    auto_nodes = " ".join(f"split_img_{i}; split_mask_{i}; split_imgf_{i}; split_maskf_{i};" for i in range(n_images))
    L(f'  {{ rank=same; {auto_nodes} }}')
    L('  { rank=same; pre_o; pre_f; }')
    L('  { rank=same; train_o; train_f; }')
    L('  { rank=same; eval_o; eval_f; }')
    L('  { rank=same; plots_o; plots_f; }')
    L('  { rank=same; out_model_o; out_eval_o; out_plot_o; out_model_f; out_eval_f; out_plot_f; }')

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
