"""
Generate the dense-IVQ figure for §4 / §5.

Source: lora_intervention/results/dense_ivq_ivq_*.json
  Structure: {results: {base: {cell_key: {position: {accuracy, ...}}},
                        lora: {cell_key: {position: {accuracy, ...}}}}}

Outputs (paper/figures/):
- fig_dense_ivq.pdf  — per-position accuracy at K=5, N=10, base vs LoRA
- fig_dense_ivq.png  — same as PNG for previewing

Run:
    .venv/bin/python paper/scripts/generate_dense_ivq_figure.py
"""

import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# Share heatmap helpers with the main results generator (percent-aware
# variants come from generate_lora_tables which already wraps them).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_main_results import _acc_shade  # noqa: E402


def _acc_pct_cell(x):
    """Shaded accuracy cell in percent form: \cellcolor{blue!24}60\%."""
    return rf"\cellcolor{{blue!{_acc_shade(x)}}}{x * 100:.0f}\%"

IVQ_GLOB = str(_ROOT / "lora_intervention" / "results" / "dense_ivq_ivq_*.json")


def load_positions(json_path):
    """Return {dataset: {cell_key: {position_int: accuracy}}}.

    JSON structure:
        results.{base,lora}.{train,held_moderate,held_hard}.{cell_key}.{position}.accuracy
    We flatten the train/held_moderate/held_hard grouping — cell_key is unique.
    """
    d = json.loads(Path(json_path).read_text())
    out = {}
    for dataset_key, groups in d["results"].items():
        out[dataset_key] = {}
        for group_key, cells in groups.items():
            for cell_key, positions in cells.items():
                posdict = {}
                for p, payload in positions.items():
                    if p == "last":
                        posdict[-1] = payload["accuracy"]
                    else:
                        try:
                            posdict[int(p)] = payload["accuracy"]
                        except ValueError:
                            continue
                out[dataset_key][cell_key] = posdict
    return out


def make_figure(out_path, cell_key="5_10"):
    files = sorted(glob.glob(IVQ_GLOB))
    if not files:
        print("  WARNING: no dense_ivq JSON found")
        return
    data = load_positions(files[-1])

    base_cell = data.get("base", {}).get(cell_key)
    lora_cell = data.get("lora", {}).get(cell_key)
    if base_cell is None or lora_cell is None:
        print(f"  WARNING: cell {cell_key} missing in dense_ivq data")
        return

    K, N = map(int, cell_key.split("_"))
    # Drop position N: the experiment script (evaluate_ivq.py:121) substitutes
    # the CVQ ``last value of'' prompt for k=N, so position N is operationally a
    # CVQ query and does not belong on an IVQ axis (matches \S4's $1 < k < N$
    # definition; we keep k=1 because the ordinal ``1st'' prompt is faithful).
    positions = sorted(p for p in base_cell.keys() if 0 < p < N)

    base_y = [base_cell.get(p, 0) for p in positions]
    lora_y = [lora_cell.get(p, 0) for p in positions]

    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    ax.plot(positions, base_y, marker="o", markersize=4, lw=1.4, color="#d62728",
            label="Base model")
    ax.plot(positions, lora_y, marker="s", markersize=4, lw=1.4, color="#1f77b4",
            label="+ LoRA (18k examples)")
    ax.set_xlabel(f"Query position $k$ in a stream of $N{{=}}{N}$ updates",
                  fontsize=9)
    ax.set_ylabel("Retrieval accuracy", fontsize=9)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(positions)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)

    # In-axes legend (centre-right empty space) to save vertical real estate.
    ax.legend(loc="center right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


CELL_ORDER = [
    ("5_10",  "train"),
    ("10_20", "train"),
    ("10_30", "held_moderate"),
    ("15_30", "held_moderate"),
    ("20_30", "held_hard"),
    ("25_50", "held_hard"),
]


def make_panel_figure(out_path):
    """2x3 panel showing per-position accuracy across all 6 IVQ cells."""
    files = sorted(glob.glob(IVQ_GLOB))
    if not files:
        print("  WARNING: no dense_ivq JSON found")
        return
    data = load_positions(files[-1])

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.2), sharey=True)
    axes = axes.flatten()
    for ax, (cell_key, group) in zip(axes, CELL_ORDER):
        base_cell = data.get("base", {}).get(cell_key)
        lora_cell = data.get("lora", {}).get(cell_key)
        K, N = map(int, cell_key.split("_"))
        if base_cell is None or lora_cell is None:
            ax.set_title(f"$K{{=}}{K}, N{{=}}{N}$ (missing)", fontsize=9)
            continue
        positions = sorted(p for p in base_cell.keys() if 0 < p < N)
        base_y = [base_cell.get(p, 0) for p in positions]
        lora_y = [lora_cell.get(p, 0) for p in positions]
        ax.plot(positions, base_y, marker="o", markersize=3, lw=1.1, color="#d62728")
        ax.plot(positions, lora_y, marker="s", markersize=3, lw=1.1, color="#1f77b4")
        # Subsample x-ticks for high-N cells
        tick_step = max(1, len(positions) // 8)
        ax.set_xticks(positions[::tick_step])
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.3)
        ax.set_title(f"$K{{=}}{K}, N{{=}}{N}$ ({group.replace('_', '-')})", fontsize=9)
    # Common labels
    for ax in axes[0:3]:
        ax.tick_params(labelbottom=False)
    for col in [0]:
        for row in [0, 1]:
            axes[row * 3 + col].set_ylabel("Accuracy", fontsize=9)
    for ax in axes[3:6]:
        ax.set_xlabel("Query position $k$", fontsize=9)

    # Shared legend
    base_proxy = plt.Line2D([0], [0], color="#d62728", marker="o", lw=1.1, label="Base model")
    lora_proxy = plt.Line2D([0], [0], color="#1f77b4", marker="s", lw=1.1, label="+ LoRA (18k examples)")
    fig.legend(handles=[base_proxy, lora_proxy], loc="lower center", ncol=2,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def make_summary_table(out_path):
    """Per-cell summary: first / intermediate-range / last accuracy, base vs LoRA."""
    files = sorted(glob.glob(IVQ_GLOB))
    if not files:
        print("  WARNING: no dense_ivq JSON found")
        return
    data = load_positions(files[-1])

    lines = [
        "% Auto-generated by paper/scripts/generate_dense_ivq_figure.py",
        "% Per-cell summary of dense-IVQ retrieval accuracy, base vs LoRA",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{\textbf{Base}} & \multicolumn{3}{c}{\textbf{+ LoRA}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"$(K, N)$ & Pos.~1 & Interm.~min--max & Last & Pos.~1 & Interm.~min--max & Last \\",
        r"\midrule",
    ]
    for cell_key, group in CELL_ORDER:
        K, N = map(int, cell_key.split("_"))
        base_cell = data.get("base", {}).get(cell_key) or {}
        lora_cell = data.get("lora", {}).get(cell_key) or {}
        positions = sorted(p for p in base_cell.keys() if 0 < p < N)
        if not positions:
            lines.append(f"  $({K}, {N})$ & --- & --- & --- & --- & --- & --- \\\\")
            continue
        first_pos, last_pos = positions[0], positions[-1]
        interm_positions = positions[1:-1] if len(positions) >= 3 else []
        b_first = base_cell.get(first_pos, 0); l_first = lora_cell.get(first_pos, 0)
        b_last  = base_cell.get(last_pos, 0);  l_last  = lora_cell.get(last_pos, 0)
        if interm_positions:
            b_int = [base_cell.get(p, 0) for p in interm_positions]
            l_int = [lora_cell.get(p, 0) for p in interm_positions]
            # Range strings: shade by the worst (min) endpoint, but show both.
            b_int_str = f"{_acc_pct_cell(min(b_int))}--{_acc_pct_cell(max(b_int))}"
            l_int_str = f"{_acc_pct_cell(min(l_int))}--{_acc_pct_cell(max(l_int))}"
        else:
            b_int_str = l_int_str = "---"
        lines.append(
            f"  $({K}, {N})$ & {_acc_pct_cell(b_first)} & {b_int_str} & {_acc_pct_cell(b_last)} & "
            f"{_acc_pct_cell(l_first)} & {l_int_str} & {_acc_pct_cell(l_last)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"  saved {out_path}")


def main():
    print(f"Generating dense-IVQ figure to {OUT_DIR} …")
    make_figure(OUT_DIR / "fig_dense_ivq.pdf")
    make_panel_figure(OUT_DIR / "fig_dense_ivq_panel.pdf")
    make_summary_table(OUT_DIR / "tab_dense_ivq.tex")
    print("Done.")


if __name__ == "__main__":
    main()
