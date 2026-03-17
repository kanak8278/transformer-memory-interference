"""
Generate publication-quality results table with Wilson confidence intervals.

Outputs Table 2: Behavioral summary at matched operating points.

Usage:
    cd v3 && python generate_paper_table.py
"""

import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI."""
    if n == 0:
        return 0, 0, 0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    halfwidth = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0, center - halfwidth), min(1, center + halfwidth)


def load_all_data():
    return json.load(open(RESULTS_DIR / "all_stage1_data.json"))


MODEL_INFO = {
    "Qwen2.5-0.5B-Instruct": {"size": "494M", "arch": "Transformer (GQA)", "type": "Instruct"},
    "Qwen2.5-1.5B-Instruct": {"size": "1.5B", "arch": "Transformer (GQA)", "type": "Instruct"},
    "Qwen2.5-3B-Instruct": {"size": "3B", "arch": "Transformer (GQA)", "type": "Instruct"},
    "Qwen2.5-3B ": {"size": "3B", "arch": "Transformer (GQA)", "type": "Base"},
    "gemma-3-1b-it": {"size": "1B", "arch": "Transformer (MHA)", "type": "Instruct"},
    "TinyLlama-1.1B-Chat-v1.0": {"size": "1.1B", "arch": "Transformer (Llama)", "type": "Chat"},
    "stablelm-2-1_6b-chat": {"size": "1.6B", "arch": "Transformer", "type": "Chat"},
    "pythia-410m": {"size": "410M", "arch": "Transformer (GPT-NeoX)", "type": "Base"},
    "mamba-1.4b-hf": {"size": "1.4B", "arch": "SSM (Mamba)", "type": "Base"},
    "mamba-130m-hf": {"size": "130M", "arch": "SSM (Mamba)", "type": "Base"},
    "rwkv-4-430m-pile": {"size": "430M", "arch": "RWKV", "type": "Base"},
}


def main():
    data = load_all_data()

    # ──────────────────────────────────────────────────────────────
    # Table 2: Behavioral results at matched operating points
    # ──────────────────────────────────────────────────────────────
    print("=" * 100)
    print("TABLE 2: PI > RI Across Architectures (2 keys)")
    print("=" * 100)

    # Standard operating points to compare
    standard_points = [(2, 5), (2, 10), (2, 20)]

    # Header
    header = f"{'Model':<30} {'Size':>5} {'Arch':>15} {'Type':>8}"
    for k, u in standard_points:
        header += f" {'RI@'+str(u):>10} {'PI@'+str(u):>10} {'Gap@'+str(u):>8}"
    print(header)
    print("-" * len(header))

    for model_name in sorted(data.keys(), key=lambda x: MODEL_INFO.get(x, {}).get("size", "0")):
        info = MODEL_INFO.get(model_name, {"size": "?", "arch": "?", "type": "?"})
        model_data = data[model_name]

        row = f"{model_name:<30} {info['size']:>5} {info['arch']:>15} {info['type']:>8}"

        for k, u in standard_points:
            cell_key = f"{k}_{u}"
            if cell_key in model_data:
                conds = model_data[cell_key]
                ri = conds.get("RI", {})
                pi = conds.get("PI", {})

                if ri and pi:
                    ri_acc = ri["acc"]
                    pi_acc = pi["acc"]
                    ri_n = ri["n"]
                    pi_n = pi["n"]

                    # Wilson CIs
                    _, ri_lo, ri_hi = wilson_ci(int(ri_acc * ri_n), ri_n)
                    _, pi_lo, pi_hi = wilson_ci(int(pi_acc * pi_n), pi_n)
                    gap = ri_acc - pi_acc

                    ri_ci = f"{ri_acc:.0%}±{(ri_hi-ri_lo)/2:.0%}"
                    pi_ci = f"{pi_acc:.0%}±{(pi_hi-pi_lo)/2:.0%}"

                    row += f" {ri_ci:>10} {pi_ci:>10} {gap:>+7.0%}"
                else:
                    row += f" {'—':>10} {'—':>10} {'—':>8}"
            else:
                row += f" {'—':>10} {'—':>10} {'—':>8}"

        print(row)

    # ──────────────────────────────────────────────────────────────
    # LaTeX version
    # ──────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("LATEX TABLE")
    print("=" * 100)

    print(r"""
\begin{table}[t]
\centering
\caption{PI > RI across architectures. All results at 2 keys with 95\% Wilson CIs. Gap = RI -- PI.}
\label{tab:behavioral}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
Model & Params & Architecture & \multicolumn{2}{c}{N=5} & \multicolumn{2}{c}{N=10} & \multicolumn{2}{c}{N=20} \\
\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
& & & RI & PI & RI & PI & RI & PI \\
\midrule""")

    for model_name in sorted(data.keys(), key=lambda x: float(
            MODEL_INFO.get(x, {}).get("size", "0").replace("B", "e9").replace("M", "e6")
            .replace("?", "0"))):
        info = MODEL_INFO.get(model_name, {"size": "?", "arch": "?", "type": "?"})
        if model_name in ("rwkv-4-430m-pile", "mamba-130m-hf"):
            continue  # skip unusable models

        model_data = data[model_name]
        display_name = model_name.replace("_", r"\_").replace("-Instruct", "").replace("-Chat-v1.0", "").strip()

        cells = []
        for k, u in standard_points:
            cell_key = f"{k}_{u}"
            if cell_key in model_data:
                conds = model_data[cell_key]
                ri = conds.get("RI", {})
                pi = conds.get("PI", {})

                if ri and pi:
                    ri_n = ri["n"]
                    pi_n = pi["n"]
                    _, ri_lo, ri_hi = wilson_ci(int(ri["acc"] * ri_n), ri_n)
                    _, pi_lo, pi_hi = wilson_ci(int(pi["acc"] * pi_n), pi_n)

                    ri_str = f"${ri['acc']:.0%}$" if ri_n >= 100 else f"${ri['acc']:.0%}^*$"
                    pi_str = f"${pi['acc']:.0%}$" if pi_n >= 100 else f"${pi['acc']:.0%}^*$"
                    cells.extend([ri_str, pi_str])
                else:
                    cells.extend(["--", "--"])
            else:
                cells.extend(["--", "--"])

        print(f"{display_name} & {info['size']} & {info['arch']} & " + " & ".join(cells) + r" \\")

    print(r"""\bottomrule
\end{tabular}%
}
\vspace{-1mm}
\footnotesize{$^*$ fewer than 100 trials (wider CIs). All others have 200 trials (Wilson CI $\leq \pm7\%$).}
\end{table}""")

    # ──────────────────────────────────────────────────────────────
    # Summary statistics
    # ──────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)

    all_gaps = []
    for model_name, model_data in data.items():
        if model_name in ("rwkv-4-430m-pile", "mamba-130m-hf"):
            continue
        for cell_key, conds in model_data.items():
            k, u = cell_key.split("_")
            if int(k) != 2:
                continue
            ri = conds.get("RI", {})
            pi = conds.get("PI", {})
            if ri and pi:
                gap = ri["acc"] - pi["acc"]
                all_gaps.append({"model": model_name, "updates": int(u), "gap": gap,
                                "ri": ri["acc"], "pi": pi["acc"]})

    gaps = [g["gap"] for g in all_gaps]
    print(f"\n  Total cells: {len(all_gaps)}")
    print(f"  Cells with PI > RI (positive gap): {sum(1 for g in gaps if g > 0)} ({sum(1 for g in gaps if g > 0)/len(gaps):.0%})")
    print(f"  Mean gap: {np.mean(gaps):.1%}")
    print(f"  Median gap: {np.median(gaps):.1%}")
    print(f"  Max gap: {max(gaps):.1%}")
    print(f"  Min gap: {min(gaps):.1%}")
    print(f"  Negative gaps: {sum(1 for g in gaps if g < 0)} (PI better than RI)")

    # Per-model summary
    print(f"\n  Per-model mean gap (2 keys only):")
    from collections import defaultdict
    model_gaps = defaultdict(list)
    for g in all_gaps:
        model_gaps[g["model"]].append(g["gap"])
    for m in sorted(model_gaps.keys()):
        gs = model_gaps[m]
        print(f"    {m:<35} mean_gap={np.mean(gs):+.1%} n_cells={len(gs)}")


if __name__ == "__main__":
    main()
