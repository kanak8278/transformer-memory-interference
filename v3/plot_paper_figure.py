"""
Generate the main paper figure: a multi-panel figure combining key results.

4-panel layout:
(a) PI accuracy vs N across models (behavioral)
(b) Logit lens: P(v_last) suppressed vs P(v_first) clean
(c) Probing: RI/PI condition discrimination across layers
(d) Jacobian: Transformer vs SSM at initialization

Usage:
    cd v3 && python plot_paper_figure.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif",
})

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"


def load_stage1(model_short):
    model_dir = RESULTS_DIR / model_short
    trials = sorted(model_dir.glob("stage1_trials_*.json"))
    if trials:
        return json.load(open(trials[-1]))
    ckpt = model_dir / "stage1_checkpoint.json"
    if ckpt.exists():
        return json.load(open(ckpt))
    sweep = sorted(model_dir.glob("stage1_sweep_*.json"))
    if sweep:
        return json.load(open(sweep[-1]))
    return None


def load_stage2(model_short):
    model_dir = RESULTS_DIR / model_short
    files = sorted(model_dir.glob("stage2_logit_lens_*.json"))
    if files:
        return json.load(open(files[-1]))
    return None


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ── Panel (a): PI accuracy vs N ──
    ax = axes[0, 0]
    models = [
        ("Qwen2.5-1.5B-Instruct", "Qwen 1.5B", "#377eb8", "o"),
        ("Qwen2.5-3B-Instruct", "Qwen 3B", "#4daf4a", "s"),
        ("gemma-3-1b-it", "Gemma 1B", "#984ea3", "^"),
        ("mamba-1.4b-hf", "Mamba 1.4B", "#e41a1c", "D"),
    ]

    for model_short, label, color, marker in models:
        s1 = load_stage1(model_short)
        if s1 is None:
            continue
        cells = s1["cells"]

        # Extract 2-key data for RI and PI
        ri_data, pi_data = {}, {}
        for ck, cell in cells.items():
            nk, nu = int(ck.split("_")[0]), int(ck.split("_")[1])
            if nk != 2:
                continue
            ri_data[nu] = cell["stats"]["RI"]["accuracy"]
            pi_data[nu] = cell["stats"]["PI"]["accuracy"]

        if not pi_data:
            continue
        updates = sorted(pi_data.keys())
        ax.plot(updates, [pi_data[u] for u in updates], color=color, linewidth=2,
                marker=marker, markersize=5, label=f"{label} (PI)", alpha=0.8)
        ax.plot(updates, [ri_data[u] for u in updates], color=color, linewidth=1,
                linestyle="--", alpha=0.4)

    ax.set_xlabel("Updates per key (N)")
    ax.set_ylabel("Accuracy")
    ax.set_title("(a) PI accuracy collapses with N; RI stays robust")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)

    # ── Panel (b): Logit lens comparison ──
    ax = axes[0, 1]
    s2_models = [
        ("Qwen2.5-1.5B-Instruct", "Qwen 1.5B", "#377eb8"),
        ("Qwen2.5-3B-Instruct", "Qwen 3B", "#4daf4a"),
    ]

    for model_short, label, color in s2_models:
        s2 = load_stage2(model_short)
        if s2 is None:
            continue
        analyses = s2["analyses"]
        n_layers = s2["n_layers"]

        # Find best operating point
        from collections import defaultdict
        groups = defaultdict(list)
        for a in analyses:
            groups[(a["num_keys"], a["num_updates"], a["condition"])].append(a)

        best_k, best_u, best_n = None, None, 0
        for (k, u, cond), items in groups.items():
            if cond == "PI":
                fails = [a for a in items if not a["correct"] and a.get("total_value_prob", 0) > 0.05]
                if len(fails) > best_n:
                    best_n = len(fails)
                    best_k, best_u = k, u

        if best_k is None:
            continue

        pi_f = [a for a in analyses if a["condition"] == "PI" and a["num_keys"] == best_k
                and a["num_updates"] == best_u and not a["correct"]
                and a.get("total_value_prob", 0) > 0.05]
        ri_c = [a for a in analyses if a["condition"] == "RI" and a["num_keys"] == best_k
                and a["num_updates"] == best_u and a["correct"]]

        x = np.linspace(0, 1, n_layers)

        if pi_f:
            avg_last = np.mean([np.array(a["value_probs_by_layer"])[-1] for a in pi_f], axis=0)
            ax.plot(x, avg_last, color=color, linewidth=2, linestyle="-",
                    label=f"{label} P(v_last) PI", alpha=0.8)

        if ri_c:
            avg_first = np.mean([np.array(a["value_probs_by_layer"])[0] for a in ri_c], axis=0)
            ax.plot(x, avg_first, color=color, linewidth=2, linestyle="--",
                    label=f"{label} P(v_first) RI", alpha=0.6)

    ax.set_xlabel("Relative depth (0=input, 1=output)")
    ax.set_ylabel("P(value)")
    ax.set_title("(b) Logit lens: v_last suppressed, v_first clean")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(0.6, 1.0)
    ax.grid(True, alpha=0.3)

    # ── Panel (c): Probing classifier ──
    ax = axes[1, 0]
    probe_models = [
        ("probing_Qwen2.5-1.5B-Instruct_2k_5u.json", "Qwen 1.5B", "#377eb8"),
        ("probing_Qwen2.5-3B-Instruct_5k_3u.json", "Qwen 3B", "#4daf4a"),
        ("probing_gemma-3-1b-it_2k_5u.json", "Gemma 1B", "#984ea3"),
    ]

    for fname, label, color in probe_models:
        fpath = RESULTS_DIR / "probing" / fname
        if not fpath.exists():
            continue
        d = json.load(open(fpath))
        n_layers = d["n_layers"]
        cond_probe = d["probe_results"].get("condition_probe", {})
        ri_probe = d["probe_results"].get("RI_correct_probe", {})

        x = np.arange(n_layers)
        cond_acc = [cond_probe.get(str(L), {}).get("accuracy", 0.5) for L in range(n_layers)]
        ax.plot(x / n_layers, cond_acc, color=color, linewidth=2,
                label=f"{label} RI/PI disc.", alpha=0.8)

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Chance")
    ax.set_xlabel("Relative depth")
    ax.set_ylabel("Probe accuracy")
    ax.set_title("(c) Probing: RI/PI perfectly discriminable at late layers")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)

    # ── Panel (d): Jacobian comparison ──
    ax = axes[1, 1]
    jac_files = [
        ("jacobian_Qwen2.5-1.5B-Instruct_untrained.json", "Qwen 1.5B (init)", "#377eb8", "--"),
        ("jacobian_Qwen2.5-1.5B-Instruct_pretrained.json", "Qwen 1.5B (trained)", "#377eb8", "-"),
        ("jacobian_mamba-1.4b-hf_untrained.json", "Mamba 1.4B (init)", "#e41a1c", "--"),
        ("jacobian_mamba-1.4b-hf_pretrained.json", "Mamba 1.4B (trained)", "#e41a1c", "-"),
    ]

    for fname, label, color, ls in jac_files:
        fpath = RESULTS_DIR / "jacobian" / fname
        if not fpath.exists():
            continue
        d = json.load(open(fpath))
        rel = np.array(d["rel_influence"])
        x = np.linspace(0, 1, len(rel))
        ax.plot(x, rel, color=color, linestyle=ls, linewidth=2, label=label, alpha=0.8)

    ax.set_xlabel("Relative position (0=first, 1=last)")
    ax.set_ylabel("Relative influence")
    ax.set_title("(d) Jacobian: SSM has extreme primacy at initialization")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Mechanistic Evidence: Why Transformers Remember First, Forget Last",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    path = FIGURES_DIR / "paper_main_figure.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
