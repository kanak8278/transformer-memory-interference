"""
Analyse + plot the Entropy-Lens results across base / lora / scratch.

Comparability is enforced two ways (see run_entropy_lens.py):
  - y-axis: normalised entropy H / log(vocab)  (Qwen V=151936 vs scratch V=51)
  - x-axis: relative depth  layer / n_layers   (Qwen 36 vs scratch 12 layers)

Produces:
  1. entropy_profiles.png  -- per query-type (FVQ/CVQ/IVQ_d50), the mean
     normalised-entropy profile vs relative depth, base vs lora vs scratch.
  2. entropy_confusion.png -- correct-vs-wrong entropy profiles per model
     (the "confusion" view: do wrong answers stay more uncertain across depth?).
  3. entropy_summary.json  -- per model/condition scalars (output entropy,
     min-entropy depth, expansion amplitude).
"""
import glob
import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
RESULTS = _HERE / "results"
MODELS = ["base", "lora", "scratch"]
COLORS = {"base": "#1f77b4", "lora": "#d62728", "scratch": "#2ca02c"}


def load():
    out = {}
    for m in MODELS:
        f = RESULTS / f"entropy_{m}.json"
        if f.exists():
            out[m] = json.load(open(f))
    return out


def rel_depth(n_layers):
    # layer 1..n_layers mapped to (0,1]
    return np.arange(1, n_layers + 1) / n_layers


def mean_profile(data, condition, subset):
    """Mean normalised-entropy profile across all cells for a condition/subset."""
    profs = []
    for cell, conds in data["cells"].items():
        c = conds.get(condition)
        if c and c.get(subset):
            profs.append(c[subset]["norm"])
    if not profs:
        return None
    return np.array(profs).mean(0)


def weighted_profile(data, condition, subset):
    """n-weighted mean across cells (for correct/wrong where n varies a lot)."""
    num, den = None, 0
    for cell, conds in data["cells"].items():
        c = conds.get(condition)
        if c and c.get(subset):
            w = c[subset]["n"]
            p = np.array(c[subset]["norm"]) * w
            num = p if num is None else num + p
            den += w
    return None if den == 0 else num / den


def plot_profiles(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = [("FVQ", "first value"), ("CVQ", "last value"), ("IVQ_d50", "intermediate (50%)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, (cond, title) in zip(axes, conds):
        for m, d in data.items():
            p = mean_profile(d, cond, "all")
            if p is None:
                continue
            ax.plot(rel_depth(d["n_layers"]), p, "-o", ms=3, color=COLORS[m],
                    label=f"{m} ({d['n_layers']}L)")
        ax.set_title(f"Entropy lens — {title}", fontsize=11)
        ax.set_xlabel("relative depth (layer / n_layers)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("normalised entropy  H / log(V)")
    fig.suptitle("Entropy-Lens across models (mean over KV-interference cells), by query type",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    print("Saved", path)


def plot_confusion(data, path, condition="IVQ_d50"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(data), figsize=(5.5 * len(data), 5), sharey=True)
    if len(data) == 1:
        axes = [axes]
    for ax, (m, d) in zip(axes, data.items()):
        x = rel_depth(d["n_layers"])
        for subset, ls, lab in [("correct", "-", "correct"), ("wrong", "--", "wrong")]:
            p = weighted_profile(d, condition, subset)
            if p is not None:
                ax.plot(x, p, ls, color=COLORS[m], label=lab, lw=2)
        ax.set_title(f"{m}: {condition} correct vs wrong", fontsize=11)
        ax.set_xlabel("relative depth")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("normalised entropy  H / log(V)")
    fig.suptitle(f'"Confusion" view: does a wrong answer stay more uncertain across depth? ({condition})',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    print("Saved", path)


def summarize(data):
    summ = {}
    for m, d in data.items():
        summ[m] = {"n_layers": d["n_layers"], "vocab": d["vocab_size"], "conditions": {}}
        for cond in d["conditions"]:
            p = mean_profile(d, cond, "all")
            if p is None:
                continue
            summ[m]["conditions"][cond] = {
                "output_entropy": float(p[-1]),
                "peak_entropy": float(p.max()),
                "min_entropy": float(p.min()),
                "peak_depth": float((p.argmax() + 1) / d["n_layers"]),
                "expansion_amplitude": float(p.max() - p[0]),
                "pruning_drop": float(p.max() - p[-1]),
            }
    return summ


def main():
    data = load()
    if not data:
        print("No entropy_*.json results yet in", RESULTS)
        return
    print("Loaded:", list(data.keys()))
    try:
        plot_profiles(data, RESULTS / "entropy_profiles.png")
        plot_confusion(data, RESULTS / "entropy_confusion.png")
    except Exception as e:
        print("(plotting skipped:", e, ")")
    summ = summarize(data)
    json.dump(summ, open(RESULTS / "entropy_summary.json", "w"), indent=2)
    print("\n=== summary (normalised entropy) ===")
    for m, s in summ.items():
        print(f"\n{m} ({s['n_layers']}L, V={s['vocab']}):")
        for cond, v in s["conditions"].items():
            print(f"  {cond:9} output={v['output_entropy']:.2f} peak={v['peak_entropy']:.2f}"
                  f"@d{v['peak_depth']:.2f} expansion={v['expansion_amplitude']:+.2f} "
                  f"pruning={v['pruning_drop']:.2f}")
    print("\nSaved", RESULTS / "entropy_summary.json")


if __name__ == "__main__":
    main()
