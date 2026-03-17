"""
Q6: Fit formal scaling law for PI accuracy as function of N and model size.

Fits:
  PI(N) = a * exp(-b * N) + c    (exponential decay to floor)
  RI(N) = d - e * log(N)          (slow logarithmic decay)

Also fits PI vs model_size at fixed N values.

Usage:
    cd v3 && python fit_scaling_law.py
"""

import json
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI."""
    if n == 0:
        return 0, 0, 0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    halfwidth = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0, center - halfwidth), min(1, center + halfwidth)


def exp_decay(N, a, b, c):
    """PI(N) = a * exp(-b * N) + c"""
    return a * np.exp(-b * np.array(N)) + c


def log_decay(N, d, e):
    """RI(N) = d - e * log(N), clipped to [0, 1]"""
    return np.clip(d - e * np.log(np.array(N)), 0, 1)


def load_all_data():
    """Load aggregated stage1 data."""
    data_file = RESULTS_DIR / "all_stage1_data.json"
    if not data_file.exists():
        raise FileNotFoundError("Run the data aggregation first")
    return json.load(open(data_file))


# Model sizes in billions of parameters
MODEL_SIZES = {
    "Qwen2.5-0.5B-Instruct": 0.494,
    "Qwen2.5-1.5B-Instruct": 1.5,
    "Qwen2.5-3B-Instruct": 3.0,
    "Qwen2.5-3B ": 3.0,  # base model (note trailing space in dir name)
    "gemma-3-1b-it": 1.0,
    "TinyLlama-1.1B-Chat-v1.0": 1.1,
    "stablelm-2-1_6b-chat": 1.6,
    "pythia-410m": 0.41,
    "mamba-1.4b-hf": 1.4,
    "mamba-130m-hf": 0.13,
}

MODEL_TYPES = {
    "Qwen2.5-0.5B-Instruct": "Transformer",
    "Qwen2.5-1.5B-Instruct": "Transformer",
    "Qwen2.5-3B-Instruct": "Transformer",
    "Qwen2.5-3B ": "Transformer (base)",
    "gemma-3-1b-it": "Transformer",
    "TinyLlama-1.1B-Chat-v1.0": "Transformer",
    "stablelm-2-1_6b-chat": "Transformer",
    "pythia-410m": "Transformer (base)",
    "mamba-1.4b-hf": "SSM",
    "mamba-130m-hf": "SSM",
}


def main():
    data = load_all_data()

    # ──────────────────────────────────────────────────────────────
    # Part 1: PI(N) decay curves for each model (at 2 keys)
    # ──────────────────────────────────────────────────────────────
    print("=" * 70)
    print("PART 1: PI(N) and RI(N) decay curves at 2 keys")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    fit_results = {}

    for model_name, model_data in sorted(data.items()):
        if model_name in ("rwkv-4-430m-pile", "mamba-130m-hf"):
            continue  # too much garbage

        N_vals, pi_vals, ri_vals, pi_ns, ri_ns = [], [], [], [], []
        for cell_key, conds in sorted(model_data.items()):
            k, u = cell_key.split("_")
            if int(k) != 2:
                continue
            ri = conds.get("RI", {})
            pi = conds.get("PI", {})
            if not ri or not pi:
                continue
            N_vals.append(int(u))
            pi_vals.append(pi["acc"])
            ri_vals.append(ri["acc"])
            pi_ns.append(pi["n"])
            ri_ns.append(ri["n"])

        if len(N_vals) < 3:
            print(f"  {model_name}: only {len(N_vals)} points, skipping fit")
            continue

        N_arr = np.array(N_vals)
        pi_arr = np.array(pi_vals)
        ri_arr = np.array(ri_vals)

        # Fit PI(N) = a * exp(-b * N) + c
        try:
            popt_pi, pcov_pi = curve_fit(
                exp_decay, N_arr, pi_arr,
                p0=[0.5, 0.1, 0.15],
                bounds=([0, 0, 0], [1.5, 2, 1]),
                maxfev=10000,
            )
            pi_pred = exp_decay(N_arr, *popt_pi)
            pi_r2 = 1 - np.sum((pi_arr - pi_pred) ** 2) / np.sum((pi_arr - np.mean(pi_arr)) ** 2)
            pi_fit = {"a": popt_pi[0], "b": popt_pi[1], "c": popt_pi[2], "R2": pi_r2}
        except Exception as e:
            pi_fit = {"error": str(e)}
            pi_r2 = -1

        # Fit RI(N) = d - e * log(N)
        try:
            popt_ri, pcov_ri = curve_fit(
                log_decay, N_arr, ri_arr,
                p0=[1.0, 0.05],
                bounds=([0, -0.5], [1.5, 0.5]),
                maxfev=10000,
            )
            ri_pred = log_decay(N_arr, *popt_ri)
            ri_r2 = 1 - np.sum((ri_arr - ri_pred) ** 2) / np.sum((ri_arr - np.mean(ri_arr)) ** 2)
            ri_fit = {"d": popt_ri[0], "e": popt_ri[1], "R2": ri_r2}
        except Exception as e:
            ri_fit = {"error": str(e)}
            ri_r2 = -1

        size = MODEL_SIZES.get(model_name, "?")
        mtype = MODEL_TYPES.get(model_name, "?")
        print(f"\n  {model_name} ({size}B, {mtype}):")
        print(f"    PI fit: PI(N) = {pi_fit.get('a',0):.3f} * exp(-{pi_fit.get('b',0):.4f} * N) + {pi_fit.get('c',0):.3f}  R²={pi_fit.get('R2',0):.3f}")
        print(f"    RI fit: RI(N) = {ri_fit.get('d',0):.3f} - {ri_fit.get('e',0):.4f} * log(N)  R²={ri_fit.get('R2',0):.3f}")
        print(f"    PI floor (c): {pi_fit.get('c',0):.1%}")
        print(f"    RI at N=50: {log_decay(50, *popt_ri) if 'error' not in ri_fit else 'N/A'}")

        fit_results[model_name] = {
            "size_B": size,
            "type": mtype,
            "N_values": N_vals,
            "PI_values": pi_vals,
            "RI_values": ri_vals,
            "PI_fit": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in pi_fit.items()},
            "RI_fit": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in ri_fit.items()},
        }

    # ──────────────────────────────────────────────────────────────
    # Part 2: PI vs Model Size at fixed N
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PART 2: PI accuracy vs Model Size at fixed N")
    print("=" * 70)

    fixed_N_values = [5, 10, 20]
    size_vs_pi = {N: {"sizes": [], "pi": [], "ri": [], "models": []} for N in fixed_N_values}

    for model_name, model_data in data.items():
        if model_name in ("rwkv-4-430m-pile", "mamba-130m-hf"):
            continue
        size = MODEL_SIZES.get(model_name)
        if size is None:
            continue

        for N in fixed_N_values:
            cell_key = f"2_{N}"
            if cell_key in model_data:
                conds = model_data[cell_key]
                ri = conds.get("RI", {})
                pi = conds.get("PI", {})
                if ri and pi:
                    size_vs_pi[N]["sizes"].append(size)
                    size_vs_pi[N]["pi"].append(pi["acc"])
                    size_vs_pi[N]["ri"].append(ri["acc"])
                    size_vs_pi[N]["models"].append(model_name)

    for N in fixed_N_values:
        d = size_vs_pi[N]
        if len(d["sizes"]) < 3:
            print(f"  N={N}: only {len(d['sizes'])} models, skipping")
            continue

        sizes = np.array(d["sizes"])
        pi = np.array(d["pi"])
        ri = np.array(d["ri"])

        # Correlation
        if len(sizes) >= 3:
            r_pi, p_pi = pearsonr(np.log(sizes), pi)
            r_ri, p_ri = pearsonr(np.log(sizes), ri)
            print(f"\n  N={N} ({len(sizes)} models):")
            print(f"    PI vs log(size): r={r_pi:.3f}, p={p_pi:.3f}")
            print(f"    RI vs log(size): r={r_ri:.3f}, p={p_ri:.3f}")
            for i, m in enumerate(d["models"]):
                print(f"      {m}: {sizes[i]:.2f}B → RI={ri[i]:.0%} PI={pi[i]:.0%}")

    # ──────────────────────────────────────────────────────────────
    # Part 3: Generate publication figure
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel (a): PI(N) with fitted curves
    ax = axes[0]
    colors = plt.cm.Set2(np.linspace(0, 1, 10))
    color_map = {}
    ci = 0
    for model_name in sorted(fit_results.keys()):
        fr = fit_results[model_name]
        if "error" in fr.get("PI_fit", {}):
            continue
        N = np.array(fr["N_values"])
        PI = np.array(fr["PI_values"])
        size = fr["size_B"]
        label = f"{model_name.split('-')[0]} {size}B"
        if "mamba" in model_name.lower():
            label = f"Mamba {size}B"
        elif "gemma" in model_name.lower():
            label = f"Gemma {size}B"
        elif "tiny" in model_name.lower():
            label = f"TinyLlama {size}B"
        elif "stable" in model_name.lower():
            label = f"StableLM {size}B"
        elif "pythia" in model_name.lower():
            label = f"Pythia {size}B"
        color_map[model_name] = colors[ci]
        ax.scatter(N, PI, color=colors[ci], s=40, zorder=5, alpha=0.8)

        # Plot fitted curve
        pf = fr["PI_fit"]
        N_smooth = np.linspace(min(N), max(N), 100)
        PI_smooth = exp_decay(N_smooth, pf["a"], pf["b"], pf["c"])
        ax.plot(N_smooth, PI_smooth, color=colors[ci], linestyle="--", alpha=0.6,
                label=f'{label} (floor={pf["c"]:.0%})')
        ci += 1

    ax.set_xlabel("Updates per key (N)", fontsize=12)
    ax.set_ylabel("PI Accuracy", fontsize=12)
    ax.set_title("(a) PI decays exponentially with N", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)

    # Panel (b): RI(N) with fitted curves
    ax = axes[1]
    ci = 0
    for model_name in sorted(fit_results.keys()):
        fr = fit_results[model_name]
        if "error" in fr.get("RI_fit", {}):
            continue
        N = np.array(fr["N_values"])
        RI = np.array(fr["RI_values"])
        size = fr["size_B"]
        label = f"{model_name.split('-')[0]} {size}B"
        if "mamba" in model_name.lower():
            label = f"Mamba {size}B"
        elif "gemma" in model_name.lower():
            label = f"Gemma {size}B"
        elif "tiny" in model_name.lower():
            label = f"TinyLlama {size}B"
        elif "stable" in model_name.lower():
            label = f"StableLM {size}B"
        elif "pythia" in model_name.lower():
            label = f"Pythia {size}B"
        ax.scatter(N, RI, color=colors[ci], s=40, zorder=5, alpha=0.8)

        rf = fr["RI_fit"]
        N_smooth = np.linspace(min(N), max(N), 100)
        RI_smooth = log_decay(N_smooth, rf["d"], rf["e"])
        ax.plot(N_smooth, RI_smooth, color=colors[ci], linestyle="-", alpha=0.6,
                label=f'{label}')
        ci += 1

    ax.set_xlabel("Updates per key (N)", fontsize=12)
    ax.set_ylabel("RI Accuracy", fontsize=12)
    ax.set_title("(b) RI degrades slowly (logarithmic)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(-0.05, 1.05)

    # Panel (c): PI vs model size at fixed N
    ax = axes[2]
    markers = ["o", "s", "D"]
    for i, N in enumerate(fixed_N_values):
        d = size_vs_pi[N]
        if len(d["sizes"]) < 2:
            continue
        sizes = np.array(d["sizes"])
        pi = np.array(d["pi"])
        ri = np.array(d["ri"])
        sort_idx = np.argsort(sizes)
        ax.plot(sizes[sort_idx], pi[sort_idx], marker=markers[i], linestyle="-",
                label=f"PI (N={N})", markersize=8)
        ax.plot(sizes[sort_idx], ri[sort_idx], marker=markers[i], linestyle="--",
                alpha=0.4, label=f"RI (N={N})", markersize=6)

    ax.set_xlabel("Model Size (B params)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("(c) RI scales with size; PI converges", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    save_path = FIGURES_DIR / "scaling_law_fitted.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {save_path}")

    # Save fit results
    save_json = RESULTS_DIR / "scaling_law_fits.json"
    json.dump(fit_results, open(save_json, "w"), indent=2, default=float)
    print(f"Fit results saved: {save_json}")

    # ──────────────────────────────────────────────────────────────
    # Part 4: Summary table for paper
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PAPER TABLE: Scaling Law Summary")
    print("=" * 70)
    print(f"\n{'Model':<30} {'Size':>5} {'PI floor':>9} {'PI decay':>9} {'PI R²':>6} {'RI@50':>6} {'RI R²':>6}")
    print("-" * 70)
    for model_name in sorted(fit_results.keys(), key=lambda x: MODEL_SIZES.get(x, 0)):
        fr = fit_results[model_name]
        pf = fr.get("PI_fit", {})
        rf = fr.get("RI_fit", {})
        size = fr["size_B"]
        pi_floor = pf.get("c", "?")
        pi_b = pf.get("b", "?")
        pi_r2 = pf.get("R2", "?")
        ri_r2 = rf.get("R2", "?")
        ri_at_50 = log_decay(50, rf.get("d", 1), rf.get("e", 0)) if "error" not in rf else "?"

        pi_floor_s = f"{pi_floor:.1%}" if isinstance(pi_floor, float) else "?"
        pi_b_s = f"{pi_b:.4f}" if isinstance(pi_b, float) else "?"
        pi_r2_s = f"{pi_r2:.3f}" if isinstance(pi_r2, float) else "?"
        ri_r2_s = f"{ri_r2:.3f}" if isinstance(ri_r2, float) else "?"
        ri_50_s = f"{ri_at_50:.0%}" if isinstance(ri_at_50, float) else "?"

        print(f"{model_name:<30} {size:>5.2f} {pi_floor_s:>9} {pi_b_s:>9} {pi_r2_s:>6} {ri_50_s:>6} {ri_r2_s:>6}")


if __name__ == "__main__":
    main()
