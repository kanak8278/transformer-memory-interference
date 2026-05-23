"""
Summarize few-shot results vs baseline for the paper.

Usage:
    python lora_intervention/fewshot_summary.py
"""
import sys, json, re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

BASELINE_PATH = _ROOT / "v3/results_vllm/arbitrary_single/Qwen2.5-3B-Instruct/stage1_sweep_20260409_000134.json"
FEWSHOT_CKPT  = _ROOT / "v3/results_vllm/fewshot/Qwen2.5-3B-Instruct/fewshot_5shot_checkpoint.json"

def load_baseline():
    with open(BASELINE_PATH) as f:
        return json.load(f)["cells"]

def load_fewshot():
    with open(FEWSHOT_CKPT) as f:
        return json.load(f)["cells"]

def main():
    base   = load_baseline()
    fs     = load_fewshot()
    common = sorted(set(base) & set(fs),
                    key=lambda x: (int(x.split("_")[0]), int(x.split("_")[1])))

    print(f"Common cells: {len(common)}")
    print()

    # ── Per-cell table ────────────────────────────────────────────────────────
    header = f"{'Cell':>8}  {'BSE RI':>6} {'BSE PI':>6} {'BSE gap':>7} {'BSE regime':>10}  " \
             f"{'FS  RI':>6} {'FS  PI':>6} {'FS  gap':>7} {'FS  regime':>10}  {'verdict':>12}"
    print(header)
    print("-" * len(header))

    stats = {"fixed": 0, "improved": 0, "same": 0, "worse": 0, "D_remain": 0}

    for ck in common:
        nk, nu = ck.split("_")
        b = base[ck];  f = fs[ck]
        b_ri  = b["stats"]["RI"]["accuracy"];  b_pi  = b["stats"]["PI"]["accuracy"]
        fs_ri = f["stats"]["RI"]["accuracy"];  fs_pi = f["stats"]["PI"]["accuracy"]
        b_gap = b_ri - b_pi;  fs_gap = fs_ri - fs_pi
        b_reg = b["regime"];  fs_reg = f["regime"]

        fixed    = fs_ri >= 0.65 and fs_pi >= 0.65
        shortcut = (fs_pi > b_pi + 0.10 and fs_ri < b_ri - 0.10)
        worse    = (fs_gap < b_gap - 0.10)

        if fixed:
            verdict = "✓ FIXED"; stats["fixed"] += 1
        elif shortcut:
            verdict = "⚠ SHORTCUT"; stats["worse"] += 1
        elif worse:
            verdict = "↓ worse"; stats["worse"] += 1
        elif fs_gap > b_gap + 0.05:
            verdict = "↑ improved"; stats["improved"] += 1
        else:
            verdict = "~ same"; stats["same"] += 1

        if fs_reg == "D":
            stats["D_remain"] += 1

        print(f"{nk:>2}k_{nu:>3}u  "
              f"{b_ri:>5.0%}  {b_pi:>5.0%}  {b_gap:>+6.0%}  {b_reg:>10}  "
              f"{fs_ri:>5.0%}  {fs_pi:>5.0%}  {fs_gap:>+6.0%}  {fs_reg:>10}  "
              f"{verdict:>12}")

    print()
    print(f"Summary over {len(common)} cells:")
    print(f"  Fixed (both RI≥65%, PI≥65%): {stats['fixed']}")
    print(f"  Gap improved (>5%):           {stats['improved']}")
    print(f"  Same:                         {stats['same']}")
    print(f"  Worse/shortcut:               {stats['worse']}")
    print(f"  Still regime D post few-shot: {stats['D_remain']}/{sum(1 for c in common if base[c]['regime']=='D')}")

    # ── PI reduction analysis (key finding) ───────────────────────────────────
    print()
    print("PI change in regime-D cells (few-shot effect on recency bias):")
    d_cells = [(ck, base[ck], fs[ck]) for ck in common if base[ck]["regime"] == "D"]
    pi_drops = [(b["stats"]["PI"]["accuracy"] - f["stats"]["PI"]["accuracy"])
                for _, b, f in d_cells]
    ri_drops = [(b["stats"]["RI"]["accuracy"] - f["stats"]["RI"]["accuracy"])
                for _, b, f in d_cells]
    if pi_drops:
        import statistics
        print(f"  Mean PI drop:  {statistics.mean(pi_drops):+.1%}  "
              f"(range {min(pi_drops):+.0%} to {max(pi_drops):+.0%})")
        print(f"  Mean RI drop:  {statistics.mean(ri_drops):+.1%}  "
              f"(range {min(ri_drops):+.0%} to {max(ri_drops):+.0%})")
        net_gap = statistics.mean([
            (f["stats"]["RI"]["accuracy"] - f["stats"]["PI"]["accuracy"]) -
            (b["stats"]["RI"]["accuracy"] - b["stats"]["PI"]["accuracy"])
            for _, b, f in d_cells
        ])
        print(f"  Mean gap change: {net_gap:+.1%} (positive = gap narrowed toward RI>PI)")


if __name__ == "__main__":
    main()
