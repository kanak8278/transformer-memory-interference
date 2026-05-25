"""
Generate the Gemma-3-4b-it LoRA per-cell tables for App J.

Inputs:
- ARB baseline + LoRA: lora_intervention/results/gemma_main_comparison.txt
  (28 held-out cells)
- SEM baseline:        v3/results_vllm/semantic_multi/gemma-3-4b-it/
                       stage1_sweep_*.json (62 cells; we read 16)
- SEM LoRA:            lora_intervention/results/gemma_main_eval_*.json
                       (16 SEM OOD cells)

Outputs (paper/figures/):
- tab_lora_gemma_arb.tex — 28 held-out ARB cells, base vs LoRA
- tab_lora_gemma_sem.tex — 16 SEM OOD cells, base vs LoRA

Run from project root:
    .venv/bin/python paper/scripts/generate_gemma_lora_tables.py
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_main_results import _acc_shade, _gap_shade  # noqa: E402


def fmt_acc(x):
    return rf"\cellcolor{{blue!{_acc_shade(x)}}}{x * 100:.0f}\%"


def fmt_gap(g):
    sign = "+" if g >= 0 else "$-$"
    return rf"\cellcolor{{red!{_gap_shade(g)}}}{sign}{abs(g) * 100:.0f}\%"


GEMMA_COMPARISON = _ROOT / "lora_intervention" / "results" / "gemma_main_comparison.txt"
GEMMA_LORA_EVAL = _ROOT / "lora_intervention" / "results" / "gemma_main_eval_20260525_165155.json"
GEMMA_SEM_BASELINE = (
    _ROOT / "v3" / "results_vllm" / "semantic_multi" / "gemma-3-4b-it"
    / "stage1_sweep_20260409_024127.json"
)


def load_gemma_arb_rows():
    """Parse gemma_main_comparison.txt rows like:
       7k_ 30u  95%  52%  +43%   100%   100%  +0%   ✓ FIXED
    """
    out = {}
    for line in GEMMA_COMPARISON.read_text().splitlines():
        s = line.strip()
        if (not s or s.startswith("=") or s.startswith("-")
                or s.startswith("Fixed:") or s.startswith("Cell")
                or "COMPARISON" in s):
            continue
        parts = s.split()
        if len(parts) < 7:
            continue
        try:
            k_token = parts[0].rstrip("k_").rstrip("k")
            n_token = parts[1].rstrip("u")
            K = int(k_token)
            N = int(n_token)
            # Baseline may be "---" (missing); LoRA columns are always numeric.
            if parts[2].startswith("---"):
                b_ri = b_pi = b_gap = None
                lora_offset = 5  # LoRA columns start at parts[5]
            else:
                b_ri = float(parts[2].rstrip("%")) / 100.0
                b_pi = float(parts[3].rstrip("%")) / 100.0
                b_gap = float(parts[4].rstrip("%").replace("+", "")) / 100.0
                lora_offset = 5
            f_ri = float(parts[lora_offset].rstrip("%")) / 100.0
            f_pi = float(parts[lora_offset + 1].rstrip("%")) / 100.0
            f_gap = float(parts[lora_offset + 2].rstrip("%").replace("+", "")) / 100.0
        except (ValueError, IndexError):
            continue
        out[(K, N)] = {
            "base_ri": b_ri, "base_pi": b_pi, "base_gap": b_gap,
            "lora_ri": f_ri, "lora_pi": f_pi, "lora_gap": f_gap,
        }
    return out


def load_gemma_sem_rows():
    """Build (K,N) -> {base_ri,base_pi,base_gap,lora_ri,lora_pi,lora_gap}.
    Baseline from stage1 sweep, LoRA from gemma_main_eval JSON.
    """
    base_d = json.loads(GEMMA_SEM_BASELINE.read_text())
    lora_d = json.loads(GEMMA_LORA_EVAL.read_text())["results"]["SEMANTIC_MULTI"]
    out = {}
    for ck, cell in lora_d.items():
        K = cell["num_keys"]
        N = cell["num_updates"]
        # baseline file uses "K_N" keys
        bkey = f"{K}_{N}"
        bcell = base_d["cells"].get(bkey)
        if bcell is None:
            continue
        b_ri = bcell["stats"]["RI"]["accuracy"]
        b_pi = bcell["stats"]["PI"]["accuracy"]
        l_ri = cell["stats"]["RI"]["accuracy"]
        l_pi = cell["stats"]["PI"]["accuracy"]
        out[(K, N)] = {
            "base_ri": b_ri, "base_pi": b_pi, "base_gap": b_ri - b_pi,
            "lora_ri": l_ri, "lora_pi": l_pi, "lora_gap": l_ri - l_pi,
        }
    return out


def make_table(rows, out_path, header_comment):
    cells_sorted = sorted(rows.keys())
    lines = [
        f"% {header_comment}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{\textbf{Base}} & \multicolumn{3}{c}{\textbf{+ LoRA}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"$(K, N)$ & FVQ & CVQ & gap & FVQ & CVQ & gap \\",
        r"\midrule",
    ]
    for K, N in cells_sorted:
        r = rows[(K, N)]
        if r["base_ri"] is None:
            base_cells = "--- & --- & ---"
        else:
            base_cells = f"{fmt_acc(r['base_ri'])} & {fmt_acc(r['base_pi'])} & {fmt_gap(r['base_gap'])}"
        lora_cells = (
            f"{fmt_acc(r['lora_ri'])} & {fmt_acc(r['lora_pi'])} & {fmt_gap(r['lora_gap'])}"
        )
        lines.append(f"  $({K}, {N})$ & {base_cells} & {lora_cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"  saved {out_path}")


def main():
    print(f"Generating Gemma LoRA tables to {OUT_DIR} …")
    arb_rows = load_gemma_arb_rows()
    sem_rows = load_gemma_sem_rows()
    make_table(
        arb_rows,
        OUT_DIR / "tab_lora_gemma_arb.tex",
        "Auto-generated by paper/scripts/generate_gemma_lora_tables.py — Gemma-3-4b-it LoRA, held-out ARB cells",
    )
    make_table(
        sem_rows,
        OUT_DIR / "tab_lora_gemma_sem.tex",
        "Auto-generated by paper/scripts/generate_gemma_lora_tables.py — Gemma-3-4b-it LoRA, SEM OOD cells",
    )
    print("Done.")


if __name__ == "__main__":
    main()
