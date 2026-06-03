"""
Extract top "query-discriminating" heads from attention_routing JSONs.

A head (layer L, head H) at source S, round_target R is "FVQ-CVQ discriminating"
if attn_FVQ[L, H, R] - attn_CVQ[L, H, R] is large in magnitude (mean over trials).

Positive cells: head attends MORE to that round under FVQ than CVQ.
Negative cells: head attends MORE under CVQ than FVQ.

We extract top-k cells by |diff| for each (source × round_target) combination
and report them in a structured way, plus aggregate findings across configs
to identify heads that show consistent discrimination.

Usage:
    .venv/bin/python v3/extract_query_heads.py --all --top-k 20
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = _SCRIPT_DIR / "results_vllm" / "attention_routing"
OUT_ROOT = _SCRIPT_DIR / "results_vllm" / "attention_routing" / "head_analysis"


def aggregate_per_head(data: dict, source: str, query: str) -> np.ndarray:
    """[n_layers, n_heads, n_rounds] mean attention across trials."""
    attn_key = f"attn_from_{'gen' if source == 'gen' else 'qcat'}"
    per_trial = []
    for t in data["trials"]:
        arr = np.asarray(t[f"expA_{query}"][attn_key], dtype=np.float32)  # [L, H, R, 2]
        per_trial.append(arr.sum(axis=-1))  # [L, H, R]
    return np.stack(per_trial).mean(axis=0)


def top_k_cells(diff_matrix: np.ndarray, k: int, sign: str = "both"):
    """Return top-k cells of a 2D matrix by |value|.

    sign: 'positive', 'negative', or 'both' for sorting by signed value.

    Returns list of (layer, head, signed_value).
    """
    flat_idx = np.argsort(-np.abs(diff_matrix), axis=None)
    out = []
    for fi in flat_idx[: k * 3]:  # over-fetch then filter
        L, H = np.unravel_index(fi, diff_matrix.shape)
        v = float(diff_matrix[L, H])
        if sign == "positive" and v <= 0:
            continue
        if sign == "negative" and v >= 0:
            continue
        out.append((int(L), int(H), v))
        if len(out) >= k:
            break
    return out


def analyse_one(path: Path, top_k: int) -> dict:
    """Extract top heads for one config. Returns dict with structured findings."""
    data = json.loads(path.read_text())
    model = data["model"]
    mode = data["mode"]
    K, N = data["K"], data["N"]

    findings = {
        "config": path.stem,
        "model": model,
        "mode": mode,
        "K": K,
        "N": N,
        "n_trials": data["n_trials_completed"],
        "discriminating_heads": {},
    }

    for source in ("gen", "qcat"):
        fvq = aggregate_per_head(data, source, "FVQ")
        cvq = aggregate_per_head(data, source, "CVQ")
        diff = fvq - cvq  # [L, H, R]

        for round_label, ridx in (("primacy_round0", 0), ("recency_last_round", -1)):
            mat = diff[:, :, ridx]  # [L, H]
            findings["discriminating_heads"][f"{source}__{round_label}"] = {
                "top_FVQ_dominant": top_k_cells(mat, top_k, sign="positive"),
                "top_CVQ_dominant": top_k_cells(mat, top_k, sign="negative"),
                "max_abs_diff": float(np.abs(mat).max()),
                "mean_abs_diff": float(np.abs(mat).mean()),
            }

    return findings


def cross_config_consistency(all_findings: list, key: str, direction: str) -> dict:
    """For a given key like 'gen__primacy_round0' and direction in {'FVQ', 'CVQ'},
    find heads in the top-k for MULTIPLE configs — robust query-discriminating heads.
    """
    bucket = "top_FVQ_dominant" if direction == "FVQ" else "top_CVQ_dominant"
    counter = defaultdict(lambda: {"appears_in": [], "values": []})
    for finding in all_findings:
        if key not in finding["discriminating_heads"]:
            continue
        for L, H, v in finding["discriminating_heads"][key][bucket]:
            counter[(L, H)]["appears_in"].append(finding["config"])
            counter[(L, H)]["values"].append(v)
    return {k: v for k, v in counter.items() if len(v["appears_in"]) >= 2}


def render_markdown(all_findings: list, top_k: int) -> str:
    """Render a human-readable markdown report."""
    lines = []
    lines.append("# Query-Discriminating Heads — Extraction Report\n")
    lines.append("FVQ - CVQ attention differences per (layer, head).  ")
    lines.append("**Positive** = head attends more to target round under FVQ than CVQ.  ")
    lines.append("**Negative** = head attends more under CVQ.  \n")

    for finding in all_findings:
        lines.append(f"## {finding['config']}")
        lines.append(f"- Model: `{finding['model']}` / mode: `{finding['mode']}`  ")
        lines.append(f"- K={finding['K']}, N={finding['N']}, trials={finding['n_trials']}\n")

        for key, info in finding["discriminating_heads"].items():
            src, tgt = key.split("__")
            src_label = "generation token" if src == "gen" else "query-cat token"
            lines.append(f"### {src_label} → {tgt}")
            lines.append(
                f"max |FVQ-CVQ| = {info['max_abs_diff']:.4f}, "
                f"mean |FVQ-CVQ| = {info['mean_abs_diff']:.5f}\n"
            )

            lines.append("**Top FVQ-dominant (FVQ > CVQ at this target):**\n")
            lines.append("| rank | layer | head | FVQ-CVQ |")
            lines.append("|---:|---:|---:|---:|")
            for r, (L, H, v) in enumerate(info["top_FVQ_dominant"][:top_k], 1):
                lines.append(f"| {r} | {L} | {H} | {v:+.4f} |")
            lines.append("")

            lines.append("**Top CVQ-dominant (CVQ > FVQ at this target):**\n")
            lines.append("| rank | layer | head | FVQ-CVQ |")
            lines.append("|---:|---:|---:|---:|")
            for r, (L, H, v) in enumerate(info["top_CVQ_dominant"][:top_k], 1):
                lines.append(f"| {r} | {L} | {H} | {v:+.4f} |")
            lines.append("")

    # Cross-config consistency for Qwen configs
    qwen_findings = [f for f in all_findings if "Qwen" in f["model"]]
    if len(qwen_findings) >= 2:
        lines.append("\n## Cross-Qwen-config consistency\n")
        lines.append(
            f"Heads in the top-{top_k} of MULTIPLE Qwen configs — "
            f"robust query-discriminating heads.\n"
        )
        for key in (
            "gen__primacy_round0",
            "gen__recency_last_round",
            "qcat__primacy_round0",
            "qcat__recency_last_round",
        ):
            for direction in ("FVQ", "CVQ"):
                consistent = cross_config_consistency(qwen_findings, key, direction)
                if not consistent:
                    continue
                lines.append(f"### {key.replace('__', ' → ')}  —  {direction}-dominant heads\n")
                lines.append("| layer | head | appears in | values |")
                lines.append("|---:|---:|---|---|")
                rows = sorted(consistent.items(),
                              key=lambda kv: -len(kv[1]["appears_in"]))
                for (L, H), info in rows[:20]:
                    where = ", ".join(c.replace("Qwen2.5-3B-Instruct__", "") for c in info["appears_in"])
                    vals = ", ".join(f"{v:+.3f}" for v in info["values"])
                    lines.append(f"| {L} | {H} | {where} | {vals} |")
                lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*", help="Result JSON paths")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out-dir", type=str, default=str(OUT_ROOT))
    args = ap.parse_args()

    paths = sorted(RESULTS_ROOT.glob("*.json")) if args.all else [Path(p) for p in (args.results or [])]
    if not paths:
        raise SystemExit("Provide --all or --results")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_findings = []
    for p in paths:
        print(f"  analysing {p.name}…")
        finding = analyse_one(p, args.top_k)
        all_findings.append(finding)
        (out_dir / f"{p.stem}__heads.json").write_text(json.dumps(finding, indent=2))

    # Markdown report
    md = render_markdown(all_findings, args.top_k)
    (out_dir / "head_analysis_report.md").write_text(md)
    print(f"\n  saved {out_dir / 'head_analysis_report.md'}")
    print(f"  saved {len(all_findings)} per-config JSONs in {out_dir}/")


if __name__ == "__main__":
    main()
