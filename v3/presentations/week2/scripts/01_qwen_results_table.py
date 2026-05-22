"""
Slide 7 — Qwen family results table (HTML + JSON).

Renders a colour-density HTML table with one row per Qwen variant and three
metric columns (RI / PI / Gap), repeated for two datasets. Colours are computed
in HSL so they translate cleanly to React/Tailwind later.

Source : v3/results_vllm/summary/all_models_{arbitrary_single,semantic_multi}.json
        (full-grid means — match results_table.txt exactly)
Outputs:
  - presentations/week2/plots/slide7_qwen_table.html   (drop into browser; screenshot for slide)
  - presentations/week2/plots/slide7_qwen_table.json   (data + colour spec for React port)
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
SRC_ARBI = ROOT / "results_vllm" / "summary" / "all_models_arbitrary_single.json"
SRC_SEM  = ROOT / "results_vllm" / "summary" / "all_models_semantic_multi.json"
OUT_HTML = HERE.parents[1] / "plots" / "slide7_qwen_table.html"
OUT_JSON = HERE.parents[1] / "plots" / "slide7_qwen_table.json"

QWEN_ORDER = [
    ("Qwen2.5-0.5B-Instruct", "Qwen 0.5B Inst", "instruct", "qwen"),
    ("Qwen2.5-1.5B-Instruct", "Qwen 1.5B Inst", "instruct", "qwen"),
    ("Qwen2.5-3B-Instruct",   "Qwen 3B Inst",   "instruct", "qwen"),
    ("Qwen2.5-3B",            "Qwen 3B Base",   "base",     "qwen"),
    ("gemma-3-270m-it",       "Gemma 270M It",  "instruct", "gemma"),
    ("gemma-3-1b-it",         "Gemma 1B It",    "instruct", "gemma"),
    ("gemma-3-4b-it",         "Gemma 4B It",    "instruct", "gemma"),
]

DATASETS = [
    ("arbitrary_single", "ARBITRARY-SINGLE",  "single-token, no semantic priors", SRC_ARBI),
    ("semantic_multi",   "SEMANTIC-MULTI",    "multi-token, real categories",     SRC_SEM),
]


# ── Colour helpers ───────────────────────────────────────────────────────────
def lerp(a, b, t):
    return a + (b - a) * t


def clip01(t):
    return max(0.0, min(1.0, t))


def ri_color(v):
    """white → green. v in [0, 1]."""
    t = clip01(v)
    # white (255,255,255) → mid (165,214,167) → deep green (46,125,50)
    if t < 0.5:
        u = t / 0.5
        r = int(lerp(255, 165, u))
        g = int(lerp(255, 214, u))
        b = int(lerp(255, 167, u))
    else:
        u = (t - 0.5) / 0.5
        r = int(lerp(165, 46,  u))
        g = int(lerp(214, 125, u))
        b = int(lerp(167, 50,  u))
    return (r, g, b)


def pi_color(v):
    """deep red → light red → white. low PI = bad = dark red."""
    t = clip01(v)
    if t < 0.5:
        u = t / 0.5
        # deep red (183,28,28) → light red (255,205,210)
        r = int(lerp(183, 255, u))
        g = int(lerp(28,  205, u))
        b = int(lerp(28,  210, u))
    else:
        u = (t - 0.5) / 0.5
        # light red (255,205,210) → white (255,255,255)
        r = int(lerp(255, 255, u))
        g = int(lerp(205, 255, u))
        b = int(lerp(210, 255, u))
    return (r, g, b)


def gap_color(v):
    """diverging: blue (negative) ↔ white (0) ↔ red (positive). v in [-0.3, 0.7]."""
    if v >= 0:
        t = clip01(v / 0.70)
        # white → red
        r = int(lerp(255, 198, t))
        g = int(lerp(255, 40,  t))
        b = int(lerp(255, 40,  t))
    else:
        t = clip01(-v / 0.30)
        # white → blue
        r = int(lerp(255, 21,  t))
        g = int(lerp(255, 101, t))
        b = int(lerp(255, 192, t))
    return (r, g, b)


def luminance(rgb):
    r, g, b = (c / 255.0 for c in rgb)
    return 0.299 * r + 0.587 * g + 0.114 * b


def text_color(rgb):
    return "#FFFFFF" if luminance(rgb) < 0.55 else "#1A1A1A"


# ── Load data ────────────────────────────────────────────────────────────────
ds_data = {}
for ds_key, _, _, src in DATASETS:
    with open(src) as f:
        ds_data[ds_key] = json.load(f)

records = {}
for model_key, label, family, family_group in QWEN_ORDER:
    records[label] = {"family": family, "family_group": family_group}
    for ds_key, _, _, _ in DATASETS:
        d = ds_data[ds_key].get(model_key, {})
        records[label][ds_key] = {
            "ri":  float(d.get("mean_ri",  float("nan"))),
            "pi":  float(d.get("mean_pi",  float("nan"))),
            "gap": float(d.get("mean_gap", float("nan"))),
            "n_cells": int(d.get("n_cells", 0)),
        }


# ── Render HTML ──────────────────────────────────────────────────────────────
def fmt(v, kind):
    if v != v:  # NaN
        return "—"
    if kind == "gap":
        return f"{v:+.1%}"
    return f"{v:.0%}"


def cell(value, kind):
    if kind == "ri":
        rgb = ri_color(value)
    elif kind == "pi":
        rgb = pi_color(value)
    else:
        rgb = gap_color(value)
    bg = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
    fg = text_color(rgb)
    return f'<td class="cell" style="background:{bg}; color:{fg}">{fmt(value, kind)}</td>'


# Build the table
def build_dataset_block(ds_key, ds_title, ds_subtitle):
    rows = []
    last_group = None
    for _, label, family, group in QWEN_ORDER:
        r = records[label][ds_key]
        family_class = "row-base" if family == "base" else "row-instruct"
        # Insert a thin family-divider row between Qwen and Gemma blocks
        if last_group is not None and group != last_group:
            rows.append('<tr class="family-divider"><td colspan="5"></td></tr>')
        last_group = group
        rows.append(
            f'<tr class="{family_class} fam-{group}">'
            f'<td class="model">{label}</td>'
            f'{cell(r["ri"], "ri")}'
            f'{cell(r["pi"], "pi")}'
            f'{cell(r["gap"], "gap")}'
            f'<td class="ncells">n={r["n_cells"]}</td>'
            f"</tr>"
        )
    return f"""
    <div class="dataset-block">
      <div class="dataset-title">{ds_title}</div>
      <div class="dataset-sub">{ds_subtitle}</div>
      <table>
        <thead>
          <tr>
            <th class="model"></th>
            <th>RI<br><span class="hint">recall first</span></th>
            <th>PI<br><span class="hint">recall last</span></th>
            <th>Gap<br><span class="hint">RI − PI</span></th>
            <th class="ncells">cells</th>
          </tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </div>
    """


html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Slide 7 — Qwen Family Results</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
    background: #FFFFFF;
    color: #1A1A1A;
    margin: 0;
    padding: 48px;
  }}
  .container {{
    max-width: 1280px;
    margin: 0 auto;
  }}
  h1 {{
    font-size: 22px;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: -0.01em;
  }}
  .subtitle {{
    color: #666;
    font-size: 13px;
    margin-bottom: 28px;
  }}
  .grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 36px;
  }}
  .dataset-block {{ }}
  .dataset-title {{
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #2C2C2C;
    margin-bottom: 2px;
  }}
  .dataset-sub {{
    font-size: 11.5px;
    color: #888;
    font-style: italic;
    margin-bottom: 12px;
  }}
  table {{
    border-collapse: separate;
    border-spacing: 3px;
    width: 100%;
  }}
  th {{
    font-size: 11px;
    font-weight: 600;
    color: #555;
    padding: 5px 4px;
    text-align: center;
    background: #F4F4F4;
    border-radius: 3px;
  }}
  th.model {{ background: transparent; text-align: left; }}
  th.ncells, td.ncells {{
    background: transparent !important;
    color: #AAA;
    font-size: 10.5px;
    text-align: center;
    font-weight: 400;
    padding: 4px 4px;
  }}
  th .hint {{
    display: block;
    font-size: 9.5px;
    color: #888;
    font-weight: 400;
    margin-top: 1px;
  }}
  td {{
    padding: 7px 6px;
    text-align: center;
    font-variant-numeric: tabular-nums;
    font-weight: 600;
    font-size: 13.5px;
    border-radius: 3px;
  }}
  td.model {{
    text-align: left;
    background: transparent;
    color: #1A1A1A;
    font-weight: 600;
    font-size: 12.5px;
    padding-left: 4px;
  }}
  tr.row-base td.model {{ font-style: italic; color: #C2185B; }}
  tr.row-base td.model::after {{
    content: " (no SFT)";
    font-size: 10.5px;
    color: #999;
    font-weight: 400;
    font-style: normal;
  }}
  tr.family-divider td {{ background: transparent !important; padding: 0 !important; height: 12px; border-top: 1px dashed #DDD; }}
  .legend {{
    display: flex;
    gap: 24px;
    margin-top: 28px;
    font-size: 11.5px;
    color: #555;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 8px; }}
  .swatch {{
    width: 18px; height: 14px; border-radius: 3px; display: inline-block;
  }}
  .footer {{
    margin-top: 24px;
    font-size: 11px;
    color: #999;
    font-style: italic;
  }}
</style>
</head>
<body>
  <div class="container">
    <h1>Qwen 2.5 Family — Position-Dependent Retrieval</h1>
    <div class="subtitle">
      Full-grid means · 100 trials/cell · Wilson 95% CIs ≤ ±10% per cell
    </div>
    <div class="grid">
      {build_dataset_block("arbitrary_single", "ARBITRARY-SINGLE", "single-token English words, no semantic priors")}
      {build_dataset_block("semantic_multi",   "SEMANTIC-MULTI",   "multi-token category-anchored values")}
    </div>
    <div class="legend">
      <div class="legend-item"><span class="swatch" style="background:#2E7D32"></span>RI high (good)</div>
      <div class="legend-item"><span class="swatch" style="background:#B71C1C"></span>PI low (bad)</div>
      <div class="legend-item"><span class="swatch" style="background:#C62828"></span>Gap &gt; 0 — primacy bias (RI &gt; PI)</div>
      <div class="legend-item"><span class="swatch" style="background:#1565C0"></span>Gap &lt; 0 — recency wins (PI &gt; RI)</div>
    </div>
    <div class="footer">
      RI = recall first value · PI = recall last value · Gap = RI − PI · n_cells = number of (K, N) cells averaged
    </div>
  </div>
</body>
</html>
"""

OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_HTML, "w") as f:
    f.write(html)
print(f"Saved → {OUT_HTML}")

# ── Dump JSON ────────────────────────────────────────────────────────────────
json_out = {
    "title": "Qwen 2.5 Family — Position-Dependent Retrieval",
    "subtitle": "Full-grid means · 100 trials/cell",
    "datasets": [
        {"key": k, "title": t, "subtitle": s}
        for k, t, s, _ in DATASETS
    ],
    "models": [
        {"key": k, "label": l, "family": f, "family_group": g}
        for k, l, f, g in QWEN_ORDER
    ],
    "data": records,
    "color_spec": {
        "ri":  {"type": "sequential", "low": "#FFFFFF", "mid": "#A5D6A7",
                "high": "#2E7D32",  "vmin": 0.0,  "vmax": 1.0},
        "pi":  {"type": "sequential", "low": "#B71C1C", "mid": "#FFCDD2",
                "high": "#FFFFFF",  "vmin": 0.0,  "vmax": 1.0},
        "gap": {"type": "diverging",  "low": "#1565C0", "mid": "#FFFFFF",
                "high": "#C62828",  "vcenter": 0.0,
                "vmin": -0.30, "vmax": 0.70},
    },
}
with open(OUT_JSON, "w") as f:
    json.dump(json_out, f, indent=2)
print(f"Saved → {OUT_JSON}")
