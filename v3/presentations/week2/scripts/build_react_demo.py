"""
Generate a self-contained HTML page that renders Slide7QwenResults via React
+ Babel (loaded from CDN). Inlines the JSON data so the page works from
file:// without a local server.

Outputs:
  - presentations/week2/react/index.html
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve()
JSON_SRC  = HERE.parents[1] / "plots" / "slide7_qwen_table.json"
JSX_SRC   = HERE.parents[1] / "react" / "Slide7QwenResults.jsx"
OUT_HTML  = HERE.parents[1] / "react" / "index.html"

with open(JSON_SRC) as f:
    data = json.load(f)
with open(JSX_SRC) as f:
    jsx = f.read()

# Strip ESM-only syntax for the in-browser Babel build:
#   - `export default ...` → `const Slide7QwenResults = ...`
jsx = jsx.replace("export default function Slide7QwenResults",
                  "function Slide7QwenResults")

inlined_json = json.dumps(data, indent=2)

html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Slide 7 — Qwen Family Results (React Demo)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script crossorigin src="https://unpkg.com/react@18/umd/react.development.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <style>
    body {{ margin: 0; background: #F7F7F7; }}
    .deck-frame {{
      max-width: 1400px; margin: 24px auto;
      background: #FFFFFF; box-shadow: 0 6px 24px rgba(0,0,0,0.06);
      border-radius: 8px;
    }}
    .slide-header {{
      padding: 12px 24px; background: #FAFAFA;
      border-bottom: 1px solid #EEE;
      font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
      font-size: 12px; color: #777;
      display: flex; justify-content: space-between; align-items: center;
    }}
    .slide-header strong {{ color: #333; font-weight: 600; }}
  </style>
</head>
<body>
  <div class="deck-frame">
    <div class="slide-header">
      <span><strong>Slide 7</strong> · Qwen Family Results</span>
      <span>Week 2 Update · React demo</span>
    </div>
    <div id="root"></div>
  </div>

  <script type="text/babel" data-presets="react">
    const SLIDE_DATA = {inlined_json};

    {jsx}

    ReactDOM.createRoot(document.getElementById("root"))
      .render(<Slide7QwenResults data={{SLIDE_DATA}} />);
  </script>
</body>
</html>
"""

OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_HTML, "w") as f:
    f.write(html)
print(f"Saved → {OUT_HTML}")
print()
print("To preview: open the file in a browser:")
print(f"  open {OUT_HTML}")
