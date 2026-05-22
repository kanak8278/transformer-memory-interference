# React Slides — Reference Implementation

This folder demonstrates how a single slide is built as a React component, so the rest of the deck can follow the same pattern.

## Files

- `Slide7QwenResults.jsx` — the slide component. Drop into a real React project as-is.
- `index.html` — self-contained preview. Loads React + Babel from CDN, inlines the data, and renders the component. Open in a browser; works from `file://` (no server needed).

## Preview the slide

```bash
open v3/presentations/week2/react/index.html
```

## How it fits together

```
plots/slide7_qwen_table.json   ← canonical data (built by scripts/01_qwen_results_table.py)
                ↓
      Slide7QwenResults.jsx    ← consumes data via `data` prop
                ↓
           index.html          ← inlines data + mounts component (preview only)
```

For the real React deck:

1. Put the `.json` files in your React project (e.g. `src/data/`) or import them at build time.
2. Drop `Slide7QwenResults.jsx` into `src/slides/` (rename if you want).
3. Replace the inline `<style>` block with CSS modules / Tailwind / styled-components — your choice.
4. Compose slides into a deck via your preferred library (Spectacle, MDX, plain `react-router`, etc.).

## Conventions to keep across slides

- **One JSON per slide.** Each `scripts/0N_*.py` writes a JSON; the component consumes that JSON and only that JSON.
- **Color spec lives in JSON.** Colours come from the `color_spec` field. The React colour helpers in this file mirror the Python ones in `scripts/01_qwen_results_table.py` — keep them in sync if you change scales.
- **No data fetching in the component.** Data is passed as a prop. This keeps the component pure and makes it trivial to drop into MDX or static sites.
