/**
 * Slide7QwenResults — React component for the Qwen-family results table.
 *
 * Drop into a React project. Pass `data` as the contents of
 *   v3/presentations/week2/plots/slide7_qwen_table.json
 *
 * Example:
 *   import data from "./slide7_qwen_table.json";
 *   <Slide7QwenResults data={data} />
 *
 * Styling is co-located via a <style> tag scoped by id selector. In a real
 * React app you'd move this to CSS modules, Tailwind, or styled-components.
 */

// ── Colour helpers ───────────────────────────────────────────────────────────
const lerp = (a, b, t) => a + (b - a) * t;
const clip01 = t => Math.max(0, Math.min(1, t));

function riColor(v) {
  const t = clip01(v);
  let r, g, b;
  if (t < 0.5) {
    const u = t / 0.5;
    r = lerp(255, 165, u); g = lerp(255, 214, u); b = lerp(255, 167, u);
  } else {
    const u = (t - 0.5) / 0.5;
    r = lerp(165, 46, u); g = lerp(214, 125, u); b = lerp(167, 50, u);
  }
  return [r, g, b];
}

function piColor(v) {
  const t = clip01(v);
  let r, g, b;
  if (t < 0.5) {
    const u = t / 0.5;
    r = lerp(183, 255, u); g = lerp(28, 205, u); b = lerp(28, 210, u);
  } else {
    const u = (t - 0.5) / 0.5;
    r = lerp(255, 255, u); g = lerp(205, 255, u); b = lerp(210, 255, u);
  }
  return [r, g, b];
}

function gapColor(v) {
  let r, g, b;
  if (v >= 0) {
    const t = clip01(v / 0.7);
    r = lerp(255, 198, t); g = lerp(255, 40, t); b = lerp(255, 40, t);
  } else {
    const t = clip01(-v / 0.3);
    r = lerp(255, 21, t); g = lerp(255, 101, t); b = lerp(255, 192, t);
  }
  return [r, g, b];
}

const luminance = ([r, g, b]) => (0.299 * r + 0.587 * g + 0.114 * b) / 255;
const textColor = rgb => (luminance(rgb) < 0.55 ? "#FFFFFF" : "#1A1A1A");
const rgbStr = ([r, g, b]) => `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;

const COLORERS = { ri: riColor, pi: piColor, gap: gapColor };

function fmt(value, kind) {
  if (Number.isNaN(value) || value == null) return "—";
  if (kind === "gap") return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)}%`;
  return `${Math.round(value * 100)}%`;
}

// ── Sub-components ───────────────────────────────────────────────────────────
function Cell({ value, kind }) {
  const rgb = COLORERS[kind](value);
  return (
    <td className="cell" style={{ background: rgbStr(rgb), color: textColor(rgb) }}>
      {fmt(value, kind)}
    </td>
  );
}

function DatasetBlock({ datasetKey, title, subtitle, models, data }) {
  const rows = [];
  let lastGroup = null;
  models.forEach(m => {
    const r = data[m.label][datasetKey];
    const rowClass = m.family === "base" ? "row-base" : "row-instruct";
    if (lastGroup !== null && m.family_group !== lastGroup) {
      rows.push(
        <tr className="family-divider" key={`div-${m.key}`}>
          <td colSpan="5" />
        </tr>
      );
    }
    lastGroup = m.family_group;
    rows.push(
      <tr key={m.key} className={`${rowClass} fam-${m.family_group}`}>
        <td className="model">{m.label}</td>
        <Cell value={r.ri} kind="ri" />
        <Cell value={r.pi} kind="pi" />
        <Cell value={r.gap} kind="gap" />
        <td className="ncells">n={r.n_cells}</td>
      </tr>
    );
  });

  return (
    <div className="dataset-block">
      <div className="dataset-title">{title}</div>
      <div className="dataset-sub">{subtitle}</div>
      <table>
        <thead>
          <tr>
            <th className="model" />
            <th>RI<br /><span className="hint">recall first</span></th>
            <th>PI<br /><span className="hint">recall last</span></th>
            <th>Gap<br /><span className="hint">RI − PI</span></th>
            <th className="ncells">cells</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  );
}

function Legend() {
  const items = [
    { swatch: "#2E7D32", label: "RI high (good)" },
    { swatch: "#B71C1C", label: "PI low (bad)" },
    { swatch: "#C62828", label: "Gap > 0 — primacy bias (RI > PI)" },
    { swatch: "#1565C0", label: "Gap < 0 — recency wins (PI > RI)" },
  ];
  return (
    <div className="legend">
      {items.map(({ swatch, label }) => (
        <div className="legend-item" key={label}>
          <span className="swatch" style={{ background: swatch }} />
          {label}
        </div>
      ))}
    </div>
  );
}

// ── Main component ───────────────────────────────────────────────────────────
export default function Slide7QwenResults({ data }) {
  return (
    <div id="slide7-qwen-results" className="slide">
      <style>{styles}</style>
      <div className="container">
        <h1>{data.title}</h1>
        <div className="subtitle">{data.subtitle}</div>
        <div className="grid">
          {data.datasets.map(ds => (
            <DatasetBlock
              key={ds.key}
              datasetKey={ds.key}
              title={ds.title}
              subtitle={ds.subtitle}
              models={data.models}
              data={data.data}
            />
          ))}
        </div>
        <Legend />
        <div className="footer">
          RI = recall first value · PI = recall last value · Gap = RI − PI · n_cells = (K, N) cells averaged
        </div>
      </div>
    </div>
  );
}

// ── Scoped styles ────────────────────────────────────────────────────────────
const styles = `
#slide7-qwen-results { background: #FFFFFF; color: #1A1A1A; padding: 48px; font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif; }
#slide7-qwen-results * { box-sizing: border-box; }
#slide7-qwen-results .container { max-width: 1280px; margin: 0 auto; }
#slide7-qwen-results h1 { font-size: 22px; font-weight: 700; margin: 0 0 6px 0; letter-spacing: -0.01em; }
#slide7-qwen-results .subtitle { color: #666; font-size: 13px; margin-bottom: 28px; }
#slide7-qwen-results .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 36px; }
#slide7-qwen-results .dataset-title { font-size: 13px; font-weight: 700; letter-spacing: 0.04em; color: #2C2C2C; margin-bottom: 2px; }
#slide7-qwen-results .dataset-sub { font-size: 11.5px; color: #888; font-style: italic; margin-bottom: 12px; }
#slide7-qwen-results table { border-collapse: separate; border-spacing: 3px; width: 100%; }
#slide7-qwen-results th { font-size: 11px; font-weight: 600; color: #555; padding: 5px 4px; text-align: center; background: #F4F4F4; border-radius: 3px; }
#slide7-qwen-results th.model { background: transparent; text-align: left; }
#slide7-qwen-results th.ncells, #slide7-qwen-results td.ncells { background: transparent !important; color: #AAA; font-size: 10.5px; text-align: center; font-weight: 400; padding: 4px 4px; }
#slide7-qwen-results th .hint { display: block; font-size: 9.5px; color: #888; font-weight: 400; margin-top: 1px; }
#slide7-qwen-results td { padding: 7px 6px; text-align: center; font-variant-numeric: tabular-nums; font-weight: 600; font-size: 13.5px; border-radius: 3px; }
#slide7-qwen-results td.model { text-align: left; background: transparent; color: #1A1A1A; font-weight: 600; font-size: 12.5px; padding-left: 4px; }
#slide7-qwen-results tr.row-base td.model { color: #C2185B; }
#slide7-qwen-results tr.row-base td.model::after { content: " (no SFT)"; font-size: 10.5px; color: #999; font-weight: 400; font-style: italic; }
#slide7-qwen-results tr.family-divider td { background: transparent !important; padding: 0 !important; height: 8px; border-top: 1px dashed #DDD; }
#slide7-qwen-results .legend { display: flex; gap: 24px; margin-top: 28px; font-size: 11.5px; color: #555; flex-wrap: wrap; }
#slide7-qwen-results .legend-item { display: flex; align-items: center; gap: 8px; }
#slide7-qwen-results .swatch { width: 18px; height: 14px; border-radius: 3px; display: inline-block; }
#slide7-qwen-results .footer { margin-top: 24px; font-size: 11px; color: #999; font-style: italic; }
`;
