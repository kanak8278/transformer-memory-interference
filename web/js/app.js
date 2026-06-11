/* ───────────────────────────────────────────────────────────────────────────
   Transformer Memory Interference — interactive showcase
   Vanilla JS + hand-rolled SVG. All data precomputed in web/data/*.json.
   ─────────────────────────────────────────────────────────────────────────── */

const SVGNS = "http://www.w3.org/2000/svg";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const pct = (x) => Math.round(x * 100);

const COL = {
  good: getCSS("--good"), bad: getCSS("--bad"), accent: getCSS("--accent"),
  lora: getCSS("--lora"), muted: getCSS("--muted"), faint: getCSS("--faint"),
  border: getCSS("--border"), borderStrong: getCSS("--border-strong"), ink: getCSS("--ink"),
};
function getCSS(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }

function el(tag, attrs = {}, kids = []) {
  const ns = ["svg", "g", "path", "line", "circle", "rect", "text", "polyline", "tspan"].includes(tag);
  const n = ns ? document.createElementNS(SVGNS, tag) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.setAttribute("class", v);
    else if (k === "text") n.textContent = v;
    else if (k === "html") n.innerHTML = v;
    else n.setAttribute(k, v);
  }
  (Array.isArray(kids) ? kids : [kids]).forEach(c => c && n.appendChild(c));
  return n;
}
const J = (p) => fetch(p).then(r => r.json());

// shared tooltip
const tip = $("#tip");
function showTip(html, x, y) {
  tip.innerHTML = html;
  tip.style.opacity = 1;
  tip.style.left = clamp(x + 14, 8, innerWidth - 250) + "px";
  tip.style.top = (y - 10) + "px";
}
function hideTip() { tip.style.opacity = 0; }

// ════════════════════════════════════════════════════════════════ boot
Promise.all([
  J("data/examples.json"), J("data/models_scatter.json"),
  J("data/logit_lens.json"), J("data/probing.json"), J("data/intervention.json"),
  J("data/position_curve.json"), J("data/load.json"),
  J("data/training_dynamics.json"), J("data/format.json"),
]).then(([examples, scatter, ll, probing, fix, pos, load, td, fmt]) => {
  buildHero(scatter, fix);
  buildDemo(examples);
  buildScatter(scatter);
  buildScatterTable(scatter);
  buildPositionCurve(pos);
  buildLoad(load);
  buildTraining(td);
  buildFormat(fmt);
  buildLogitLens(ll);
  buildProbing(probing);
  buildFix(fix);
  setupReveal();
  addEventListener("resize", debounce(() => {
    buildScatter(scatter); buildPositionCurve(pos, true); buildLoad(load, true);
    buildTraining(td); buildFormat(fmt, true);
    buildLogitLens(ll, true); buildProbing(probing, true);
  }, 180));
}).catch(e => console.error("data load failed", e));

function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

// ════════════════════════════════════════════════════════════════ hero
function buildHero(scatter, fix) {
  const h = scatter.headline;
  const stats = [
    [h.n_models_paper, "models tested"],
    ["+" + h.gap_range_pp[0] + " to +" + h.gap_range_pp[1] + " pp", "first − last gap range"],
    [h.n_gap_positive + "/" + h.n_shown, "land above the line"],
    ["→ ~0 pp", "gap, after a fine-tune"],
  ];
  const wrap = $("#hero-stats");
  stats.forEach(([n, l]) => wrap.appendChild(
    el("div", { class: "stat" }, [el("div", { class: "num", text: n }), el("div", { class: "lbl", text: l })])
  ));
}

// ════════════════════════════════════════════════════════════════ §1 demo
function buildDemo(data) {
  const scen = data.scenarios;
  const nVals = data.n_values;
  let si = 0;                                  // scenario index
  let q = "first";                             // first | last
  let demoExpanded = false;                    // stream middle collapsed by default
  let ni = nVals.indexOf(data.default_n || nVals[Math.floor(nVals.length / 2)]);
  if (ni < 0) ni = nVals.length - 1;

  const slider = $("#demo-nslider");
  slider.min = 0; slider.max = nVals.length - 1; slider.step = 1;

  function cur() { return scen[si].by_n[String(nVals[ni])]; }

  function render() {
    const s = scen[si], b = cur();
    const N = nVals[ni];
    const qd = q === "first" ? b.fvq : b.cvq;
    const targetFirst = q === "first";

    // header
    $("#demo-target").innerHTML =
      `Tracking <span class="tk">${s.target_key}</span> ` +
      `<span class="dk">+ ${s.keys.filter(k => k !== s.target_key).join(", ")}</span>`;
    $("#demo-streamlen").textContent = `· ${b.stream.length} updates (${s.keys.length} keys × ${N})`;

    // stream list (with collapsible middle)
    const list = $("#demo-stream"); list.replaceChildren();
    const stream = b.stream, len = stream.length, lastT = b.n_target - 1;
    const HEAD = 4, TAIL = 4;
    const predIdx = (!qd.correct && qd.predicted_tpos != null && qd.predicted_tpos >= 0)
      ? stream.findIndex(it => it.target && it.tpos === qd.predicted_tpos) : -1;
    // indices that must always be visible: head, tail, target-first, target-last, model pick
    const must = new Set();
    for (let i = 0; i < HEAD; i++) must.add(i);
    for (let i = Math.max(0, len - TAIL); i < len; i++) must.add(i);
    must.add(stream.findIndex(it => it.target && it.tpos === 0));
    must.add(stream.findIndex(it => it.target && it.tpos === lastT));
    if (predIdx >= 0) { must.add(predIdx); if (predIdx > 0) must.add(predIdx - 1); }

    function makeRow(it) {
      let cls = "srow " + (it.target ? "tgt" : "distractor");
      let tag = "";
      if (it.target && it.tpos === 0) { cls += " is-first"; if (targetFirst) tag = "first"; }
      if (it.target && it.tpos === lastT) { cls += " is-last"; if (!targetFirst) tag = "last · current"; }
      if (it.target && it.tpos === qd.predicted_tpos && !qd.correct) cls += " is-pred";
      const row = el("div", { class: cls }, [
        el("span", { class: "sk", text: it.key }),
        el("span", { class: "sv", text: it.value }),
      ]);
      if (tag) row.appendChild(el("span", { class: "tag", text: tag }));
      else if (cls.includes("is-pred")) row.appendChild(el("span", { class: "tag pred", text: "model picked" }));
      else row.appendChild(el("span", {}));
      return row;
    }

    if (demoExpanded) {
      const bar = el("div", { class: "collapse-row" }, [
        el("span", { text: "Collapse the middle" }), el("span", { class: "chev", text: "⌃" })]);
      bar.onclick = () => { demoExpanded = false; render(); };
      list.appendChild(bar);
      stream.forEach(it => list.appendChild(makeRow(it)));
    } else {
      let i = 0;
      while (i < len) {
        if (must.has(i)) { list.appendChild(makeRow(stream[i])); i++; continue; }
        // hidden run
        let j = i; while (j < len && !must.has(j)) j++;
        const count = j - i;
        const bar = el("div", { class: "collapse-row" }, [
          el("span", { class: "dots", text: "•••" }),
          el("span", { text: `${count} more update${count > 1 ? "s" : ""} hidden` }),
          el("span", { class: "chev", text: "⌄" })]);
        bar.onclick = () => { demoExpanded = true; render(); };
        list.appendChild(bar);
        i = j;
      }
    }

    // memory-now panel
    const mem = $("#demo-memory"); mem.replaceChildren();
    s.keys.forEach(k => {
      mem.appendChild(el("div", { class: "mem-row" + (k === s.target_key ? " tgt" : "") }, [
        el("span", { class: "mk", text: k }), el("span", { class: "mv", text: b.memory_now[k] }),
      ]));
    });

    // answer
    const qword = targetFirst ? '<span style="color:var(--good)">first</span>'
                              : '<span style="color:var(--bad)">last</span>';
    $("#demo-answer").replaceChildren(
      el("div", { class: "qline", html: `<b>Q:</b> What was the ${qword} value of <b>${s.target_key}</b>?` }),
      el("div", { class: "answer-grid" }, [
        el("div", { class: "answer-box" }, [el("div", { class: "k", text: "correct" }),
          el("div", { class: "v", text: qd.expected })]),
        el("div", { class: "answer-box" }, [el("div", { class: "k", text: "model said" }),
          el("div", { class: "v", text: qd.predicted, style: `color:${qd.correct ? COL.good : COL.bad}` })]),
      ]),
      el("div", { class: "verdict " + (qd.correct ? "ok" : "bad"),
        text: qd.correct ? "✓ correct" : "✗ wrong" }),
      el("div", { class: "verdict-note", text: qd.note }),
    );

    $("#demo-nval").textContent = `N = ${N}`;
    slider.value = ni;
  }

  $$("#q-toggle button").forEach(btn => btn.onclick = () => {
    $$("#q-toggle button").forEach(x => x.classList.remove("active"));
    btn.classList.add("active"); q = btn.dataset.q; render();
  });
  slider.oninput = () => { ni = +slider.value; demoExpanded = false; render(); };
  $("#demo-next").onclick = () => { si = (si + 1) % scen.length; demoExpanded = false; render(); };
  render();
}

// ════════════════════════════════════════════════════════════════ §2 scatter
const FAM_COL = { "Qwen2.5": "#5b5bd6", "Qwen3.5": "#7c3aed", "Gemma-3": "#0d9488",
  "GPT": "#10a37f", "Claude": "#d97757", "Gemini": "#4285f4", "TinyLlama": "#e0484d",
  "StableLM": "#f59e0b", "Pythia": "#8b5cf6", "Mamba": "#64748b" };
function famColor(f) { return FAM_COL[f] || COL.muted; }

function buildScatter(data) {
  const host = $("#scatter"); host.replaceChildren();
  const W = 880, H = 560, m = { t: 24, r: 24, b: 56, l: 60 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const x = v => m.l + v * iw, y = v => m.t + (1 - v) * ih;

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", style: "max-height:600px" });

  // gridlines + axes
  for (let i = 0; i <= 5; i++) {
    const t = i / 5;
    svg.appendChild(el("line", { class: "gridline", x1: x(t), y1: m.t, x2: x(t), y2: m.t + ih }));
    svg.appendChild(el("line", { class: "gridline", x1: m.l, y1: y(t), x2: m.l + iw, y2: y(t) }));
    svg.appendChild(el("text", { x: x(t), y: m.t + ih + 22, "text-anchor": "middle",
      "font-size": 12, fill: COL.muted, text: pct(t) + "%" }));
    svg.appendChild(el("text", { x: m.l - 12, y: y(t) + 4, "text-anchor": "end",
      "font-size": 12, fill: COL.muted, text: pct(t) + "%" }));
  }
  // diagonal (equal)
  svg.appendChild(el("line", { x1: x(0), y1: y(0), x2: x(1), y2: y(1),
    stroke: COL.borderStrong, "stroke-width": 1.5, "stroke-dasharray": "5 4" }));
  svg.appendChild(el("text", { x: x(0.78), y: y(0.82), "font-size": 12, fill: COL.faint,
    transform: `rotate(-32 ${x(0.78)} ${y(0.82)})`, text: "equally good at both" }));

  // axis labels
  svg.appendChild(el("text", { x: m.l + iw / 2, y: H - 8, "text-anchor": "middle",
    "font-size": 13, "font-weight": 600, fill: COL.ink, text: "CVQ accuracy — recalls the LAST value →" }));
  svg.appendChild(el("text", { x: -(m.t + ih / 2), y: 16, "text-anchor": "middle",
    transform: "rotate(-90)", "font-size": 13, "font-weight": 600, fill: COL.ink,
    text: "FVQ accuracy — recalls the FIRST value →" }));

  // shaded "below line" region label
  svg.appendChild(el("text", { x: x(0.06), y: y(0.96), "font-size": 12, fill: COL.good,
    "font-weight": 600, text: "↑ better at first than last" }));

  // points
  data.points.forEach(p => {
    const c = el("circle", {
      cx: x(p.cvq), cy: y(p.fvq), r: p.proprietary ? 7 : 6,
      fill: famColor(p.family), "fill-opacity": p.proprietary ? .95 : .8,
      stroke: "#fff", "stroke-width": 1.5, style: "cursor:pointer;transition:r .1s",
    });
    c.onmouseenter = (e) => {
      c.setAttribute("r", 9);
      showTip(`<div class="t-title">${p.model}</div>
        <div class="t-row">first (FVQ): <b style="color:#fff">${pct(p.fvq)}%</b></div>
        <div class="t-row">last (CVQ): <b style="color:#fff">${pct(p.cvq)}%</b></div>
        <div class="t-row">gap: <b style="color:#fff">${p.gap > 0 ? "+" : ""}${pct(p.gap)}%</b> · K=${p.cell.K},N=${p.cell.N}</div>`,
        e.clientX, e.clientY);
    };
    c.onmousemove = (e) => showTip(tip.innerHTML, e.clientX, e.clientY);
    c.onmouseleave = () => { c.setAttribute("r", p.proprietary ? 7 : 6); hideTip(); };
    svg.appendChild(c);
  });

  host.appendChild(svg);

  // legend (families present)
  const fams = [...new Set(data.points.map(p => p.family).filter(Boolean))];
  const lg = $("#scatter-legend"); lg.replaceChildren();
  fams.forEach(f => lg.appendChild(el("span", {}, [
    el("span", { class: "dot", style: `background:${famColor(f)}` }), document.createTextNode(f),
  ])));
  $("#scatter-note").textContent = data.note;
}

// ════════════════════════════════════════════════════════════════ §3 position curve
let posState = { v: "base", cell: null };
function buildPositionCurve(data, keepState = false) {
  const sel = $("#pos-cell");
  if (!keepState) {
    sel.replaceChildren(...data.cells.map(c => el("option", { value: c.key, text: `K=${c.K}, N=${c.N}` })));
    posState.cell = (data.default_cell && data.cells.find(c => c.key === data.default_cell))
      ? data.default_cell : data.cells[0].key;
  }
  sel.value = posState.cell;

  function draw() {
    const c = data.cells.find(x => x.key === posState.cell) || data.cells[0];
    const v = posState.v;
    const series = [
      { data: c.base, color: COL.muted, width: v === "base" ? 3 : 1.5, opacity: v === "base" ? 1 : .3, name: "base" },
      { data: c.lora, color: COL.lora, width: v === "lora" ? 3 : 1.5, opacity: v === "lora" ? 1 : .3, name: "lora" },
    ];
    lineChart($("#poscurve"), {
      series, nLayers: c.positions.length, yMax: 1,
      yLabel: "retrieval accuracy", xLabel: "position queried (the k-th value of the key)",
      tickLabel: i => c.positions[i],
    });
    $("#pos-caption").textContent = `${data.model} · ${data.dataset} · K=${c.K}, N=${c.N}`;
    $("#pos-legend").replaceChildren(
      legItem(COL.muted, "Baseline"), legItem(COL.lora, "+ LoRA"),
      el("span", { class: "muted", text: `${data.trials_per_position} trials/position` }),
    );
    const b = c.base, p1 = b[0], lastv = b[b.length - 1];
    const midVals = b.slice(1, -1), mid = midVals.length ? Math.min(...midVals) : 0;
    const loraMin = Math.min(...c.lora);
    $("#pos-insight").innerHTML = `Baseline: the <b>first</b> value is recalled <b style="color:var(--good)">${pct(p1)}%</b> of the time, the middle positions collapse to as low as <b style="color:var(--bad)">${pct(mid)}%</b>, with a faint recency bump at the end (${pct(lastv)}%). <span style="color:var(--lora);font-weight:600">+LoRA</span> lifts the whole curve (lowest point ${pct(loraMin)}%) — the values were always written into the stream; the baseline just couldn't read most of them back.`;
  }

  $$("#pos-version button").forEach(b => b.onclick = () => {
    $$("#pos-version button").forEach(x => x.classList.remove("active"));
    b.classList.add("active"); posState.v = b.dataset.v; draw();
  });
  sel.onchange = () => { posState.cell = sel.value; draw(); };
  draw();
}

// ════════════════════════════════════════════════════════════════ §4 load (CVQ vs N)
let loadMi = 0;
function buildLoad(data, keepState = false) {
  const tog = $("#load-model");
  if (!keepState) {
    tog.replaceChildren(...data.models.map((m, i) => el("button", { "data-i": i, class: i === 0 ? "active" : "", text: m.model })));
    loadMi = Math.max(0, data.models.findIndex(m => m.model === data.default_model));
    [...tog.children].forEach((b, i) => b.classList.toggle("active", i === loadMi));
  }
  const Ns = data.N_values;

  function draw() {
    const m = data.models[loadMi];
    const series = [
      { data: m.fvq, color: COL.good, width: 2.75, name: "FVQ" },
      { data: m.cvq, color: COL.bad, width: 2.75, name: "CVQ" },
    ];
    lineChart($("#loadchart"), {
      series, nLayers: Ns.length, yMax: 1,
      yLabel: "accuracy", xLabel: "updates per key (N)", tickLabel: i => Ns[i],
    });
    $("#load-legend").replaceChildren(
      legItem(COL.good, "First value (FVQ)"), legItem(COL.bad, "Current value (CVQ)"),
      el("span", { class: "muted", text: `${m.model} · K=${data.K}` }),
    );
    const f = m.fvq.filter(x => x != null), cv = m.cvq.filter(x => x != null);
    const fLo = Math.min(...f), cFirst = cv[0], cLast = cv[cv.length - 1];
    $("#load-insight").innerHTML = `As updates grow from N=${Ns[0]} to N=${Ns[Ns.length - 1]}, <b style="color:var(--good)">first</b>-value recall stays high (≥${pct(fLo)}%) while <b style="color:var(--bad)">current</b>-value recall slides from ${pct(cFirst)}% to <b style="color:var(--bad)">${pct(cLast)}%</b>. Same keys, same question format — only the amount written after the target changes.`;
  }

  $$("#load-model button").forEach(b => b.onclick = () => {
    $$("#load-model button").forEach(x => x.classList.remove("active"));
    b.classList.add("active"); loadMi = +b.dataset.i; draw();
  });
  draw();
}

// ════════════════════════════════════════════════════════════════ grouped bar chart
function groupedBar(host, { groups, yMax = 1, yLabel, barColors }) {
  host.replaceChildren();
  const W = 880, H = 380, m = { t: 18, r: 16, b: 52, l: 48 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const y = v => m.t + (1 - v / yMax) * ih;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", style: "max-height:400px" });
  for (let i = 0; i <= 4; i++) {
    const t = (yMax * i) / 4;
    svg.appendChild(el("line", { class: "gridline", x1: m.l, y1: y(t), x2: m.l + iw, y2: y(t) }));
    svg.appendChild(el("text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end", "font-size": 11, fill: COL.muted, text: pct(t / yMax) + "%" }));
  }
  const gw = iw / groups.length;
  groups.forEach((g, gi) => {
    const n = g.bars.length, pad = gw * 0.18, bw = (gw - 2 * pad) / n;
    g.bars.forEach((b, bi) => {
      const x = m.l + gi * gw + pad + bi * bw;
      svg.appendChild(el("rect", { x: x + 2, y: y(b.v), width: bw - 4, height: Math.max(0, ih - (y(b.v) - m.t)),
        rx: 3, fill: b.color, "fill-opacity": b.opacity == null ? 1 : b.opacity }));
      svg.appendChild(el("text", { x: x + bw / 2, y: y(b.v) - 5, "text-anchor": "middle", "font-size": 10,
        fill: COL.muted, text: pct(b.v) + "%" }));
    });
    svg.appendChild(el("text", { x: m.l + gi * gw + gw / 2, y: m.t + ih + 20, "text-anchor": "middle",
      "font-size": 12, fill: COL.ink, "font-weight": 600, text: g.label }));
  });
  host.appendChild(svg);
}

// ════════════════════════════════════════════════════════════════ §8 training dynamics
function buildTraining(data) {
  const p = data.points;
  $("#td-caption").textContent = `${data.model} · first vs current accuracy across pretraining (averaged over the K×N grid)`;
  lineChart($("#tdchart"), {
    series: [
      { data: p.map(x => x.fvq), color: COL.good, width: 2.75, name: "FVQ" },
      { data: p.map(x => x.cvq), color: COL.bad, width: 2.75, name: "CVQ" },
    ],
    nLayers: p.length, yMax: 1, yLabel: "accuracy", xLabel: data.x_unit,
    tickLabel: i => (p[i].step / 1e6).toFixed(1) + "M",
  });
  $("#td-legend").replaceChildren(
    legItem(COL.good, "First value (FVQ)"), legItem(COL.bad, "Current value (CVQ)"),
    el("span", { class: "muted", text: data.model }));
  const a = p[0], z = p[p.length - 1];
  $("#td-insight").innerHTML = `From step ${(a.step/1e6).toFixed(1)}M to ${(z.step/1e6).toFixed(1)}M, first-value recall climbs <b style="color:var(--good)">${pct(a.fvq)}% → ${pct(z.fvq)}%</b> while current-value recall trails (<b style="color:var(--bad)">${pct(a.cvq)}% → ${pct(z.cvq)}%</b>). The gap is present early and widens — the model <b>learns to favour the first value</b> over the course of pretraining.`;
}

// ════════════════════════════════════════════════════════════════ §9 format intervention
let fmtMi = 0;
function buildFormat(data, keepState = false) {
  const tog = $("#fmt-model");
  if (!keepState) {
    tog.replaceChildren(...data.models.map((mm, i) => el("button", { "data-i": i, class: i === 0 ? "active" : "", text: mm.model })));
    fmtMi = Math.max(0, data.models.findIndex(mm => mm.model === data.default_model));
    [...tog.children].forEach((b, i) => b.classList.toggle("active", i === fmtMi));
  }
  function draw() {
    const m = data.models[fmtMi];
    const groups = data.formats.map((f, i) => ({
      label: f, bars: [{ v: m.fvq[i], color: COL.good }, { v: m.cvq[i], color: COL.bad }],
    }));
    groupedBar($("#fmtchart"), { groups, yMax: 1 });
    $("#fmt-legend").replaceChildren(
      legItem(COL.good, "First value (FVQ)"), legItem(COL.bad, "Current value (CVQ)"),
      el("span", { class: "muted", text: `${m.model} · ${data.cell}` }));
    const cvqMin = Math.min(...m.cvq), cvqMax = Math.max(...m.cvq);
    const bestF = data.formats[m.cvq.indexOf(cvqMax)], worstF = data.formats[m.cvq.indexOf(cvqMin)];
    $("#fmt-insight").innerHTML = `For ${m.model}, current-value accuracy swings from <b style="color:var(--bad)">${pct(cvqMin)}% (${worstF})</b> to <b>${pct(cvqMax)}% (${bestF})</b> — the layout matters enormously. But the first-value bars stay high throughout: formatting shuffles <i>how badly</i> the current value is lost, without removing the asymmetry.`;
  }
  $$("#fmt-model button").forEach(b => b.onclick = () => {
    $$("#fmt-model button").forEach(x => x.classList.remove("active"));
    b.classList.add("active"); fmtMi = +b.dataset.i; draw();
  });
  draw();
}

// ════════════════════════════════════════════════════════════════ §2 per-model table
function buildScatterTable(data) {
  const host = $("#scatter-table");
  if (!host) return;
  const rows = data.points.map(p => `<tr>
    <td>${p.model}</td>
    <td class="muted">${p.family || "—"}</td>
    <td class="num">${pct(p.fvq)}%</td>
    <td class="num">${pct(p.cvq)}%</td>
    <td class="num ${p.gap > 0 ? "gap-pos" : "gap-neg"}">${p.gap > 0 ? "+" : ""}${pct(p.gap)}%</td>
    <td class="num muted">K=${p.cell.K}, N=${p.cell.N}</td></tr>`).join("");
  host.innerHTML = `<table class="data-table">
    <thead><tr><th>Model</th><th>Family</th><th>FVQ</th><th>CVQ</th><th>Gap</th><th>Cell</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

// ════════════════════════════════════════════════════════════════ shared line chart
function lineChart(host, { series, nLayers, yMax = 1, yLabel, xLabel, markerLayer, chance, xStart = 0, tickLabel }) {
  const xtick = tickLabel || (i => "L" + i);
  host.replaceChildren();
  const W = 880, H = 420, m = { t: 20, r: 20, b: 48, l: 56 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const span = Math.max(1, (nLayers - 1) - xStart);
  const x = i => m.l + ((i - xStart) / span) * iw;
  const y = v => m.t + (1 - v / yMax) * ih;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", style: "max-height:440px" });

  // y grid
  for (let i = 0; i <= 4; i++) {
    const t = (yMax * i) / 4;
    svg.appendChild(el("line", { class: "gridline", x1: m.l, y1: y(t), x2: m.l + iw, y2: y(t) }));
    svg.appendChild(el("text", { x: m.l - 10, y: y(t) + 4, "text-anchor": "end",
      "font-size": 11, fill: COL.muted, text: pct(t / yMax * yMax) + "%" }));
  }
  // x ticks (every ~6 layers, within the visible [xStart, nLayers-1] range)
  const step = Math.max(1, Math.round((nLayers - xStart) / 6));
  for (let i = xStart; i < nLayers; i += step) {
    svg.appendChild(el("text", { x: x(i), y: m.t + ih + 20, "text-anchor": "middle",
      "font-size": 11, fill: COL.muted, text: String(xtick(i)) }));
  }
  if (chance != null) {
    svg.appendChild(el("line", { x1: m.l, y1: y(chance), x2: m.l + iw, y2: y(chance),
      stroke: COL.faint, "stroke-width": 1, "stroke-dasharray": "4 4" }));
    svg.appendChild(el("text", { x: m.l + iw, y: y(chance) - 6, "text-anchor": "end",
      "font-size": 11, fill: COL.faint, text: "chance" }));
  }
  // axis labels
  if (xLabel) svg.appendChild(el("text", { x: m.l + iw / 2, y: H - 6, "text-anchor": "middle",
    "font-size": 12, fill: COL.muted, text: xLabel }));
  if (yLabel) svg.appendChild(el("text", { x: -(m.t + ih / 2), y: 14, "text-anchor": "middle",
    transform: "rotate(-90)", "font-size": 12, fill: COL.muted, text: yLabel }));

  // marker (vertical layer line)
  let marker = null;
  if (markerLayer != null) {
    marker = el("line", { x1: x(markerLayer), y1: m.t, x2: x(markerLayer), y2: m.t + ih,
      stroke: COL.accent, "stroke-width": 1.5, "stroke-opacity": .5 });
    svg.appendChild(marker);
  }

  // series lines (only within visible [xStart, nLayers-1])
  series.forEach(s => {
    if (!s.data) return;
    const pts = s.data.map((v, i) => [i, v])
      .filter(([i, v]) => i >= xStart && v != null && isFinite(v))
      .map(([i, v]) => `${x(i)},${y(v)}`).join(" ");
    svg.appendChild(el("polyline", { points: pts, fill: "none", stroke: s.color,
      "stroke-width": s.width || 2.5, "stroke-dasharray": s.dash || "",
      "stroke-opacity": s.opacity == null ? 1 : s.opacity,
      "stroke-linejoin": "round" }));
  });

  // marker dots on each series at markerLayer
  const dots = [];
  if (markerLayer != null) series.forEach(s => {
    if (!s.data) return;
    const d = el("circle", { cx: x(markerLayer), cy: y(s.data[markerLayer]), r: 4.5,
      fill: s.color, stroke: "#fff", "stroke-width": 1.5,
      "fill-opacity": s.opacity == null ? 1 : s.opacity });
    svg.appendChild(d); dots.push({ s, d });
  });

  host.appendChild(svg);
  return { x, y, marker, dots, svg };
}

// ════════════════════════════════════════════════════════════════ §3 logit lens
// Competition view: for the selected model / query / version, show P(correct
// value), P(strongest competing value), and total mass on any tracked value,
// across the late layers. Gemma recalls the first value almost perfectly (P≈1)
// while the current value is weaker; Qwen shows surfaced-then-suppressed.
let llState = { m: null, v: "base", c: "PI", cell: null, layer: null };
function buildLogitLens(root, keepState = false) {
  if (!keepState || !llState.m || !root.models[llState.m]) {
    llState.m = root.default_model;
    llState.c = root.default_cond || "PI";
    llState.cell = root.default_cell;
    llState.layer = null;
  }
  const sel = $("#ll-cell");

  function syncForModel() {
    const M = root.models[llState.m];
    const cells = M.cells;
    sel.replaceChildren(...cells.map(c => el("option", { value: `${c.K}_${c.N}`, text: c.label })));
    if (!cells.find(c => `${c.K}_${c.N}` === llState.cell)) llState.cell = `${cells[0].K}_${cells[0].N}`;
    sel.value = llState.cell;
    const xStart = Math.max(0, M.n_layers - 12);
    const slider = $("#ll-slider");
    slider.min = xStart; slider.max = M.n_layers - 1;
    if (llState.layer == null || llState.layer < xStart || llState.layer > M.n_layers - 1)
      llState.layer = M.n_layers - 1;
    slider.value = llState.layer;
    return { M, cells, xStart };
  }

  function draw() {
    const { M, cells, xStart } = syncForModel();
    const v = llState.v, c = llState.c;
    const s = M.series[v][c][llState.cell];
    if (!s) return;
    const correctCol = c === "PI" ? COL.bad : COL.good;
    const isCVQ = c === "PI";
    const series = [
      { data: s.total, color: COL.faint, width: 1.5, dash: "3 3", opacity: .85, name: "total" },
      { data: s.correct, color: correctCol, width: 2.75, name: "correct" },
      { data: s.top_other, color: COL.lora, width: 2.5, dash: "6 3", name: "competitor" },
    ];
    lineChart($("#logitlens"), {
      series, nLayers: M.n_layers, yMax: 1, xStart,
      yLabel: "probability", xLabel: "layer (late layers only)", markerLayer: +llState.layer,
    });
    const qword = isCVQ ? "last" : "first";
    $("#ll-caption").textContent =
      `${llState.m} · ${M.dataset} · ${v === "base" ? "Baseline" : "+ LoRA"} · query: ${qword} value · ${cells.find(c2 => `${c2.K}_${c2.N}` === llState.cell).label}`;
    const L = +llState.layer;
    $("#ll-layerval").innerHTML =
      `L${L} · correct=${s.correct[L].toFixed(2)} · top competitor=${s.top_other[L].toFixed(2)}`;
    $("#ll-legend").replaceChildren(
      legItem(correctCol, `Correct ${qword} value`),
      legItem(COL.lora, "Strongest competing value"),
      legItem(COL.faint, "Any tracked value (total mass)"),
    );
    // insight
    const cEnd = s.correct[s.correct.length - 1], oEnd = s.top_other[s.top_other.length - 1];
    const cPk = Math.max(...s.correct), cPkL = s.correct.indexOf(cPk);
    const acc = pct(s.accuracy);
    const firstReadable = s.total.findIndex(x => x > 0.02);
    const readNote = firstReadable >= 0
      ? ` Everything sits at ≈0 until ~L${firstReadable} (the flat region is cropped) — retrieval is a late-layer event.`
      : "";
    if (v === "base" && isCVQ) {
      const suppressed = cPk - cEnd > 0.05;
      $("#ll-insight").innerHTML = `The model answers this <b>last-value</b> query correctly only <b style="color:var(--bad)">${acc}%</b> of the time. The correct value reaches <b style="color:var(--bad)">P=${cEnd.toFixed(2)}</b>${suppressed ? ` (after surfacing to ${cPk.toFixed(2)} at L${cPkL} and being suppressed)` : ""}, while a <b>competing</b> value holds <b style="color:var(--lora)">P=${oEnd.toFixed(2)}</b> — the right answer is computed but doesn't dominate the readout.${readNote}`;
    } else if (v === "lora" && isCVQ) {
      $("#ll-insight").innerHTML = `With LoRA, last-value accuracy rises to <b style="color:var(--lora)">${acc}%</b> and the correct value dominates the output (<b style="color:var(--bad)">P=${cEnd.toFixed(2)}</b>, competitors collapse to ${oEnd.toFixed(2)}). The fine-tune didn't change <i>where</i> retrieval happens — only whether the right value wins.${readNote}`;
    } else {
      $("#ll-insight").innerHTML = `The model recalls the <b>first</b> value almost perfectly here — <b style="color:var(--good)">${acc}%</b> accuracy, P=<b style="color:var(--good)">${cEnd.toFixed(2)}</b> at the output, with competing values near zero (${oEnd.toFixed(2)}). Primacy is clean${v === "lora" ? "; LoRA leaves it intact" : ""}.${readNote} Toggle to <b>Last (CVQ)</b> to see the contrast.`;
    }
  }

  $$("#ll-version button").forEach(b => b.onclick = () => {
    $$("#ll-version button").forEach(x => x.classList.remove("active"));
    b.classList.add("active"); llState.v = b.dataset.v; draw();
  });
  $$("#ll-cond button").forEach(b => b.onclick = () => {
    $$("#ll-cond button").forEach(x => x.classList.remove("active"));
    b.classList.add("active"); llState.c = b.dataset.c; draw();
  });
  sel.onchange = () => { llState.cell = sel.value; draw(); };
  $("#ll-slider").oninput = () => { llState.layer = +$("#ll-slider").value; draw(); };
  draw();
}

// ════════════════════════════════════════════════════════════════ §4 probing
let prState = { v: "base" };
function buildProbing(data, keepState = false) {
  const P = data.probes;
  $("#pr-caption").textContent = `${data.model} · linear probe accuracy across layers`;
  function draw() {
    const v = prState.v;
    const other = v === "base" ? "lora" : "base";
    const series = [
      { data: P.condition_probe[v], color: COL.accent, width: 2.5, name: "knows-which-question" },
      { data: P.RI_correct_probe[v], color: COL.good, width: 2.5, name: "Will it get FIRST right?" },
      { data: P.PI_correct_probe[v], color: COL.bad, width: 3, name: "Will it get LAST right?" },
      // ghost of PI-correct in the other version, to show the jump
      { data: P.PI_correct_probe[other], color: COL.bad, width: 1.5, dash: "4 4", opacity: .35,
        name: "(other version)" },
    ];
    lineChart($("#probing"), {
      series, nLayers: data.layers.length, yMax: 1, chance: 0.5,
      yLabel: "probe accuracy", xLabel: "layer",
    });
    $("#pr-legend").replaceChildren(
      legItem(COL.accent, "Can decode which query was asked (control)"),
      legItem(COL.good, "Can predict a correct FIRST answer"),
      legItem(COL.bad, "Can predict a correct LAST answer"),
      el("span", { class: "muted", text: "dashed = other version, for comparison" }),
    );
    const piBaseEnd = P.PI_correct_probe.base[P.PI_correct_probe.base.length - 1];
    const piLoraEnd = P.PI_correct_probe.lora[P.PI_correct_probe.lora.length - 1];
    $("#pr-insight").innerHTML = `The <b style="color:var(--accent)">control probe</b> (can we tell whether the model was asked for FIRST vs LAST?) is ~100% from the early layers — so the model is never <i>confused about the question</i>. But whether it will get the <b>last</b> value right is unreadable in the baseline (<b>${pct(piBaseEnd)}%</b>, near chance) — and jumps to <b style="color:var(--lora)">${pct(piLoraEnd)}%</b> after LoRA, in the very same late layers. The substrate was there; LoRA installed the readout.`;
  }
  $$("#pr-version button").forEach(b => b.onclick = () => {
    $$("#pr-version button").forEach(x => x.classList.remove("active"));
    b.classList.add("active"); prState.v = b.dataset.v; draw();
  });
  draw();
}

function legItem(color, label) {
  return el("span", {}, [el("span", { class: "dot", style: `background:${color}` }), document.createTextNode(label)]);
}

// ════════════════════════════════════════════════════════════════ §5 fix bars
let fixCellIdx = 2; // default to (15,30) — the clearest reversal->fixed cell
function buildFix(data) {
  const cells = data.cells;

  // cell selector
  const head = $("#fix-bars").parentElement.querySelector(".fix-cellsel");
  if (!head) {
    const sel = el("div", { class: "controls-row fix-cellsel" }, [
      el("span", { class: "dim-label", text: `${data.model} · load level (K keys, N updates)` }),
      el("div", { class: "toggle", id: "fix-cells" },
        cells.map((c, i) => el("button", { "data-i": i, class: i === fixCellIdx ? "active" : "",
          text: `K=${c.K}, N=${c.N}` }))),
    ]);
    $("#fix-bars").parentElement.insertBefore(sel, $("#fix-bars"));
    $$("#fix-cells button").forEach(b => b.onclick = () => {
      $$("#fix-cells button").forEach(x => x.classList.remove("active"));
      b.classList.add("active"); fixCellIdx = +b.dataset.i; render();
    });
  }

  function group(title, d, accent) {
    const g = el("div", { class: "bargroup" });
    g.appendChild(el("div", { class: "head", text: title, style: accent ? `color:${accent}` : "" }));
    [["FIRST value (FVQ)", d.fvq, "fvq"], ["LAST value (CVQ)", d.cvq, "cvq"]].forEach(([lab, val, cls]) => {
      const row = el("div", { class: "bar-row" });
      row.appendChild(el("div", { class: "lab", html: `<span>${lab}</span><span class="mono"><b>${pct(val)}%</b></span>` }));
      const track = el("div", { class: "bar-track" });
      const fill = el("div", { class: "bar-fill " + cls, style: "width:0%" });
      track.appendChild(fill); row.appendChild(track); g.appendChild(row);
      requestAnimationFrame(() => setTimeout(() => fill.style.width = pct(val) + "%", 120));
    });
    const gap = d.gap, big = Math.abs(gap) > 0.03;
    g.appendChild(el("div", { style: "margin-top:14px",
      html: `<span class="gap-badge" style="background:${big?'var(--bad-soft)':'var(--good-soft)'};color:${big?'var(--bad)':'var(--good)'}">gap ${gap>0?'+':''}${pct(gap)}%</span>` }));
    return g;
  }

  function render() {
    const c = cells[fixCellIdx];
    const wrap = $("#fix-bars"); wrap.replaceChildren();
    wrap.appendChild(group("Baseline", c.base, COL.ink));
    wrap.appendChild(group("+ LoRA fine-tune", c.lora, COL.lora));
    const bg = c.base.gap;
    $("#fix-gap").innerHTML = `<span class="lead">At K=${c.K}, N=${c.N} the baseline gap of <b style="color:var(--bad)">${bg>0?'+':''}${pct(bg)}%</b> collapses to <b style="color:var(--lora)">0%</b> — both queries hit <b>100%</b>.</span>`;
  }
  render();

  // mechanism deltas (probe / logit lens / attention) — the "why"
  const m = data.mechanism;
  const rows = [...m.probe, ...m.logit_lens, ...(m.attention || [])];
  const mech = el("div", { style: "margin-top:30px" }, [
    el("div", { class: "dim-label", style: "margin-bottom:12px", text: `What changed inside ${data.model} (paper §7, bootstrap 95% CIs)` }),
  ]);
  rows.forEach(r => {
    const row = el("div", { class: "bar-row", style: "display:grid;grid-template-columns:1fr auto;gap:14px;align-items:center" });
    row.appendChild(el("div", { class: "mono", style: "font-size:.86rem", text: r.q }));
    row.appendChild(el("div", { html:
      `<span class="mono muted">${r.base.toFixed(2)}</span> <span style="color:var(--faint)">→</span> <span class="mono" style="color:var(--lora);font-weight:600">${r.lora.toFixed(2)}</span>` }));
    mech.appendChild(row);
  });
  const existing = $("#fix-mech"); if (existing) existing.remove();
  mech.id = "fix-mech";
  $("#fix-bars").parentElement.appendChild(mech);

  // per-cell LoRA transfer table
  const ft = $("#fix-table");
  if (ft) {
    const rows = data.cells.map(c => `<tr>
      <td class="num">K=${c.K}, N=${c.N}</td>
      <td class="num">${pct(c.base.fvq)}%</td><td class="num">${pct(c.base.cvq)}%</td>
      <td class="num ${Math.abs(c.base.gap) > 0.03 ? "gap-pos" : "gap-neg"}">${c.base.gap > 0 ? "+" : ""}${pct(c.base.gap)}%</td>
      <td class="num">${pct(c.lora.fvq)}%</td><td class="num">${pct(c.lora.cvq)}%</td>
      <td class="num gap-neg">${c.lora.gap > 0 ? "+" : ""}${pct(c.lora.gap)}%</td></tr>`).join("");
    ft.innerHTML = `<table class="data-table">
      <thead><tr><th>Cell</th><th>FVQ base</th><th>CVQ base</th><th>Gap base</th>
        <th>FVQ +LoRA</th><th>CVQ +LoRA</th><th>Gap +LoRA</th></tr></thead>
      <tbody>${rows}</tbody></table>`;
  }

  // attention-routing: 4 promoter heads, baseline → +LoRA
  const fr = $("#fix-routing");
  if (fr && m.attention) {
    const groups = m.attention.map(a => ({
      label: a.q.replace("P(attend v_last), ", ""),
      bars: [{ v: a.base, color: COL.muted }, { v: a.lora, color: COL.lora }],
    }));
    groupedBar(fr, { groups, yMax: 1 });
    const lg = el("div", { class: "legend" }, [legItem(COL.muted, "Baseline"), legItem(COL.lora, "+ LoRA")]);
    fr.appendChild(lg);
  }
  // attention-vs-MLP ablation table
  const fa = $("#fix-ablation");
  if (fa && data.ablation) {
    const g = v => `<td class="num ${Math.abs(v) > 0.03 ? "gap-pos" : "gap-neg"}">${v > 0 ? "+" : ""}${pct(v)}%</td>`;
    const rows = data.ablation.cells.map(c => `<tr>
      <td class="num">K=${c.K}, N=${c.N}</td>${g(c.base_gap)}${g(c.attn_gap)}${g(c.mlp_gap)}</tr>`).join("");
    fa.innerHTML = `<table class="data-table">
      <thead><tr><th>Cell</th><th>Gap · baseline</th><th>Gap · attention-LoRA</th><th>Gap · MLP-LoRA</th></tr></thead>
      <tbody>${rows}</tbody></table>`;
  }

  const story = m.story.replace(/^The skill was latent, not missing\.\s*/, "");
  $("#fix-insight").innerHTML = `<strong>The skill was latent, not missing.</strong> ${story}`;
}

// ════════════════════════════════════════════════════════════════ reveal
function setupReveal() {
  const nodes = $$("section .card, .section-head, .key-insight");
  if (!("IntersectionObserver" in window)) { nodes.forEach(n => n.classList.add("in")); return; }
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); obs.unobserve(e.target); } });
  }, { threshold: 0.08 });
  nodes.forEach(n => { n.classList.add("reveal"); obs.observe(n); });
  // Safety net: never leave content hidden if the observer doesn't fire
  // (e.g. headless render, no-scroll environments).
  setTimeout(() => nodes.forEach(n => n.classList.add("in")), 1500);
}
