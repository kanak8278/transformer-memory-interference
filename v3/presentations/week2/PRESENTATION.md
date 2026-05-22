# Week 2 Research Update — Presentation Plan (compressed to 9 slides)

**Format:** ~25–30 min talk + discussion. Posture: research-discussion update with open questions, not a final-results report.

**One-line thesis:** *"We've designed a controlled retrieval task to study positional asymmetry in in-context information access. Pilot experiments on the Qwen family reveal three distinct regimes — and that variation, not a single finding, is the most interesting result so far."*

**Slide count:** 9 (incl. title + discussion). Compressed from 12 — methodology folded into task-design slide; three-regimes folded into results slide; Phase 2 plan paired with open questions.

**Folder layout:**
- `PRESENTATION.md` — this file (the plan)
- `plots/` — figures (PNGs + JSON for React)
- `examples/` — sample prompts (one per dataset + the vitals log)
- `scripts/` — Python generators
- `react/` — single self-contained `deck.html` + portable `Slide7QwenResults.jsx`

**Plot legend used in this doc:**
- `[HAVE]` — file already exists, path given
- `[BUILD]` — must be generated; data source named, script path named
- `[TEXT]` — slide-native (no PNG needed)

---

# Slide 1 — Title

## On screen
- **Title:** Position-Dependent Retrieval in Language Models
- **Subtitle:** Week 2 Update — Research Discussion
- Your name · date

## Say (≈30 sec)
- "This is a research discussion, not a results readout. I'll share what we've designed, what we've measured, what surprised us, and what we plan to do next. Please interrupt with questions — that's the point."

## Visual
- None. Clean title.

## Status
- `[TEXT]` only

---

# Slide 2 — The Problem

## On screen
- **Header:** "Does the position of information in context change retrieval?"
- **Left half — the motivating example** (monospace, dense):
  ```
  ICU monitor → log file → LLM directly answering queries

  [14:02]  HR=82   BP=128/82   SpO2=98   Temp=37.1
  [14:17]  HR=88   BP=125/80   SpO2=97   Temp=37.3
  [14:32]  HR=94   BP=118/76   SpO2=96   Temp=37.6
  [14:47]  HR=102  BP=110/72   SpO2=94   Temp=38.0
  [15:02]  HR=115  BP=98/64    SpO2=92   Temp=38.4
  [15:17]  HR=128  BP=92/58    SpO2=90   Temp=38.7
  ```
  | Query                                        | Expected | What it tests |
  |---|---|---|
  | "What was the heart rate at admission?"       | **82**   | Recall first value (RI) |
  | "What is the patient's current heart rate?"   | **128**  | Recall most-recent value (PI) |

- **Right half — Why this isn't Lost-in-the-Middle:**
  | | **Lost in the Middle** | **This work** |
  |---|---|---|
  | Information structure | One fact + many *static distractors* | Same key, *repeatedly overwritten* |
  | Context fill | Near-saturated (often >75% of capacity) | Moderate (5–30% of capacity) |
  | Failure type | Retrieval of a unique target | Conflict resolution between alternatives |
  | Question being asked | "Where does the model look?" | "Which version wins?" |

## Say (≈3 min)
- Anchor on the ICU example. *"A monitor writes vitals every 15 min, fed directly to an LLM. Two clinically valid queries — admission baseline vs current state. If the model returns 82 when asked 'current,' patient decompensates and no one knows."*
- *"This pattern is everywhere — chat with mid-context corrections, RAG with updated docs, agent tool use, instruction-following with revisions. Anywhere state is updated in-context, the most recent value is the one that matters most — and the one we'll show is the least reliably recalled."*
- Pivot to LITM table. *"This will sound like Lost-in-the-Middle. It isn't. LITM is no-interference retrieval at near-saturated context. We're studying interference resolution at moderate context fill. Different problem space, different mechanism candidates. I'm flagging this up front so you don't mentally substitute LITM for what I'm about to describe."*

## Status
- `[TEXT]`

---

# Slide 3 — Goal & Roadmap

## On screen
- **Header:** Three-phase research arc. Today is Phase 1.
- Three boxes in a horizontal row with arrows:
  - 1. CHARACTERISE — Does it exist? When? For whom? **(TODAY)**
  - 2. ABLATE — What controls it? Test causal candidates. (~2 weeks)
  - 3. PROBE — Where in the network does it live? (~4–6 weeks)
- **Today's discussion:** Phase 1 results, open questions, planned ablations. Want feedback on (a) task-design cleanliness, (b) which open question to chase first, (c) whether the planned controls are right.

## Say (≈1 min)
- Walk the roadmap. Emphasise that today is Phase 1 only. Be explicit about wanting feedback on the foundation before chasing the mechanistic question.

## Status
- `[TEXT]`

---

# Slide 4 — Task Design  (incl. methodology footer)

## On screen
- **Header:** Controlled key–value interference task
- **Top — schematic of the interference stream:**
  ```
  Stream of K keys, each updated N times in random order:

  Key1: value_a
  Key2: value_b
  Key1: value_c     ← Key1 updated
  Key3: value_d
  Key1: value_e     ← Key1 updated again
  Key2: value_f     ← Key2 updated
  ...
  ```
- **Middle — two query types side-by-side** (boxed):
  | **RI** (Retroactive Interference) | **PI** (Proactive Interference) |
  |---|---|
  | "What was the **first** value of Key1?" | "What is the **current** value of Key1?" |
  | Expected: `value_a` | Expected: `value_e` |
  | Tests: recall first despite later updates | Tests: recall last despite earlier updates |
- **Bottom — independent variables + footer:**
  - **K** keys (working-memory load): {2, 3, 5, 7, 10, 15, 20, 25, 30}
  - **N** updates per key (interference depth): {5, 7, 10, 15, 20, 30, 50, 75, 100}
  - 100 trials per (K, N, condition) cell · greedy decoding · exact-match scoring · Wilson 95% CIs
- **Footer note (faded text):** *"Methodology check: in our first sweep garbage rate was 60–80% (model parroting question / leaking role tags). Tightening the system prompt cut it to <6%. Numbers reported today are post-fix."*

## Say (≈3 min)
- Walk the stream. Then the two queries.
- Emphasise K and N varying *independently* — separates working memory from interference.
- One sentence on the prompt-fix.

## Status
- `[TEXT]`

---

# Slide 5 — Datasets (Design as Falsification)

## On screen
- **Header:** Three datasets, each removes one suspected confound.
- **Three-column comparison:**

  | Dataset | What it is | Confound removed |
  |---|---|---|
  | **ARBITRARY-SINGLE** | 2,300 single-token English words across 46 categories. No semantic association between key and value. | "Maybe the model is using semantic priors." Forces *positional* retrieval. |
  | **SEMANTIC-MULTI** | 2,403 multi-token category-anchored values, per-category pools (e.g., fruit pool, country pool). | "Maybe the effect is specific to single-token answers." Tests *vocabulary realism* and tokenization. |
  | **NARRATIVE** | Generated naturalistic prose with implicit updates embedded in story flow (Dota-2 match-commentary generator). | "Maybe it's an artifact of artificial KV format." Tests *real-world prose*. |

- **Sample prompts (small, one per dataset)** — see `examples/{arbi,sem,narrative}_prompt.txt`.

## Say (≈2 min)
- Three datasets, each designed to remove a different "just because" counter-explanation.
- Mention that more naturalistic narrative domains are in progress (planned: ICU, GPS-collar wildlife, ATC).

## Status
- `[TEXT]`

---

# Slide 6 — Results: Qwen Family  (incl. three-regimes takeaway)

## On screen
- **Header:** Qwen 2.5 family — full-grid means, 100 trials/cell
- **Centerpiece visual:** color-density table — 4 models × {RI, PI, Gap}, two side-by-side panels (ARBI / SEM).
- **Below the table — three-regime takeaway box** with the puzzle framing:

  | **Strong primacy bias** | **Symmetric** | **Reversed** |
  |---|---|---|
  | RI ≫ PI · gap +30 to +46% | RI ≈ PI · gap +2.5% | PI > RI · gap −12% |
  | Qwen 0.5B Inst, 1.5B Inst | Qwen 3B Inst | Qwen 3B Base (no SFT) |
  | *"Smaller instruct → primacy"* | *"Capacity overcomes it"* | *"Removing SFT flips it"* |

## Say (≈4 min)
- Walk the table left-to-right by model. Lead with the three-regime claim.
- *"Within one architecture family, three regimes. The asymmetry is not universal — it can be created and undone."*
- Two-dataset cross-validation: directions agree across ARBI and SEM.

## Visual
- **Table:** `[BUILD/HAVE]` — `plots/slide7_qwen_table.html` (also: PNG screenshot, JSON data)
- **Three-regime takeaway:** `[TEXT]` typeset in deck.

## Status
- HTML table embedded; rest text-native.

---

# Slide 7 — Error Classification: How They Fail

## On screen
- **Header:** When PI fails, where does the answer come from?
- **Top — error taxonomy (one-line each):**
  - **CORRECT** — exact match with last value
  - **PRIMACY INTRUSION** — returned the *first* value
  - **INTERMEDIATE INTRUSION** — returned a value from somewhere in the stream
  - **GARBAGE** — output not in the value stream at all
- **Middle — stacked bar:** `plots/slide9a_error_types.png` (aggregate + 2k_10u, 4 Qwen models)
- **Bottom — failure-position histogram:** `plots/slide9b_failure_pos.png` (3 instruct Qwen variants, low N vs high N)
- **One-line takeaway:** *"At low N, PI failures cluster off-by-one (penultimate). At high N they spread. Failure mode is N-dependent — a clue about the mechanism."*

## Say (≈3 min)
- Walk the taxonomy.
- Stacked bar: orange (intermediate intrusion) is the dominant error mode for the failing models.
- Histogram: low-N spike at v_{N-1}; high-N flattens. The model knows the answer is *near* the end at low N, but loses precision; at high N it can't address late positions at all.

## Status
- 2 PNGs `[HAVE]` + `[TEXT]` for the taxonomy

---

# Slide 8 — Open Questions  (paired with planned next experiments)

## On screen
- **Header:** Open questions and the experiment that answers each.
- **Two-column table** — left = question, right = planned ablation:

  | Open question | Planned next experiment |
  |---|---|
  | **Q1.** Is this Qwen-specific or general across families? | **Cross-family sweep** — Gemma, TinyLlama, StableLM at the same operating points. |
  | **Q2.** Why does Qwen 3B Inst escape it? | Same cross-family sweep — tests if "symmetric at large size" is universal. |
  | **Q3.** Why does removing SFT flip the direction? | **SmolLM3 checkpoint sweep** — pretraining → SFT → alignment checkpoints. |
  | **Q4.** Does the failure surface look U-shaped, single-sided, or other? | **Lost-in-the-Middle scan** — measure accuracy(*i*) for every position *i*, not just first/last. |
  | **Q5.** Is the bottleneck positional addressing or content binding? | **Numbered-update probe** — tag updates with explicit integers; query by tag. If gap collapses, it's positional. |

- **Bottom note:** *"Phase 3 (mechanistic probing) only after these ablations narrow the candidate causes."*

## Say (≈4 min)
- Each row: name the question, name the experiment, name the bet.
- *"For each of these I have a current bet, but I want your priors. Which is most worth resolving first? That decides Phase 2's order of operations."*

## Status
- `[TEXT]`

---

# Slide 9 — Discussion

## On screen
- **Header:** Discussion
- Re-display the 5 open questions (small text).
- Single bold prompt at bottom: *"Which question is most worth answering first?"*

## Say (≈30 sec to set up, then conversation)
- "I'd like to spend the rest of our time on three things, in order: is the task design clean enough to trust the regime distinctions; which of the open questions matters most to resolve first; what would falsify the hypothesis that this is a training-induced bias."

## Status
- `[TEXT]`

---

# Plot Build Inventory

| Slide | Visual | Status | Source / Path |
|---|---|---|---|
| 1 | none | — | — |
| 2 | vitals log + LITM table | `[TEXT]` | typeset on slide |
| 3 | 3-phase roadmap | `[TEXT]` | typeset on slide |
| 4 | KV stream + RI/PI box + footer | `[TEXT]` | typeset on slide |
| 5 | 3-dataset table + 3 prompt excerpts | `[TEXT]` | typeset on slide; see `examples/` |
| **6** | **HTML table + 3-regime takeaway** | `[BUILD]` ✅ | `plots/slide7_qwen_table.html` + JSON |
| **7** | **Stacked bar + position histogram** | `[BUILD]` ✅ | `plots/slide9a_error_types.png` + `plots/slide9b_failure_pos.png` (+ JSONs) |
| 8 | Q-vs-experiment table | `[TEXT]` | typeset on slide |
| 9 | discussion prompt | `[TEXT]` | typeset on slide |

**Build summary:** 1 HTML table, 2 matplotlib PNGs, 4 prompt text files. All other slides are typed directly.

---

# Discussion-Defence Pre-Reads (your prep, not on slides)

- **"Why only Qwen?"** *"Controlled experiment — same architecture, three sizes, base vs instruct. Cross-family is week 3–4."*
- **"What's your hypothesis for why?"** *"Three candidates in tension — positional addressing under softmax dispersion, training-induced primacy bias from SFT data distributions, attention-sink reinforcement. The Phase 2 ablations distinguish them."*
- **"Is the base-model reversal real or a garbage artifact?"** *"Open question. Garbage rate on Qwen 3B Base is [TBD — fill in from trial JSONs before the meeting]. If well below 30%, the reversal is real. If higher, it's confounded."*
- **"Have you tested API/proprietary models?"** *"Not yet, deliberately. We want behavioral baselines on open weights first so the Phase 3 mechanistic work has compatible models."*
- **"How does this differ from Lost-in-the-Middle?"** Refer to slide 2 distinction.
- **"How does this differ from needle-in-haystack?"** *"NIH retrieves one fact from a long noisy context. We have repeated overwrites of the *same* fact at moderate context fill."*
