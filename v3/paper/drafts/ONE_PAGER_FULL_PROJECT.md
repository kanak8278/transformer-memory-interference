# Transformers Remember First, Forget Last Extension

---

## The Problem

`Made up Scenario and Facts are not correct`

Give any language model a sequence of updates to the same value:

> *"The item costs $42... the item now costs $87... the item now costs $15..."*

Then ask: **"What did the item cost originally?"** — models answer correctly ~80% of the time.
Ask: **"What does the item cost now?"** — models answer correctly ~30% of the time.

Models are dramatically better at recalling the *first* value they saw than the *most recent*. We call this **PI > RI**: Proactive Interference dominates Retroactive Interference. It is the opposite of human memory.

---

## Scale and Universality

We test across **9 models, 2 architecture families** (transformers + Mamba state-space), plus 2 frontier API models.

- **91% of test conditions** show PI > RI
- **Mean gap: +50 percentage points**
- **PI decays exponentially** with the number of updates: PI(N) = a·e^{−bN} + c
- **RI remains robust** (80–100%) regardless of how many updates follow the first

The effect is not subtle. At 10 updates, TinyLlama scores RI=100%, PI=0%.

---

## Why It's Architectural, Not Learned

Three lines of evidence establish this is built into the architecture, not an artifact of training data or fine-tuning.

**1. It's there before any training.**
Jacobian analysis at random initialization shows primacy bias: the first-quarter of a sequence influences the output 1.47× more than the middle (transformers), and even more extremely for Mamba (295×). The bias is in the architecture's geometry, not learned weights.

**2. Attention isn't the cause.**
Mamba SSMs — which have *no* causal attention, softmax, or positional encoding — show PI > RI with gap +57–70%. Bidirectional models (Flan-T5) show *no* PI > RI (gap = −9%). The culprit is **autoregressive sequential processing with fixed-capacity state**, not any one mechanism unique to transformers.

**3. The correct value is computed, then suppressed.**
Logit lens analysis shows the last value (v_last) reaches its peak probability at intermediate layers, but is consistently *outcompeted* in the final layers by adjacent values. First-value retrieval (v_0) is clean and monotonic from early layers to output. Probing classifiers encode RI correctness at 87% accuracy but PI correctness at only 61% — barely above chance. The model knows the last value exists but can't hold onto it.

---

## Formal Theory

We formalize two conditions under which first-value retrieval necessarily dominates:

- **Monotone overwrite:** Each new update partially overwrites the previous state (proved for single-layer attention)
- **Diminishing marginal overwrite:** Each successive overwrite has less impact than the last (derived from existing theory, Chowdhury 2026)

Together these predict all observed phenomena: exponential PI decay, robust RI, the Mamba result, and the bidirectional control.

---

## It Transfers to Real Text

We test whether the effect survives when the task is embedded in naturalistic narrative — not a stripped-down KV sequence, but actual prose.

We built three narrative generators: **wildlife tracking** (GPS-collar field reports), **hospital ICU** (clinical shift notes), and **air traffic control** (sector transcripts).

| Domain | Qwen 1.5B | Claude Haiku |
|--------|-----------|--------------|
| Synthetic KV pairs | PI>RI, +19% gap | No effect |
| Wildlife narratives | **PI>RI, +27% gap** | No effect |
| ICU clinical notes | No effect | No effect |
| ATC transcripts | Slight positive | **Reversed, −34%** |

**Wildlife:** PI > RI transfers with *larger* effect size than synthetic data. Not an artifact.
**ICU:** Null result — explicit value labeling ("HR = 125 bpm") equalizes retrieval difficulty.
**ATC:** Haiku *reverses* — transcript format ends with the most recent instruction, so strong models exploit document structure to find the last value. This is a different failure mode: strategic recency rather than architectural primacy.

---

## The Big Picture

| Model strength | KV pairs | Narrative |
|----------------|----------|-----------|
| **Weak (Qwen 1.5B)** | PI>RI everywhere | PI>RI where format requires sequential tracking |
| **Strong (Haiku)** | No primacy bias | No primacy bias; shows recency bias in transcript format |

Weaker models show primacy bias everywhere. Stronger models escape primacy but become susceptible to document-level recency when format enables it. Neither is "correct" — both are architecture-induced biases.

---

## Why This Matters

**For AI safety:** Models are systematically worse at recalling *what you just told them* than *what you told them first*. In any application that updates instructions or facts during a session, the most recent information is the least reliably recalled.

**For mechanistic interpretability:** We show for the first time that a major behavioral asymmetry is present at initialization, survives architecture changes, and has a geometric explanation rooted in how sequential state is maintained.

**For benchmarking:** Standard RAG and in-context learning benchmarks don't stress the primacy/recency axis. Our operating-point framework (num_keys × num_updates) provides a principled way to do so.

---

*9 models tested · 3 narrative domains · 50,000+ trials · Code + generators open-sourced*
