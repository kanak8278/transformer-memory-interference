# Mechanistic Probing Design: Small Model Experiments

## Model Selection

| Model | Size | TransformerLens | Instruction Following? | MPS Compatible |
|-------|------|----------------|----------------------|----------------|
| GPT-2 small | 124M | Native | No (completion only) | Yes |
| GPT-2 medium | 355M | Native | No (completion only) | Yes |
| Pythia-70M | 70M | Native | No (completion only) | Yes |
| Pythia-160M | 160M | Native | No (completion only) | Yes |
| Pythia-410M | 410M | Native | No (completion only) | Yes |
| Gemma-2-2B | 2B | Supported | Basic yes | Yes but slower |
| Llama-3.2-1B | 1B | Community support | Yes | Yes but slower |

**Primary:** Pythia-160M (fast, well-studied, training checkpoints available at every 1000 steps)
**Validation:** GPT-2 medium (355M)
**If instruction-following needed:** Gemma-2-2B

Pythia bonus: EleutherAI released checkpoints throughout training — enables studying how interference patterns emerge *during training*.

---

## Task Reformulation for Completion Models

Completion models can't follow instructions like "What was the INITIAL value?" — they just continue text. Reformulate as pattern completion with few-shot in-context demonstrations.

### Prompt Format

```
art: impressionism
tool: hammer
gem: ruby
art: baroque
tool: screwdriver
gem: sapphire
art: cubism
tool: wrench
gem: emerald

The first value of art was: impressionism
The first value of tool was: hammer
The first value of gem was:
```

Model should complete with `ruby`. For PI (proactive interference), same sequence but:
```
The last value of gem was:
```

The few solved examples before the test query teach the model the task format in-context.

### Task Simplifications (vs. original paper)

- **5-10 categories** (not 46) — small models can't handle 46
- **Short keys** (3-5 chars): "art", "gem", "tool" — not "visual art", "gemstone"
- **N ∈ {1, 3, 5, 10, 20}** updates — small models cap out earlier
- **Single-token values** where possible — makes logit analysis clean (one token to track)

### Controlled Conditions

```
Condition 1 (baseline):  art: X                      → "first value of art was:" → expect X
Condition 2 (N=3 RI):   art: X, art: Y, art: Z      → "first value of art was:" → expect X
Condition 3 (N=3 PI):   art: X, art: Y, art: Z      → "last value of art was:"  → expect Z
Condition 4 (N=10 RI):  art: X, ..., art: Z10        → "first value of art was:" → expect X
Condition 5 (N=10 PI):  art: X, ..., art: Z10        → "last value of art was:"  → expect Z10
```

---

## What We Measure (The Probing Pipeline)

### Analysis 1: Attention Patterns (easiest, most direct)

**What:** For every attention head at every layer, extract attention weights from the **query token** (answer generation position) to all preceding tokens.

**How:**
```python
# TransformerLens gives cache["pattern", layer]
# Shape: [batch, heads, seq_pos, seq_pos]
# Take the row for the final token position (where answer is generated)
attn_to_positions = cache["pattern", layer][0, :, -1, :]  # [heads, seq_len]
```

**What we look for:**
- Which positions get highest attention? Initial value token? Latest value? Category keys?
- Does this differ between RI (asking for first) vs PI (asking for last)?
- Across layers: do early layers attend broadly, late layers narrow to specific positions?

**Visualization:** Heatmap of attention weights × position, colored by token type (initial value, update, query).

### Analysis 2: Logit Lens (the killer analysis)

**What:** At *every layer*, take the residual stream at the answer position, project through unembedding matrix, get vocabulary distribution.

**How:**
```python
# At layer L, for the answer token position:
residual = cache["resid_post", L][:, -1, :]  # [batch, d_model]
logits = residual @ model.W_U + model.b_U     # [batch, vocab_size]
probs = torch.softmax(logits, dim=-1)

# Track probability of specific tokens across layers:
p_initial_value = probs[:, token_id_of_initial_value]
p_latest_value = probs[:, token_id_of_latest_value]
```

**What we look for:**
- At which layer does the **correct answer** first appear with high probability?
- At which layer does it get **suppressed** (for failure cases)?
- For RI failures: does initial value appear early, then get overwritten by later update?
- For PI failures: does initial value appear and persist, blocking final value?

**This is the money analysis.** If the correct answer rises in probability at early layers then falls at later layers, that's direct evidence of interference happening *within the forward pass*.

### Analysis 3: Activation Patching (causal evidence)

**What:** Determine which (layer, position) is **causally responsible** for correct vs. incorrect retrieval.

**How — clean vs. corrupted runs:**
1. **Clean run:** No interference ("art: impressionism" → "first value of art was:"). Model gets it right.
2. **Corrupted run:** Full interference (N updates). Model may get it wrong.
3. **Patching:** Take clean activation at (layer L, position P), inject into corrupted run. Measure accuracy recovery.

```python
# Pseudocode for activation patching
def patch_and_measure(model, clean_tokens, corrupt_tokens, layer, position):
    # Get clean activations
    _, clean_cache = model.run_with_cache(clean_tokens)
    clean_activation = clean_cache["resid_post", layer][:, position, :]

    # Run corrupted with hook that patches in clean activation
    def hook_fn(activation, hook):
        activation[:, position, :] = clean_activation
        return activation

    patched_logits = model.run_with_hooks(
        corrupt_tokens,
        fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]
    )
    return patched_logits[:, -1, :]  # logits at answer position
```

**What we look for:**
- For RI: which layer/position restores access to initial value?
- For PI: which layer/position breaks initial value's dominance?
- Are these **different layers**? → Direct evidence for dual-process

### Analysis 4: Probing Classifiers (representation analysis)

**What:** Train linear classifiers on activations to check if information is *present but inaccessible* vs. *actually erased*.

**How:**
```python
# At each layer, collect residual stream at answer position
# Labels: correct initial value, correct final value
# Train: linear probe to predict initial value from activations
# Train: linear probe to predict final value from activations

from sklearn.linear_model import LogisticRegression

# Collect data across many examples
X_layer_L = []  # activations at layer L, answer position
y_initial = []   # label: initial value token id
y_final = []     # label: final value token id

for prompt in dataset:
    _, cache = model.run_with_cache(prompt.tokens)
    X_layer_L.append(cache["resid_post", L][:, -1, :].cpu().numpy())
    y_initial.append(prompt.initial_value_token)
    y_final.append(prompt.final_value_token)

probe_initial = LogisticRegression().fit(X_layer_L, y_initial)
probe_final = LogisticRegression().fit(X_layer_L, y_final)
```

**What we look for:**
- For RI failures: Is initial value still **linearly decodable** even when model outputs wrong answer?
  - If yes → information present but retrieval fails (passive failure, matches error taxonomy)
- For PI failures: Is final value linearly decodable even when model outputs initial value?
  - If yes → information present but attention can't access it (active suppression)

Directly tests "passive vs. active" failure distinction with representation-level evidence.

---

## Data Flow Pipeline

```
Dataset (simplified AB-AC pairs, 5-10 categories, N updates)
    ↓
Tokenizer (convert to token IDs, record position of each value token)
    ↓
Model forward pass (TransformerLens hooks capture everything)
    ↓
Cache: attention patterns, residual streams, MLP outputs per layer
    ↓
Analysis:
  ├── Attention heatmaps (per head, per layer)
  ├── Logit lens trajectories (correct vs incorrect answer probability across layers)
  ├── Activation patching grid (layer × position → accuracy recovery)
  └── Probing classifiers (linear decodability per layer)
    ↓
Plots & results
```

---

## Token Position Tracking

Critical bookkeeping: we must know *exactly* which token positions correspond to which values.

```python
@dataclass
class TokenizedPrompt:
    tokens: torch.Tensor              # Full tokenized sequence
    answer_position: int              # Where model generates answer
    initial_value_positions: dict     # {category: [token_pos, ...]}
    update_value_positions: dict      # {category: {update_idx: [token_pos, ...]}}
    final_value_positions: dict       # {category: [token_pos, ...]}
    category_key_positions: dict      # {category: [all positions where key appears]}
    initial_value_token: int          # Token ID of correct initial value
    final_value_token: int            # Token ID of correct final value
```

This mapping lets us:
- Index attention patterns to see if the model looks at the right positions
- Color-code attention heatmaps by token type
- Target activation patching at specific value positions

---

## Priority Ordering

1. **Logit lens** — fastest, most informative, tells us what the model "believes" at each layer
2. **Attention patterns** — easy to implement, good visuals for paper
3. **Activation patching** — harder, needs multiple forward passes, but gives causal evidence
4. **Probing classifiers** — needs training loop, but answers the "present vs. erased" question

---

## Dependencies

```
transformer_lens    # Core: activation access, hooks, caching
torch               # Already installed (2.8.0 with MPS)
einops              # TransformerLens dependency
circuitsvis         # Attention visualization (optional)
plotly              # Interactive plots (optional)
matplotlib          # Static plots
scikit-learn        # Probing classifiers
```

---

## File Structure (planned)

```
mechanistic_probing/
├── models/
│   └── model_loader.py          # Load Pythia/GPT-2 via TransformerLens
├── data/
│   └── interference_dataset.py  # Generate simplified AB-AC prompts for completion models
├── analysis/
│   ├── logit_lens.py            # Layer-by-layer vocabulary projection
│   ├── attention_patterns.py    # Attention weight extraction and visualization
│   ├── activation_patching.py   # Clean vs corrupted patching
│   └── probing.py               # Linear probe training and evaluation
├── utils/
│   └── token_tracking.py        # Token position bookkeeping
├── experiments/
│   ├── run_basic_interference.py    # Behavioral: does the model show PI > RI?
│   ├── run_logit_lens.py            # Logit lens across layers
│   ├── run_attention_analysis.py    # Attention pattern extraction
│   └── run_activation_patching.py   # Causal tracing
├── notebooks/
│   └── visualize_results.ipynb      # Interactive visualization
└── results/
    └── (output plots and data)
```
