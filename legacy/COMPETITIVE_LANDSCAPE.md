# Competitive Landscape: NeurIPS Submission on Transformer Memory Interference

**Last updated:** 2026-02-20
**Purpose:** Map all related work that competes with, supports, or must be cited in a paper claiming "Transformers show PI > RI due to causal attention architecture."

---

## TIER 1: DIRECT COMPETITORS (Same phenomenon, overlapping claims)

### 1. "Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length"
- **Authors:** Chupei Wang, Jiaqiu Vince Sun
- **Venue:** ICML 2025 Workshop on Long Context Foundation Models (ICFM); also submitted to NeurIPS 2025 (OpenReview)
- **Date:** June 2025
- **Key findings:**
  - Introduces PI-LLM evaluation: sequential key-value updates, query only final values
  - Retrieval accuracy declines log-linearly as interference accumulates
  - Errors dominated by retrieval of prior (interfering) values — classic PI
  - Prompt engineering mitigation yields limited success
  - Claims a "working memory bottleneck beyond mere context access"
- **Relationship to your work:** **THIS IS YOUR MOST DIRECT COMPETITOR.** They study the same PI phenomenon in LLMs with a similar key-value paradigm. However, critical differences:
  - They only study PI (not RI), so they lack the PI vs RI asymmetry finding
  - They have no mechanistic explanation (no logit lens, no attention analysis, no theory)
  - They don't compare across architectures (no Mamba/SSM controls)
  - They don't have the dual-process hypothesis or the scaling dissociation (RI scales with size, PI doesn't)
  - Your 39-model sweep with Cohen's d = 1.73 is far more comprehensive
- **Risk level:** MEDIUM-HIGH. Reviewers will ask "how is this different?" You must cite this and clearly articulate what you add: (a) the RI side + asymmetry, (b) mechanistic theory, (c) architectural controls.

### 2. "Serial Position Effects of Large Language Models"
- **Authors:** Xiaobo Guo, Soroush Vosoughi
- **Venue:** arXiv (June 2024)
- **Date:** June 2024
- **Key findings:**
  - LLMs exhibit serial position effects (primacy + recency biases)
  - Dominant primacy effect, less pronounced recency
  - Tested on ChatGPT, GPT-3.5, GPT-4, Claude-instant-1.2
  - Prompt-based mitigation inconsistent
- **Relationship to your work:** Overlaps on behavioral observation of primacy bias. But they frame it as a cognitive bias in decision-making (e.g., option ordering), not as interference in associative memory. They have no mechanistic analysis.
- **Risk level:** MEDIUM. Different framing but same underlying phenomenon. Must cite.

---

## TIER 2: MECHANISTIC EXPLANATIONS (Explains WHY position biases exist)

### 3. "On the Emergence of Position Bias in Transformers"
- **Authors:** Xinyi Wu, Yifei Wang, Stefanie Jegelka, Ali Jadbabaie
- **Venue:** ICML 2025
- **Date:** February 2025
- **Key findings:**
  - Graph-theoretic framework for multi-layer attention analysis
  - Causal masking inherently biases attention toward EARLIER positions (primacy) because deeper-layer tokens attend to increasingly contextualized early representations
  - RoPE introduces distance-based decay but aggregated across layers produces a trade-off between long-term decay and cumulative early-position importance
  - Validates predictions on real LLMs
- **Relationship to your work:** **STRONGLY SUPPORTS your theory.** Their formal proof that causal masking creates primacy bias is essentially proving one component of your "PI > RI because of causal attention" claim. You should cite this heavily and build on it. Their limitation: they don't connect to interference paradigms or associative memory.
- **Risk level:** LOW (supportive). But reviewers may say "Wu et al. already showed this" — you need to show your contribution is the interference framing + dual-process + mechanistic probing.

### 4. "LayerNorm Induces Recency Bias in Transformer Decoders"
- **Authors:** Junu Kim, Xiao Liu, Zhenghao Lin, Lei Ji, Yeyun Gong, Edward Choi
- **Venue:** arXiv (September 2025, revised January 2026)
- **Date:** September 2025
- **Key findings:**
  - Stacked causal self-attention ALONE biases toward EARLIER tokens
  - But combined with LayerNorm, the bias flips to RECENCY
  - This explains why practical models show recency bias despite causal masking favoring primacy
  - Residual connections and input distribution also modulate this
- **Relationship to your work:** **CRITICAL nuance.** This paper says causal attention = primacy, but LayerNorm = recency. Your PI > RI finding (primacy dominates) may need to reconcile with this. Possible resolution: in your interference paradigm, the conflicting key-value pairs create a different regime where the raw causal masking primacy dominates over the LayerNorm recency effect. This is a potential weakness reviewers could exploit.
- **Risk level:** MEDIUM-HIGH. You need a clear story for why PI > RI despite LayerNorm's recency effect.

### 5. "When Attention Sink Emerges in Language Models: An Empirical View"
- **Authors:** (ICLR 2025 Spotlight)
- **Venue:** ICLR 2025
- **Key findings:**
  - First token receives disproportionate attention ("attention sink")
  - Acts as key biases, not semantic — stores extra attention scores
  - Emerges from softmax normalization (doesn't appear with sigmoid attention)
  - Requires sufficient training data and optimization
- **Relationship to your work:** Attention sinks explain WHY early tokens get disproportionate attention weight, which directly supports the PI > RI asymmetry. The first-token bias from softmax normalization is a mechanistic component of why proactive interference dominates.
- **Risk level:** LOW (supportive). Cite as part of the mechanistic explanation chain.

### 6. "Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization"
- **Authors:** (NeurIPS 2024)
- **Venue:** NeurIPS 2024
- **Key findings:**
  - U-shaped attention bias: beginning and end tokens receive higher attention
  - Middle positions systematically disadvantaged
  - Calibrating positional attention bias can improve middle-context utilization
- **Relationship to your work:** Related but different granularity. Your work is about conflicting key-value recall; theirs is about retrieval from long contexts. The U-shape they find is consistent with your findings about primacy and recency in different conditions.
- **Risk level:** LOW. Standard citation.

### 7. "Lost in the Middle: How Language Models Use Long Contexts"
- **Authors:** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang
- **Venue:** TACL 2024
- **Date:** July 2023 (published 2024)
- **Key findings:**
  - Performance highest when relevant info at beginning or end of context
  - Significant degradation for middle-positioned information
  - Persists across model types, even long-context models
- **Relationship to your work:** Seminal related work. Their "lost in the middle" is the behavioral manifestation of position bias. Your interference paradigm provides a more controlled experimental framework and a mechanistic explanation.
- **Risk level:** LOW (must cite, but not competing).

---

## TIER 3: HOPFIELD/ASSOCIATIVE MEMORY THEORY

### 8. "Provably Optimal Memory Capacity for Modern Hopfield Models: Transformer-Compatible Dense Associative Memories as Spherical Codes"
- **Authors:** Jerry Yao-Chieh Hu, Dennis Wu, Han Liu
- **Venue:** NeurIPS 2024
- **Key findings:**
  - Tight optimal asymptotic memory capacity for modern Hopfield models
  - Connection between KHM memory configurations and spherical codes
  - Exponential storage capacity with pattern dimension
  - Sub-linear algorithm (U-Hop+) achieving optimal capacity
- **Relationship to your work:** Your formal theory section uses the Hopfield-attention equivalence. This paper gives you the capacity bounds you need. If you claim "RI resistance scales with d_model (capacity)", this paper's capacity theory is the foundation.
- **Risk level:** LOW (foundational, supportive). Cite heavily in your theory section.

### 9. "Hopfield Networks is All You Need" (Ramsauer et al., 2021)
- **Venue:** ICLR 2021
- **Key findings:**
  - Transformer attention IS modern Hopfield retrieval
  - Exponential storage capacity
  - One-step convergence
- **Relationship to your work:** The entire Hopfield-attention equivalence your theory rests on. Already in your plan.
- **Risk level:** LOW (foundational).

### 10. "Understanding Factual Recall in Transformers via Associative Memories"
- **Authors:** Eshaan Nichani, Jason D. Lee, Alberto Bietti
- **Venue:** arXiv, December 2024
- **Key findings:**
  - Storage capacities of linear and MLP associative memories scale linearly with parameter count
  - Single attention layer + MLP achieves 100% accuracy when parameters scale linearly with facts
  - Model can trade off between value matrices and MLP as associative memory
  - Sequential learning behavior during training
- **Relationship to your work:** Directly relevant to your theory. The trade-off between attention-based and MLP-based memory explains why RI resistance (attention capacity) scales with model size but PI (attention sharpness/positional) does not.
- **Risk level:** LOW (supportive). Strong theoretical backing for your capacity arguments.

---

## TIER 4: INDUCTION HEADS & MECHANISTIC INTERPRETABILITY

### 11. "What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation"
- **Authors:** Aaditya K. Singh, Ted Moskovitz, Felix Hill, Stephanie C.Y. Chan, Andrew M. Saxe
- **Venue:** arXiv, April 2024
- **Key findings:**
  - Three underlying subcircuits drive induction head formation
  - Subcircuits interact additively
  - Phase change in loss coincides with IH emergence
  - Data-dependent properties influence formation timing
- **Relationship to your work:** Induction heads are the mechanism for in-context key-value recall. Understanding their formation helps explain why PI dominates: the match-and-copy mechanism has an inherent primacy bias from causal masking.
- **Risk level:** LOW. Important context for your mechanistic probing section.

### 12. "Selective Induction Heads: How Transformers Select Causal Structures in Context"
- **Authors:** (ICLR 2025)
- **Venue:** ICLR 2025
- **Key findings:**
  - New circuit type: selective induction heads that choose WHICH position to copy from
  - 3-layer construction: compute transition probabilities, aggregate over past, select correct lag
  - Asymptotically converges to maximum likelihood
  - Can handle variable causal structures
- **Relationship to your work:** If transformers use selective induction heads to choose which value to recall for a given key, the selection mechanism under interference is exactly your PI vs RI paradigm. When multiple values compete, the selection bias toward earlier entries = PI.
- **Risk level:** LOW (supportive mechanistic detail).

### 13. "Knowledge Circuits in Pretrained Transformers"
- **Authors:** Yunzhi Yao, Ningyu Zhang, et al.
- **Venue:** NeurIPS 2024
- **Key findings:**
  - Information heads, relation heads, and MLPs collaboratively encode knowledge
  - Knowledge distributed across attention heads and MLP matrices
  - In GPT-2: knowledge accumulated throughout model; in TinyLLAMA: more concentrated
  - Target knowledge emerges as top token in residual stream after specific MLP layers
- **Relationship to your work:** Provides the circuit-level understanding of how factual recall works. Your logit lens and activation patching experiments are probing the same circuits but under interference conditions.
- **Risk level:** LOW (methodological foundation).

### 14. "Insights into LLM Long-Context Failures: When Transformers Know but Don't Tell"
- **Authors:** (EMNLP 2024 Findings)
- **Key findings:**
  - LLMs encode position of target information but fail to use it in generation
  - "Know but don't tell" phenomenon
- **Relationship to your work:** Supports the idea that the failure in PI is not about encoding but about retrieval/generation under interference.
- **Risk level:** LOW. Nice supporting evidence.

---

## TIER 5: SSM/MAMBA COMPARISONS (Architectural controls)

### 15. "Emergence of Primacy and Recency Effect in Mamba: A Mechanistic Point of View"
- **Authors:** Muhammad Cendekia Airlangga, Hilal AlQuabeh, Munachiso S Nwadike, Kentaro Inui
- **Venue:** arXiv, June 2025
- **Key findings:**
  - Mamba shows U-shaped recall (both primacy AND recency)
  - Primacy: sparse long-term memory channels in SSM
  - Recency: delta-modulated decay (collapses with distractors)
  - Dynamic allocation via semantic regularity
  - Tested on 1.4B and 7B models
- **Relationship to your work:** **CRITICAL for your architectural controls section.** If Mamba shows a DIFFERENT interference profile (U-shaped vs your PI > RI), this confirms your hypothesis that PI > RI is attention-specific. If Mamba shows similar PI > RI, your theory needs revision.
- **Risk level:** MEDIUM. You MUST run the Mamba experiments and reconcile with this paper.

### 16. "Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-smoothing"
- **Authors:** Peihao Wang, Ruisi Cai, et al.
- **Venue:** ICLR 2025
- **Key findings:**
  - SSMs have strong intrinsic recency bias
  - Over-smoothing when deepened (token representations become indistinguishable)
  - Polarization technique (state transition matrices to 0 and 1) mitigates both
- **Relationship to your work:** SSMs' intrinsic RECENCY bias is the opposite of transformers' primacy bias from causal attention. This supports your prediction that SSMs should show RI > PI (the opposite of transformers).
- **Risk level:** LOW (strongly supports your architectural control prediction).

### 17. "The Hidden Attention of Mamba Models"
- **Authors:** (ACL 2025)
- **Venue:** ACL 2025
- **Key findings:**
  - Selective SSMs can be viewed as attention-driven models
  - Enables empirical comparison to transformer attention
  - Mamba shows patterns aligning with attention-based behaviors
- **Relationship to your work:** Useful for framing your Mamba comparison section.
- **Risk level:** LOW.

---

## TIER 6: SUPPORTING METHODOLOGY & TOOLS

### 18. "How to use and interpret activation patching"
- **Authors:** Stefan Heimersheim
- **Venue:** arXiv, April 2024
- **Key findings:** Methodological guide for activation patching in mechanistic interpretability
- **Relationship:** Methodological foundation for your activation patching experiments.

### 19. "LLM-Microscope: Uncovering the Hidden Role of Punctuation in Context Memory of Transformers"
- **Venue:** arXiv, February 2025
- **Key findings:** Open-source toolkit for token-level nonlinearity, contextual memory, logit lens visualization
- **Relationship:** Potential tool/comparison for your mechanistic probing.

### 20. "Entropy-Lens: The Information Signature of Transformer Computations"
- **Venue:** arXiv, February 2025
- **Key findings:** Architecture-agnostic, scalable interpretability framework using entropy
- **Relationship:** Alternative lens for your analysis.

---

## TIER 7: BROADER CONTEXT (Must cite but not competing)

### 21. "Exploiting Primacy Effect to Improve Large Language Models"
- **Venue:** RANLP 2025
- **Key findings:** Fine-tuning amplifies primacy bias; reordering options by semantic similarity exploits primacy to improve MCQA performance
- **Relationship:** Shows primacy bias has practical implications. Different framing (exploitation vs understanding).

### 22. "Linear Recency Bias During Training Improves Transformers"
- **Venue:** COLING 2025
- **Key findings:** Adding linear recency bias during training improves performance
- **Relationship:** Shows the tension between primacy and recency in transformer design.

### 23. "Attention Sorting Combats Recency Bias in Long Context Language Models"
- **Venue:** arXiv (2023)
- **Key findings:** Proposes attention sorting to combat recency bias
- **Relationship:** Mitigation strategy; background.

### 24. "Memory-Augmented Transformers: A Systematic Review"
- **Venue:** arXiv, August 2025
- **Key findings:** Comprehensive survey of memory mechanisms in transformers from neuroscience principles
- **Relationship:** Survey paper; cite for broad context.

### 25. "Equivalence of Context and Parameter Updates in Modern Transformer Blocks"
- **Venue:** arXiv, November 2025
- **Key findings:** Context updates are mathematically equivalent to rank-1 patches to MLP weights
- **Relationship:** Theoretical backing for why in-context key-value updates behave like weight edits, connecting to your interference paradigm.

---

## STRATEGIC ASSESSMENT

### Your Unique Contributions (what no one else has):
1. **PI vs RI asymmetry** — Everyone studies one direction. You have both + the asymmetry finding.
2. **Dual-process hypothesis** — RI and PI uncorrelated (R²=0.044) → different mechanisms.
3. **Scaling dissociation** — RI scales with model size, PI does not. No one else has this.
4. **39-model sweep** — Most comprehensive behavioral study (Cohen's d = 1.73).
5. **Mechanistic theory** — Hopfield formalism explaining WHY PI > RI.
6. **Mechanistic probing** — Logit lens + activation patching under interference (if you deliver).
7. **Architectural controls** — SSM comparison (if you deliver).

### Biggest Risks:
1. **Wang & Sun (2025)** — Direct competitor on PI in LLMs. You must differentiate clearly.
2. **LayerNorm recency paper** — Seems to contradict your primacy claim. Need reconciliation.
3. **Wu et al. ICML 2025** — Already proved causal masking → primacy formally. Your theory section must add beyond this.
4. **Mamba primacy/recency paper** — If their Mamba results conflict with your predictions, your theory is weakened.

### Recommended Actions:
1. Run Mamba experiments ASAP to validate/update architectural control predictions.
2. Address LayerNorm recency bias explicitly in the paper (Section on competing mechanisms).
3. Ensure your Hopfield theory adds beyond Wu et al.'s graph-theoretic proof.
4. Frame contribution clearly: "The first work to (a) document PI/RI asymmetry, (b) provide dual-process mechanistic theory, (c) validate across architectures."

---

## CITATION PRIORITY

---

## BENCHMARK: Top 3 A* Mechanistic Interpretability Papers (Deep Dive)

### Paper 1: IOI Circuit — Wang et al., ICLR 2023

**Task:** Reverse-engineer indirect object identification in GPT-2 Small.

| Dimension | IOI Paper | Our Work |
|-----------|-----------|----------|
| Models | **1 (GPT-2 Small only)** | 4 models, 2 families |
| Heads identified | 26 heads, 7 functional categories | Per-head knockout (25a) on all models |
| Causal method | Path patching + causal scrubbing | Per-head knockout + activation patching |
| Effect sizes | "87% of logit difference explained" | Knockout: 17→86% (Gemma), 44→92% (1.5B) |
| Contradictions reported | Yes — 13% unexplained, acknowledged honestly | Yes — methods disagree (25a vs 25b), distributed vs concentrated varies by model |
| Experiments | 30+ figures | 12+ experiments per model per operating point |

**Key lesson:** They found 26 heads, grouped them into 7 roles, then tested with causal scrubbing. They acknowledged 13% unexplained. **We should aim for similar honesty — explain 80-90%, acknowledge the gap.**

**Their weakness we address:** Single model, no cross-architecture validation.

### Paper 2: ACDC — Conmy et al., NeurIPS 2023

**Task:** Automated circuit discovery via iterative edge pruning.

| Dimension | ACDC | Our Work |
|-----------|------|----------|
| Models | **1 (GPT-2 Small)** | 4 models, 2 families |
| Discovery | Automated (32,000 edges → 68 selected) | Semi-automated (25a knockout sweep) |
| Tasks tested | 4 (IOI, Greater-Than, Docstring, Induction) | 1 task family across multiple interference levels |
| Validation | Edge recovery vs manual ground truth | Cross-operating-point stability (25d) |
| Contradictions | No detailed error analysis reported | Yes — 25b fails on indirect effects, old exp 16 was wrong |

**Key lesson:** Automation is powerful but they still tested on 1 model. **Our multi-model approach is stronger than ACDC's breadth.**

### Paper 3: Circuit Reuse — Merullo et al., ICLR 2024

**Task:** Do the same heads solve different tasks? (IOI → Colored Objects transfer)

| Dimension | Circuit Reuse | Our Work |
|-----------|---------------|----------|
| Models | **1 (GPT-2 Medium)** | 4 models |
| Key result | 78% head overlap between tasks, r=0.69 correlation | Same behavioral pattern across all models, different heads |
| Effect size | +44.1 percentage points from intervention | +69% (Gemma), +48% (1.5B) from single-head knockout |
| Contradictions | "Overlap decreases at larger scales" — reported honestly | Primacy mechanism differs across architectures (A+B vs B) |
| Generalization | Across tasks within 1 model | Across models within 1 task family |

**Key lesson:** They found that circuits are reusable across tasks but break down at scale. **We found that the behavioral phenomenon (PI > RI) is universal but the mechanism differs. Similar honest complexity.**

### What This Means for Our Paper

**We are AHEAD of these papers on:**
1. **Multi-model validation** — all 3 papers used 1 model. We have 4 models, 2 architecture families.
2. **Cross-architecture comparison** — none of them tested Qwen vs Gemma equivalent. We show same behavior, different mechanisms.
3. **Effect sizes** — our knockout effects (+48-69% accuracy from single head) are larger than IOI's +44% from 4-head intervention.
4. **Novel task** — interference paradigm is not a canonical task. Novel contribution.

**We are BEHIND on:**
1. **Complete circuit mapping** — IOI mapped 26 heads into 7 roles. We haven't mapped the full circuit, just identified the top primacy heads.
2. **Number of figures** — IOI had 30+. We need to plan our figure set carefully.
3. **Theoretical framing** — IOI didn't need theory (it was pure reverse-engineering). We claim "PI > RI is architectural" which needs stronger theoretical backing.

**Standards to match:**
- Report effect sizes with exact numbers (not just "significant")
- Acknowledge gaps honestly (IOI's 13% unexplained)
- Show causal evidence, not just attention patterns (we learned this lesson with exp 16 → 25a)
- Cross-validate findings (IOI used 3 metrics: faithfulness, completeness, minimality)

---

## CITATION PRIORITY

---

## BENCHMARK: Latest A* Papers — Deep Experimental Analysis (2024-2026)

### Paper 4: "Retrieval Head Mechanistically Explains Long-Context Factuality" — Wu et al., ICLR 2025 Oral

**Experimental depth:**
- **10 models, 4 architecture families:** LLaMA-2 (7B, 7B-80K, 13B-64K), Mistral (7B-v0.2, Instruct), Mixtral (8×7B), Yi (6B, 6B-200K, 34B-200K), Qwen (1.5-14B, 1.5-14B-Chat)
- **5 datasets:** Needle-in-a-Haystack (custom), Extractive QA (GPT-4 synthesized), MMLU, MuSiQue, GSM8K
- **~600 test instances per model** for needle tests; 20 context lengths × 10 depth positions
- **All pretrained** — no training from scratch
- **Causal evidence:** Pruning retrieval heads → hallucination. Pruning random non-retrieval heads → no effect. Clean control.
- **Key finding:** <5% of heads are retrieval heads, universal across all 10 models
- **Statistical rigor:** Explicit comparison vs random-head baseline

**What we can learn:**
- 10 models across 4 families is the gold standard for universality claims. We have 4 models across 2 families — less but still multi-family.
- They validated on 5 independent benchmarks. We have 1 task family. This is our biggest gap.
- Copy score > 0.1 threshold was simple and worked universally. Our 25a knockout is stronger causally but slower.

### Paper 5: "On the Role of Attention Heads in LLM Safety" (SAHARA) — ICLR 2025 Oral

**Experimental depth:**
- **2 models:** Llama-2-7b-chat, Vicuna-7b-v1.5
- **3 adversarial datasets:** AdvBench, JailbreakBench, Malicious Instruct
- **5 helpfulness benchmarks:** BoolQ, RTE, WinoGrande, ARC Challenge, OpenBookQA
- **Ablation:** Single-head ablation → 16× more harmful outputs. 0.006% parameter modification breaks safety.
- **Key finding:** Safety is extremely sparse — one head matters
- **Contradictions reported:** Models fine-tuned from same base have overlapping but not identical safety heads

**What we can learn:**
- Only 2 models but very thorough ablation + control conditions (helpfulness preservation).
- They tested 8 benchmarks total (3 attack + 5 helpfulness). Breadth of evaluation compensated for model count.
- **Directly analogous to our finding:** their safety head is like our primacy head — sparse, causally verified, huge single-head effect.

### Paper 6: "LLM Circuit Analyses Are Consistent Across Training and Scale" — Lieberum et al., NeurIPS 2024

**Experimental depth:**
- **5 model sizes trained from scratch:** 70M, 160M, 410M, 1.3B, 2.8B
- **300 billion tokens tracked** with multiple checkpoints
- **Key finding:** Task abilities and circuits emerge at similar token counts across scales. Individual heads may shift, but the algorithm is stable.
- **Training dynamics:** Full training curves showing when circuits appear
- **No external datasets** — focused on circuit evolution during training

**What we can learn:**
- They used Pythia checkpoints (same model family we planned). 5 sizes is comprehensive.
- Training dynamics analysis is publishable on its own — we have Pythia checkpoints planned but haven't used them.
- Their finding that "individual heads shift but algorithm is stable" is exactly what we see: different heads across Qwen/Gemma but same PI > RI behavior.

### Experimental Depth Comparison

| Dimension | Wu (ICLR Oral) | SAHARA (ICLR Oral) | Lieberum (NeurIPS) | **Our Work** |
|-----------|----------------|--------------------|--------------------|--------------|
| **Models** | 10 | 2 | 5 | **4** |
| **Families** | 4 | 1 | 1 | **2** |
| **Sizes** | 6B-34B | 7B | 70M-2.8B | **0.5B-3B** |
| **Datasets/tasks** | 5 | 8 (3+5) | 1 (training) | **1** |
| **Trained from scratch?** | No | No | **Yes** | No |
| **Causal method** | Head masking | Head ablation | Head masking | **Per-head knockout + patching** |
| **Statistical rigor** | Random baseline | 5 control benchmarks | Training curves | **200-trial CIs, 3-way validation** |
| **Training dynamics** | No | No | **Yes (full)** | Planned (Pythia) |
| **Contradictions reported** | Implicitly | Overlapping but not identical | Heads shift across scale | **Yes — methods disagree, mechanism differs across architectures** |

### Where We Stand

**Ahead of SOTA:**
- Our 3-way validation (25d) with 200-trial CIs is more rigorous than any single paper's ablation
- Cross-architecture mechanistic comparison (Qwen vs Gemma showing different mechanisms) is novel — no paper does this
- We report contradictions and methodology failures honestly (exp 16 → 25a evolution)

**Behind SOTA:**
- **Single task** — biggest gap. Wu tested 5 benchmarks. We need at least 1 out-of-domain validation.
- **Model sizes are small** (0.5B-3B vs 7B-34B). Acceptable for mechanistic work (Lieberum used 70M-2.8B) but won't impress reviewers who expect big models.
- **No training dynamics yet** — Lieberum made a whole paper on this. We have the Pythia plan but haven't executed.

---

## CITATION PRIORITY

**Must cite (related work section):**
- Wang & Sun 2025 (PI-LLM)
- Liu et al. 2024 (Lost in the Middle)
- Wu et al. 2025 (Position Bias Emergence)
- Kim et al. 2025 (LayerNorm Recency)
- Guo & Vosoughi 2024 (Serial Position Effects)
- Ramsauer et al. 2021 (Hopfield Networks)
- Hu et al. 2024 (Hopfield Capacity, NeurIPS)
- Airlangga et al. 2025 (Mamba Primacy/Recency)
- Wang et al. 2025 (SSM Recency, ICLR)
- ICLR 2025 Attention Sink paper

**Should cite (theory/methods):**
- Nichani et al. 2024 (Factual Recall Associative Memories)
- Singh et al. 2024 (Induction Head Formation)
- ICLR 2025 Selective Induction Heads
- Yao et al. 2024 (Knowledge Circuits, NeurIPS)
- EMNLP 2024 "Know but Don't Tell"

**Nice to cite (broader context):**
- Memory-Augmented Transformers survey
- Equivalence of Context and Parameter Updates
- Attention Sorting / Linear Recency Bias papers
