# EMNLP 2026 / ARR May 2026 Submission Checklist

Submission deadline: **May 26 2026 11:59 UTC-0**. Metadata grace period: **May 28 EoD AoE**.

Legend:
- [x] = done in `main.tex` (cite section)
- [ ] = missing or insufficient (action required)
- [?] = author decision needed (no right answer, just pick)
- N/A = not applicable to this paper

---

## 1. Desk-reject prevention (CFP hard rules)

- [x] **Page limit.** ≤ 8 pages of body content. Current PDF: 23 pages (body 8 + Limitations + Ethics + references + appendix). Body ends with Figure 5 on page 8.
- [x] **Limitations section.** Required, must be titled exactly `Limitations`. Present at L733 (`\section*{Limitations}`).
- [ ] **Anonymisation audit.** Required. **Action items:**
  - `\author{Anonymous}` at L30 — OK.
  - Author email/path strings: grep for `kanak.raj`, working-dir path `/Users/kanak.raj/...` in any committed file before archiving.
  - Software/data uploads: strip Git history (`git archive --format=tar.gz HEAD` or `rm -rf .git` inside copy), then anonymise.
  - Delete `paper/EMNLP_SUBMISSION_CHECKLIST.md` (this file) from the archive — it names the author.
  - No file-hosting links (Dropbox / Google Drive personal). Use OpenReview Software / Data slots (≤ 200 MB each).
- [x] **References pages.** Unlimited. OK.
- [x] **Appendix.** Unlimited, after Limitations. Present.

---

## 2. OpenReview submission form (metadata)

Fill in the OpenReview form with these values.

### 2.1 Required text fields
- [x] **Title:** `Tracked but Suppressed: How LLMs Fail Current-Value Retrieval`
  (matches `main.tex` L34; do NOT type the older "What Pretraining
  Doesn't Learn..." variant into the form)
- [x] **Keywords** (comma-separated):
  `current-value retrieval, in-context update tracking, LoRA, mechanistic interpretability, pretraining dynamics, position-indexed retrieval`
- [x] **TL;DR** (one sentence):
  `Pretrained LLMs fail to retrieve the current value of repeatedly-updated in-context variables; a 0.24%-parameter LoRA recovers the capability by amplifying a partial attention circuit already present in pretraining, and a training-free format change recovers it via a distinct mechanism — the skill is latent, not absent.`
- [x] **Abstract:** copy verbatim from `main.tex` L37–61.

### 2.2 PDF upload
- [ ] Upload `main.pdf` (23 pp, 603 KB after most recent build). **Action:** rebuild once more *immediately before* submitting so the timestamp is fresh and no late edits are missing.

### 2.3 Paper type & research area
- [x] **Paper Type:** Long.
- [x] **Research Area:** `Interpretability and Analysis of Models for NLP`
  - Rationale: §7 (probing, logit lens, attention routing, causal
    ablation) is canonical mechanistic interp; §4 (behavioural
    sweep across 20 models) and §6 (79-checkpoint training-dynamics
    analysis) are also "analysis of models" work. Three of the four
    contribution buckets fit this area squarely.
- [x] **Research Area Keywords:**
  `mechanistic interpretability, probing, causal analysis, behavioural analysis of LLMs, model analysis`
- [x] **Contribution Types** (multi-select; recommended):
  - [x] Model analysis & interpretability
  - [x] NLP engineering experiment
  - [x] Data analysis
  - [x] Publicly available software and/or pre-trained models *(only if you release the code + LoRA weights)*
- [x] **Languages Studied:** English.

### 2.4 Resubmission fields
- [x] **Previous URL:** leave blank — this is a first submission.
- [x] **Explanation of Revisions PDF:** N/A.
- [x] **Reassignment Request Area Chair / Reviewers:** select `This is not a resubmission`.

### 2.5 Self-disclosure & anonymisation
- [x] **Preprint Status:** `There is a non-anonymous preprint (URL specified in the next question).`
  - Reason: `https://arxiv.org/abs/2603.00270` (Chattaraj \& Raj 2026,
    "Transformers Remember First, Forget Last") is a preliminary
    non-archival version of an earlier round of this work
    (behavioural results only; no mechanism, no LoRA, no
    cross-family).
  - Per ACL `formatting.md` (Anonymity section): *"Any preliminary
    non-archival versions of submitted papers should be listed in the
    submission form **but not in the review version of the paper**."*
    So we declare on the form here; we do NOT cite the arXiv paper
    in `main.tex` / `references.bib` during review. The bibtex entry
    will be added for the camera-ready version after acceptance.
- [x] **Existing Preprints (URL field):** `https://arxiv.org/abs/2603.00270`
- [x] **Preprint Field (release public anonymous version):** `yes`
  — gives the paper visibility during review.
- [x] **EMNLP 2026 AI Reviewing Experiment:** `no` (recommended — opting in lets an LLM influence your reviews).
- [x] **Preferred Venue:** `EMNLP`.

### 2.6 Logistics
- [x] **Visa Needs / Country of Origin:** Yes / `IN`.
- [x] **Consent To Share Data / Submission Details:** Yes (both).
- [x] **License:** `CC BY 4.0`.
- [x] **Blind Submission License Agreement:** `On behalf of all authors, I agree`.

### 2.7 Software / Data uploads
- [ ] **Software archive** (≤ 200 MB `.tgz`/`.zip`): anonymised codebase.
  - Strip path strings (`/Users/kanak.raj/...`) and email addresses.
  - Strip Git history.
  - Delete `EMNLP_SUBMISSION_CHECKLIST.md` and other author-naming docs from the archive.
- [ ] **Data archive** (≤ 200 MB): ARB + SEM vocabulary pools, prompts, grid configs. Anonymise as above.

### 2.8 Author registration
- [ ] After submission, all listed authors must complete the **author registration form** (link appears in the OpenReview author console post-submission).

---

## 3. Responsible NLP Research Checklist — filled

Each item lists the form answer (Yes / No / N/A) and the elaboration text to paste into the form's elaboration field.

### A. For every submission

- **A1 Limitations section** — Form: ☑ "This paper has a limitations section."
  - Section: `\section*{Limitations}` at L733.

- **A2 Potential Risks** — Form: **Yes**
  - Elaboration: *"§Ethical Considerations (L762). The paper documents a deployment-relevant failure mode (LLMs cannot reliably track repeatedly-updated in-context variables) and recommends explicit positional cues or targeted fine-tuning. The risk discussion is brief because the work is on a synthetic controlled task; the dual-use surface is limited."*

### B. Scientific artifacts

- **B Use Or Create Scientific Artifacts** — Form: **Yes**
  - Used: 20 pretrained LLMs (open-weight + proprietary).
  - Created: Arbitrary-Single (ARB) and Semantic-Multi (SEM) vocabulary pools; LoRA adapters (Qwen2.5-3B-Instruct + Gemma-3-4b-it); analysis code.

- **B1 Cite Creators Of Artifacts** — Form: **Yes**
  - Elaboration: *"All models cited in App C 'Model Details' table (L993) with per-family attribution. LoRA cited at first use (hu2021lora); SmolLM2/3 cited (allal2025smollm2, smollm3); Qwen tech report cited (qwen25_techreport); Gemma family attributed via the Gemma terms of use referenced in App C."*

- **B2 Discuss The License For Artifacts** — Form: **Yes**
  - Elaboration: *"App C 'Model licenses' itemised list (L1048), with per-model HuggingFace path + verified license name + URL where the license differs from Apache 2.0. The released LoRA adapter inherits the Qwen Research License from its base model Qwen/Qwen2.5-3B-Instruct (also stated in App C)."*

- **B3 Artifact Use Consistent With Intended Use** — Form: **Yes**
  - Elaboration: *"All open-weight model use is research/non-commercial and consistent with each model's license (App C). Proprietary model evaluation is via official APIs (OpenAI, Anthropic, Google) and consistent with each provider's terms of service. The ARB and SEM datasets we create are released for non-commercial research use, consistent with our LoRA's inherited license."*

- **B4 Data Contains PII Or Offensive Content** — Form: **Yes**
  - Elaboration: *"§Ethical Considerations (L762): the ARB and SEM pools are constructed from common-English category vocabulary (synthetic key–value streams); no human subjects, no PII. Vocabulary was inspected for offensive content; none was found."*

- **B5 Documentation Of Artifacts** — Form: **Yes**
  - Elaboration: *"App A 'Dataset Details' (L777) documents ARB (2,300 single-token English words across 46 categories) and SEM (2,403 multi-token category members across the same 46 categories); App B documents the prompt structure, matching rule, and seed scheme. Language: English. Demographic groups represented: N/A (synthetic words)."*

- **B6 Statistics For Data** — Form: **Yes**
  - Elaboration: *"§3 'Task and Setup' (L233) reports the grid (9 × 9 cells, up to 200 trials/cell); App G 'LoRA Training Details' Table 9 (L1378) reports the 18,000 / 2,000 train/val split for the LoRA. No conventional train/dev/test split because we evaluate pretrained models on a fixed grid; LoRA's train/eval split is documented separately."*

### C. Computational experiments

- **C Computational Experiments** — Form: **Yes**

- **C1 Model Size And Budget** — Form: **Yes**
  - Elaboration: *"Parameter counts in App C Table 7 (L993, the 20-model table); GPU-hour budget and hardware in App G 'Compute Budget and Software Stack' (L1469) — ~120 GPU-hours on a single NVIDIA L40S for main open-weight evaluation + Qwen LoRA; NVIDIA L4 for Gemma family-control and cross-family analyses; proprietary models accessed via official APIs."*

- **C2 Experimental Setup And Hyperparameters** — Form: **Yes**
  - Elaboration: *"App G 'LoRA Training Details' (L1346): full hyperparameter table (rank 16, α=32, dropout 0.05, lr 2e-4 cosine, effective batch 64, bf16, 2 epochs for Qwen / 1.42 epochs for Gemma, attention-only Q/K/V/O target modules). Hyperparameters were not searched per task — a single configuration consistent with prior LoRA work (hu2021lora) was used."*

- **C3 Descriptive Statistics** — Form: **Yes**
  - Elaboration: *"Wilson 95% confidence intervals are reported throughout (§3.4 'Metric' L274; appendices). Bootstrap 95% CIs (n_boot=2000, seed=17) are used for mechanism Δs (§7, Table 5 caption; App I, App K). All inference uses greedy decoding (temperature 0), eliminating run-to-run sampling variance."*

- **C4 Parameters For Packages** — Form: **Yes**
  - Elaboration: *"App G 'Software Stack' (L1469): itemised versions for every package — Python 3.12.13, PyTorch 2.11.0 (CUDA 13.0, cuDNN 9.19), transformers 4.57.6, peft 0.19.1, trl 1.4.0, datasets 4.8.5, accelerate 1.13.0, bitsandbytes 0.49.2, vllm 0.21.0, transformer_lens 3.2.1, scikit-learn 1.7.2, numpy 2.3.5, etc. Decoding: greedy (temperature 0), max new tokens 32."*

### D. Human subjects / annotators

- **D Human Subjects** — Form: **No**
- **D1 Instructions Given To Participants** — Form: **N/A**
- **D2 Recruitment And Payment** — Form: **N/A**
- **D3 Data Consent** — Form: **N/A**
- **D4 Ethics Review Board Approval** — Form: **N/A**
  - **Single elaboration for D1–D4** (paste into each elaboration field):
    *"This work uses entirely synthetic data constructed from common-English category vocabulary. No human annotators, crowdworkers, or human-subject participants are involved at any stage."*

### E. AI assistants

- **E AI Assistants In Research Or Writing** — Form: **Yes**

- **E1 Information About Use Of AI Assistants** — Form: **Yes**
  - Elaboration (paste directly into the form text field; the field allows in-form elaboration):
    *"AI coding assistants (Claude Code) were used to help draft analysis scripts, plotting code, LaTeX, and to iterate on prose. All experimental claims and data were inspected and approved by the authors; AI-generated text was edited for accuracy. No experimental results were obtained by AI agents acting autonomously."*

### Final confirmation

- [ ] **Author Submission Checklist** — Form: ☑ Yes.

---

## 4. Remaining action items (ordered by urgency)

What still needs doing before clicking Submit:

1. **Anonymise software / data archive.** (2.7) — `git archive HEAD` + `grep` for path strings + delete this checklist file from the archive.
2. **Author email + path-string sweep.** Grep the codebase once more for `kanak.raj`, `@thomsonreuters.com`, `/Users/kanak.raj/...`.
3. **Rebuild PDF once more** immediately before submitting (timestamp + final compile-check).
4. **Complete author registration form** in OpenReview console *after* submission (mandatory).

**Estimated total time:** ~30 minutes of archive anonymisation + 10 minutes of form filling = under one hour.

---

## 5. Things already handled (not action items)

- ✅ Anonymisation of the LaTeX file itself: `\author{Anonymous}` at L30.
- ✅ Page limit: body fits in 8 pages (Figure 5 placement forced via `[!ht]`).
- ✅ Limitations section: present at L733.
- ✅ Ethical Considerations: present at L762.
- ✅ License declarations for every model used (App C "Model licenses", L1048).
- ✅ Compute Budget: documented (App G, L1469).
- ✅ Software Stack with all versions: documented (App G, L1469).
- ✅ Statistical reporting: Wilson CIs + bootstrap CIs throughout — exemplary.
- ✅ Data documentation: appendix covers grid, splits, prompts, seeds.
- ✅ All required Responsible-NLP-Research checklist sections present in paper.
- ✅ Citation hygiene: refereed publications cited.
- ✅ **Preprint status decided:** `There is a non-anonymous preprint`,
  arXiv `2603.00270` declared on form, NOT cited in paper (per ACL
  rule on preliminary non-archival versions).
- ✅ **Research area decided:** `Interpretability and Analysis of
  Models for NLP` (5 keywords selected).
- ✅ **TL;DR finalised** (paste-ready in §2.1 above).

---

## 6. Section-pointer cheat sheet (for the form)

Quick reference when filling the form fields:

| RNRC item | Paper location |
|---|---|
| A1 Limitations | `§Limitations` (L733) |
| A2 Risks | `§Ethical Considerations` (L762) |
| B1 Cite creators | App C Model Details (L993) + citations throughout |
| B2 Licenses | App C "Model licenses" (L1048) |
| B3 Intended use | §Ethical Considerations + App C |
| B4 PII/offensive | §Ethical Considerations + App A |
| B5 Documentation | App A Dataset Details (L777) |
| B6 Statistics | §3 Setup + App A |
| C1 Compute | App G "Compute Budget and Software Stack" (L1469) |
| C2 Hyperparameters | App G LoRA Training Details Table 9 (L1378) |
| C3 Descriptive stats | Throughout (§3.4, all CI tables) |
| C4 Package versions | App G "Software Stack" itemised (L1469) |
| E1 AI assistants | (paste elaboration directly into form text field) |
