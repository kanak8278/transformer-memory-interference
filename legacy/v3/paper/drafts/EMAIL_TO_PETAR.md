# Email to Petar Veličković

**To:** petarv@google.com  
**Subject:** Follow-up: Mechanistic study of primacy bias in LLMs (LinkedIn conversation)  
**Attachments:** Primacy_Bias_Mechanistic_Evidence_KanakRaj.pdf, LLM___Retrospective_interference.pdf

---

Hi Petar,

I hope you're doing well. We connected briefly on LinkedIn back in March — I shared our paper on interference patterns in LLMs (arxiv.org/abs/2603.00270) and you kindly suggested I reach out by email to continue the conversation.

Since then I've been heads-down running the mechanistic study you pointed me toward. Your framing was exactly right: the combination of poor computational graph choice (Glasses paper) and softmax dispersion at longer inputs maps cleanly onto what we observe behaviorally. The logit lens analysis confirms it — the last value *is* computed correctly at ~85% network depth, then gets outcompeted and collapses in the final layers, precisely as your proofs predict.

I've attached a short one-pager summarising what I've built and where I'm stuck. The highlights:

- **11 models** across transformers and Mamba SSM, unified grid of 650 test conditions (100 trials/cell)
- **Logit lens, probing, causal ablation, Jacobian at init, training dynamics** across 40+ pretraining checkpoints
- The bias is present from **step 40K of pretraining** — essentially fully formed before 1% of training is complete — and persists through all SFT and alignment stages
- Mamba (no attention) shows the same pattern, which combined with your bidirectionality insight suggests the culprit is the autoregressive computational graph, not attention specifically

The open questions I'm most stuck on are in the document, but the one I'd value your perspective on most is **Q1**: causal ablation shows no single suppression circuit — the mechanism is distributed across the final 15% of layers. Given your work on over-squashing as a graph-theoretic phenomenon, I'm curious whether there's a path-count or spectral characterisation of why the last few layers are where the collapse manifests.

If you have any bandwidth to connect for 30 minutes, I'd genuinely value your guidance — both on interpreting what I have and on what experiments would most sharpen the story. Happy to work around your schedule.

Thank you again for the pointers in March — they set the direction for everything since.

Best,  
Kanak Raj  
kanak8278@gmail.com
