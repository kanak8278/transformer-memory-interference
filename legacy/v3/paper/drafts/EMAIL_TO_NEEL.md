# Email to Neel Nanda

**To:** neelnanda27@gmail.com  
**Subject:** Mechanistic analysis of primacy bias in LLMs — stuck on circuit finding  
**Attachments:** Primacy_Bias_Mechanistic_Evidence_KanakRaj.pdf

---

Hi Neel,

I'm working on a mechanistic study of a behavioral asymmetry in LLMs: models are dramatically better at recalling the *first* value assigned to a key than the *most recent* one, even when the recent value is what's being asked for. The effect holds across 11 models including Mamba SSMs, is present from the earliest pretraining checkpoints (step 40K, ~1% through training), and the logit lens localises the failure to the final 15% of layers — the correct value peaks there and then collapses before output.

The part I'm stuck on: causal ablation finds no single bottleneck head responsible for the suppression. Ablating the top contributing heads *hurts* retrieval rather than helping it — they're contributing to finding the answer, not suppressing it. The mechanism appears distributed across the final layer stack and I don't have a good handle on how to characterise it further.

I've attached a one-pager with the full setup and findings. If you have any thoughts on how to approach distributed suppression circuits — or know of a better experimental handle than head ablation — I'd be very grateful. Happy to send over the full data and code if useful.

Best,  
Kanak Raj  
kanak8278@gmail.com
