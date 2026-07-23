"""Real GPT-2 (small) architecture, from scratch — no pretrained weights.

Same drop-in interface as experiments 01-04 (`forward(input_ids) -> (logits, attn)`).

Differences from experiment 04's model:
  - dropout = 0.0. Research-backed best practice for FROM-SCRATCH pretraining
    (dropout is for small-data finetuning); exp04 used 0.1 to "match GPT-2's
    published default / exp03's pretrained run", which is the wrong regime here.
  - `configure_optimizers` implements nanoGPT's decoupled weight-decay param
    groups: weight decay only on tensors with dim >= 2 (matmul weights +
    embeddings); biases and LayerNorm params get weight_decay=0.0.

HuggingFace's GPT2LMHeadModel handles weight tying (wte.weight shared with
lm_head.weight; tie_word_embeddings defaults True) — verified in sanity_check.py.

It does NOT, in transformers 5.14.1, apply the GPT-2 residual scaled init: measured
c_proj std was 0.02 rather than the expected 0.02/sqrt(2*n_layer). The GPT-2 paper
(section 2.3) scales residual-projection weights by 1/sqrt(N) with N the number of
residual layers, to keep the residual-stream variance from growing with depth;
nanoGPT does this explicitly. Every prior from-scratch run here (exp02/exp04) used
the raw HF init and silently missed it — a plausible contributor to their plateau.
We therefore apply it ourselves in _apply_residual_scaled_init(), independent of
the transformers version.
"""

import math

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

import vocab

GPT2_SMALL_DEFAULTS = dict(n_positions=1024, n_embd=768, n_layer=12, n_head=12)


class GPT2FromScratch(nn.Module):
    def __init__(self, vocab_size=vocab.VOCAB_SIZE, dropout=0.0):
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            bos_token_id=vocab.BOS,
            eos_token_id=vocab.EOS,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            **GPT2_SMALL_DEFAULTS,
        )
        self.gpt2 = GPT2LMHeadModel(config)
        self._apply_residual_scaled_init(config.n_layer)

    @torch.no_grad()
    def _apply_residual_scaled_init(self, n_layer):
        """GPT-2 paper section 2.3 residual init: scale c_proj (attn output proj +
        MLP down-proj, the two residual-stream writes per block) to std
        0.02/sqrt(2*n_layer). HF 5.14.1 does not do this; we do it explicitly."""
        std = 0.02 / math.sqrt(2 * n_layer)
        n = 0
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                p.normal_(mean=0.0, std=std)
                n += 1
        return n

    def forward(self, input_ids, return_attn=False):
        out = self.gpt2(input_ids=input_ids, output_attentions=return_attn, logits_to_keep=2)
        attn = list(out.attentions) if return_attn else None
        return out.logits, attn

    def configure_optimizers(self, weight_decay, lr, betas, eps, device_type):
        return _configure_optimizers(self, weight_decay, lr, betas, eps, device_type)


def _configure_optimizers(module, weight_decay, lr, betas, eps, device_type):
    """nanoGPT-style AdamW: decay only dim>=2 params, fused on CUDA. Shared by the
    from-scratch and pretrained models."""
    import inspect

    param_dict = {n: p for n, p in module.named_parameters() if p.requires_grad}
    decay = [p for p in param_dict.values() if p.dim() >= 2]
    no_decay = [p for p in param_dict.values() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    n_decay = sum(p.numel() for p in decay)
    n_no_decay = sum(p.numel() for p in no_decay)
    use_fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
    opt = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=use_fused)
    return opt, {"n_decay_tensors": len(decay), "n_decay_params": n_decay,
                 "n_nodecay_tensors": len(no_decay), "n_nodecay_params": n_no_decay,
                 "fused": use_fused}


class GPT2Pretrained(nn.Module):
    """Real pretrained GPT-2 (small), off-the-shelf weights — exp07.

    Same approach as exp03: feed our synthetic token ids (0-50) straight into GPT-2's
    unmodified 50257-row pretrained embedding (no resize/remap). NO residual re-init
    (that would destroy the pretrained weights — the whole point is to keep them).
    dropout stays at GPT-2's shipped config value.
    """

    def __init__(self):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

    def forward(self, input_ids, return_attn=False):
        out = self.gpt2(input_ids=input_ids, output_attentions=return_attn, logits_to_keep=2)
        attn = list(out.attentions) if return_attn else None
        return out.logits, attn

    def configure_optimizers(self, weight_decay, lr, betas, eps, device_type):
        return _configure_optimizers(self, weight_decay, lr, betas, eps, device_type)
