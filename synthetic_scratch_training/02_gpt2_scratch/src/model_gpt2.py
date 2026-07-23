"""Real GPT-2 (small) architecture, from scratch — no pretrained weights.

Thin wrapper so this drops into the same eval_utils.py / evaluate.py / dump_val_samples.py
interface experiment 01 already built (`forward(input_ids) -> (logits, attn_maps)`),
without touching that shared code.
"""

import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

import vocab

GPT2_SMALL_DEFAULTS = dict(n_positions=1024, n_embd=768, n_layer=12, n_head=12)


class GPT2FromScratch(nn.Module):
    def __init__(self, vocab_size=vocab.VOCAB_SIZE):
        super().__init__()
        # vocab_size, bos/eos (must point at real vocab entries), and dropout (zeroed to
        # match experiment 01's TinyTransformer, which never used dropout) are overridden.
        # GPT2_SMALL_DEFAULTS is passed explicitly rather than relying on HF's library
        # defaults, so this stays "real GPT-2-small" even if a future transformers version
        # changes its own defaults. from_config, never from_pretrained: random init,
        # trained from scratch, same as experiment 01.
        config = GPT2Config(
            vocab_size=vocab_size,
            bos_token_id=vocab.BOS,
            eos_token_id=vocab.EOS,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            **GPT2_SMALL_DEFAULTS,
        )
        self.gpt2 = GPT2LMHeadModel(config)

    def forward(self, input_ids, return_attn=False):
        # logits_to_keep=2: every caller in this codebase only reads logits[:, -2:, :]
        # (VALUE, EOS). Cheap at this vocab_size (51) but kept consistent with
        # model_gpt2_pretrained.py, where it's load-bearing (50257-token vocab).
        out = self.gpt2(input_ids=input_ids, output_attentions=return_attn, logits_to_keep=2)
        attn = list(out.attentions) if return_attn else None
        return out.logits, attn
