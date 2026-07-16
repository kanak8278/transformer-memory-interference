"""Real pretrained GPT-2 (small, 124M) — off-the-shelf weights, no architecture changes.

Feeds our synthetic vocab's token ids (0-50) directly as input_ids into GPT-2's
pretrained embedding table (50257 rows) unmodified — no resizing, no remapping. This is
deliberately the crudest possible test: does GPT-2's pretrained transformer body do
anything useful with token ids that mean something completely different to it (whatever
real BPE tokens happen to sit at ids 0-50) than they do to us? See ../setup.md.
"""

import torch.nn as nn
from transformers import GPT2LMHeadModel


class GPT2Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

    def forward(self, input_ids, return_attn=False):
        out = self.gpt2(input_ids=input_ids, output_attentions=return_attn)
        attn = list(out.attentions) if return_attn else None
        return out.logits, attn
