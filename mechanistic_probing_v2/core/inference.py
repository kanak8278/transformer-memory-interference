"""
Inference utilities for HuggingFace models.

Provides batched generation with OOM fallback and cache management.
Used by behavioral sweep scripts and any experiment that needs
HuggingFace model.generate().

Usage:
    from core.inference import run_batch
    from core.model_loader import load_model_hf, clear_accelerator_cache

    model, tokenizer, info = load_model_hf("Qwen/Qwen2.5-0.5B-Instruct")
    answers = run_batch(model, tokenizer, ["prompt1", "prompt2"],
                        max_new_tokens=20, device=info.device)
"""

import torch


def run_batch(model, tokenizer, prompts, max_new_tokens=20, device="cuda",
              max_length=2048):
    """Run a batch of prompts through the model.

    Uses left-padding for batched generation and greedy decoding.

    Args:
        model: HuggingFace CausalLM model.
        tokenizer: HuggingFace tokenizer.
        prompts: List of prompt strings.
        max_new_tokens: Max tokens to generate per prompt.
        device: Device string (cuda:0, mps, cpu).
        max_length: Max input length for truncation.

    Returns:
        List of generated answer strings (one per prompt).
    """
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
        )

    answers = []
    for i in range(len(prompts)):
        prompt_len = inputs["attention_mask"][i].sum().item()
        new_ids = gen_ids[i, prompt_len:]
        answer = tokenizer.decode(
            new_ids, skip_special_tokens=True
        ).strip().split("\n")[0].strip()
        answers.append(answer)

    return answers


def run_batch_with_oom_fallback(model, tokenizer, prompts, max_new_tokens=20,
                                device="cuda", max_length=2048):
    """Run a batch with automatic OOM recovery.

    Tries the full batch first. On OOM, halves batch size until it works.
    Returns (answers, effective_batch_size).

    Args:
        Same as run_batch.

    Returns:
        (answers, effective_batch_size) — answers is a list of strings,
        effective_batch_size is the size that worked (for sticky reduction).
    """
    from .model_loader import clear_accelerator_cache

    attempt_size = len(prompts)
    while True:
        try:
            answers = []
            for start in range(0, len(prompts), attempt_size):
                chunk = prompts[start:start + attempt_size]
                answers.extend(run_batch(
                    model, tokenizer, chunk,
                    max_new_tokens=max_new_tokens,
                    device=device, max_length=max_length,
                ))
            return answers, attempt_size

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                clear_accelerator_cache(device)
                new_size = max(1, attempt_size // 2)
                if new_size < attempt_size:
                    print(f"  OOM at batch_size={attempt_size}, "
                          f"reducing to {new_size}")
                    attempt_size = new_size
                else:
                    # Already at 1, give up
                    print(f"  OOM even at batch_size=1, returning empty")
                    return [""] * len(prompts), 1
            else:
                raise
