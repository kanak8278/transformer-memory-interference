"""
Jacobian at Initialization: Does PI > RI exist before ANY training?

Computes the input-output Jacobian norm for each token position on
a randomly initialized (untrained) model. If position 0 has higher
Jacobian norm than position N-1, then PI > RI is ARCHITECTURAL, not learned.

This connects to Chowdhury (2603.10123) who showed the primacy tail +
recency delta + dead zone structure exists at initialization.

Usage:
    cd v3
    python jacobian_at_init.py --model Qwen/Qwen2.5-1.5B-Instruct --seq-len 50
"""

import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_V3_SCRIPTS = _SCRIPT_DIR.parent
_REPO_ROOT = _V3_SCRIPTS.parent.parent
for p in [str(_V3_SCRIPTS), str(_REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--seq-len", type=int, default=50)
    p.add_argument("--n-samples", type=int, default=20,
                   help="Number of random input samples to average over")
    p.add_argument("--untrained", action="store_true",
                   help="Use randomly initialized (untrained) model")
    return p.parse_args()


def compute_jacobian_norms(model, tokenizer, seq_len, n_samples, device):
    """Compute Jacobian norm: ||d output / d input_embedding[pos]|| for each position.

    Measures how sensitive the final output is to each input position.
    Higher norm = more influence on output.
    """
    vocab_size = tokenizer.vocab_size
    d_model = getattr(model.config, 'hidden_size', None) or getattr(model.config, 'text_config', model.config).hidden_size

    all_norms = np.zeros((n_samples, seq_len))

    for s in range(n_samples):
        # Random input tokens
        torch.manual_seed(s * 1000)
        input_ids = torch.randint(100, vocab_size - 100, (1, seq_len)).to(device)

        # Get embeddings and make them require grad
        if hasattr(model, "get_input_embeddings"):
            embed_layer = model.get_input_embeddings()
        else:
            embed_layer = model.transformer.wte if hasattr(model, "transformer") else model.model.embed_tokens

        embeddings = embed_layer(input_ids).detach().clone().requires_grad_(True)

        # Forward pass through the model using embeddings directly
        # We need to bypass the embedding layer
        try:
            # Try HuggingFace standard interface
            outputs = model(inputs_embeds=embeddings)
            logits = outputs.logits  # [1, seq_len, vocab_size]
        except Exception as e:
            print(f"  Forward pass failed: {e}")
            continue

        # Compute gradient of final position logit norm w.r.t. each position's embedding
        # metric = ||logits[0, -1, :]||^2 (total energy at final position)
        metric = (logits[0, -1, :] ** 2).sum()

        metric.backward()

        if embeddings.grad is not None:
            # Grad shape: [1, seq_len, d_model]
            # Norm per position: ||grad[0, pos, :]||
            grad_norms = embeddings.grad[0].float().norm(dim=-1).detach().cpu().numpy()
            all_norms[s] = grad_norms
        else:
            print(f"  No gradient for sample {s}")

        model.zero_grad()
        if embeddings.grad is not None:
            embeddings.grad = None

        if (s + 1) % 5 == 0:
            print(f"  [{s+1}/{n_samples}]")

    return all_norms


def main():
    args = parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model} ({'untrained' if args.untrained else 'pretrained'})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.untrained:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        print(f"  Created UNTRAINED model from config")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, trust_remote_code=True
        )
        print(f"  Loaded pretrained model")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.0f}M params on {device}")

    print(f"\nComputing Jacobian norms (seq_len={args.seq_len}, n_samples={args.n_samples})...")
    norms = compute_jacobian_norms(model, tokenizer, args.seq_len, args.n_samples, device)

    # Average across samples
    avg_norms = np.mean(norms, axis=0)
    std_norms = np.std(norms, axis=0)

    # Normalize to relative influence
    total = avg_norms.sum()
    rel_influence = avg_norms / total if total > 0 else avg_norms

    print(f"\n{'='*60}")
    label = "UNTRAINED" if args.untrained else "PRETRAINED"
    print(f"JACOBIAN NORMS — {args.model} ({label})")
    print(f"{'='*60}")

    print(f"\nPosition influence (relative):")
    print(f"  Position 0 (first):  {rel_influence[0]:.4f} ± {std_norms[0]/total:.4f}")
    print(f"  Position {args.seq_len-1} (last):   {rel_influence[-1]:.4f} ± {std_norms[-1]/total:.4f}")
    print(f"  Ratio (first/last):  {rel_influence[0]/max(rel_influence[-1], 1e-10):.2f}×")

    # U-shape analysis
    first_quarter = rel_influence[:args.seq_len//4].mean()
    middle_half = rel_influence[args.seq_len//4:3*args.seq_len//4].mean()
    last_quarter = rel_influence[3*args.seq_len//4:].mean()
    print(f"\n  First quarter avg:   {first_quarter:.4f}")
    print(f"  Middle half avg:     {middle_half:.4f}")
    print(f"  Last quarter avg:    {last_quarter:.4f}")
    print(f"  Primacy ratio (first/middle): {first_quarter/max(middle_half, 1e-10):.2f}×")

    # Is there a U-shape?
    if first_quarter > middle_half and last_quarter > middle_half:
        print(f"\n  U-SHAPE DETECTED: first and last quarters > middle")
        print(f"  Primacy vs recency: {first_quarter/max(last_quarter, 1e-10):.2f}× (primacy/recency)")
    elif first_quarter > middle_half:
        print(f"\n  PRIMACY ONLY: first quarter > middle, no recency spike")
    else:
        print(f"\n  NO CLEAR POSITIONAL BIAS in Jacobian norms")

    # Save
    save_dir = _SCRIPT_DIR / "results" / "jacobian"
    save_dir.mkdir(parents=True, exist_ok=True)
    from mechanistic_probing_v2.core.model_loader import model_short_name
    m_short = model_short_name(args.model)
    tag = "untrained" if args.untrained else "pretrained"
    save_path = save_dir / f"jacobian_{m_short}_{tag}.json"

    output = {
        "model": args.model,
        "mode": tag,
        "seq_len": args.seq_len,
        "n_samples": args.n_samples,
        "avg_norms": avg_norms.tolist(),
        "std_norms": std_norms.tolist(),
        "rel_influence": rel_influence.tolist(),
        "first_influence": float(rel_influence[0]),
        "last_influence": float(rel_influence[-1]),
        "ratio_first_last": float(rel_influence[0] / max(rel_influence[-1], 1e-10)),
    }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
