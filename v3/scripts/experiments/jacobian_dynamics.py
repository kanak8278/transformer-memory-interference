"""
Jacobian-at-Initialization for Training Checkpoints.

A FAST alternative to behavioral sweeps: measures the Jacobian norm
at each training checkpoint WITHOUT running generation. This shows
how the positional influence profile changes during training.

Faster than training_dynamics.py because it just needs forward passes,
not 50 trials of generation.

Usage:
    cd /Users/kanak.raj/workspace/hobby/research_work_ri
    .venv/bin/python v3/scripts/experiments/jacobian_dynamics.py
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_V3_DIR = _SCRIPT_DIR.parent.parent
_PROJECT_ROOT = _V3_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

CHECKPOINT_REPO = "HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints"
FINAL_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

CHECKPOINTS = [
    ("step-125000",  CHECKPOINT_REPO),
    ("step-375000",  CHECKPOINT_REPO),
    ("step-625000",  CHECKPOINT_REPO),
    ("step-875000",  CHECKPOINT_REPO),
    ("step-1125000", CHECKPOINT_REPO),
    ("step-1375000", CHECKPOINT_REPO),
    ("step-1625000", CHECKPOINT_REPO),
    ("step-1875000", CHECKPOINT_REPO),
    ("main",         FINAL_MODEL),   # final instruct
]

SEQ_LEN = 50
N_RANDOM_INPUTS = 10


def compute_jacobian_influence(model, tokenizer, seq_len=50, n_inputs=10):
    """
    Compute ||∂output/∂input_j||_F for each position j.
    Returns array of shape [seq_len] with relative influence per position.
    """
    device = next(model.parameters()).device
    influence_per_pos = np.zeros(seq_len)

    for _ in range(n_inputs):
        # Random input tokens
        vocab_size = model.config.vocab_size
        input_ids = torch.randint(100, vocab_size - 100, (1, seq_len)).to(device)

        # Get embeddings — must be a leaf tensor to get gradient
        with torch.no_grad():
            raw_embeds = model.model.embed_tokens(input_ids).float()

        # Create leaf tensor with grad
        inputs_embeds = raw_embeds.detach().requires_grad_(True)
        inputs_embeds.retain_grad()

        # Forward pass
        out = model(inputs_embeds=inputs_embeds)
        logits = out.logits  # [1, seq_len, vocab]

        # Gradient of last token output wrt all input embeddings
        last_token_logit = logits[0, -1, :].sum()
        last_token_logit.backward()

        # Jacobian norm per position
        if inputs_embeds.grad is not None:
            grad = inputs_embeds.grad[0]  # [seq_len, d_model]
            norms = grad.norm(dim=-1).detach().cpu().numpy()  # [seq_len]
            influence_per_pos += norms
        else:
            print("    Warning: grad is None even with retain_grad()")

        inputs_embeds.grad = None

    influence_per_pos /= n_inputs
    return influence_per_pos


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("=== Jacobian Dynamics: SmolLM2-1.7B ===")
    print(f"Measuring positional influence at {len(CHECKPOINTS)} checkpoints\n")

    save_dir = _V3_DIR / "results" / "training_dynamics"
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.plasma(np.linspace(0, 1, len(CHECKPOINTS)))

    for i, (revision, repo) in enumerate(CHECKPOINTS):
        step = int(revision.split("-")[1]) if revision.startswith("step-") else 2000000
        step_label = f"{step//1000}K" if step < 2000000 else "Final"
        print(f"  [{i+1}/{len(CHECKPOINTS)}] {revision} (step {step_label})...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
            model = AutoModelForCausalLM.from_pretrained(
                repo, revision=revision,
                torch_dtype=torch.float32, device_map=str(device)
            )
            model.eval()
        except Exception as e:
            print(f"    FAILED: {e}")
            continue

        influence = compute_jacobian_influence(model, tokenizer, SEQ_LEN, N_RANDOM_INPUTS)
        rel_influence = influence / (influence.mean() + 1e-8)

        # Key metrics
        x = np.linspace(0, 1, SEQ_LEN)
        first_q = rel_influence[:SEQ_LEN//4].mean()
        last_q = rel_influence[3*SEQ_LEN//4:].mean()
        mid = rel_influence[SEQ_LEN//4:3*SEQ_LEN//4].mean()
        primacy_ratio = first_q / (mid + 1e-8)
        recency_ratio = last_q / (mid + 1e-8)

        print(f"    first_q={first_q:.3f} mid={mid:.3f} last_q={last_q:.3f}")
        print(f"    primacy={primacy_ratio:.2f}x recency={recency_ratio:.2f}x")

        all_results[revision] = {
            "step": step,
            "influence": influence.tolist(),
            "rel_influence": rel_influence.tolist(),
            "first_quarter_mean": float(first_q),
            "last_quarter_mean": float(last_q),
            "middle_mean": float(mid),
            "primacy_ratio": float(primacy_ratio),
            "recency_ratio": float(recency_ratio),
        }

        # Plot
        axes[0].plot(x, rel_influence, color=colors[i], alpha=0.7, linewidth=1.5,
                    label=step_label)
        axes[1].bar(i, primacy_ratio, color=colors[i], alpha=0.8)
        axes[2].bar(i, recency_ratio, color=colors[i], alpha=0.8)

        del model
        torch.mps.empty_cache() if device.type == "mps" else None

    # Formatting
    axes[0].set_xlabel("Relative position (0=first, 1=last)")
    axes[0].set_ylabel("Relative influence")
    axes[0].set_title("Positional influence profile over training", fontweight="bold")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Training checkpoint")
    axes[1].set_ylabel("Primacy ratio (first/middle)")
    axes[1].set_title("Primacy bias across training", fontweight="bold")
    step_labels = [f"{int(r.split('-')[1])//1000}K" if r.startswith("step-") else "Final"
                  for r, _ in CHECKPOINTS if r in all_results]
    axes[1].set_xticks(range(len(step_labels)))
    axes[1].set_xticklabels(step_labels, rotation=45, fontsize=8)
    axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[1].grid(True, alpha=0.3, axis="y")

    axes[2].set_xlabel("Training checkpoint")
    axes[2].set_ylabel("Recency ratio (last/middle)")
    axes[2].set_title("Recency bias across training", fontweight="bold")
    axes[2].set_xticks(range(len(step_labels)))
    axes[2].set_xticklabels(step_labels, rotation=45, fontsize=8)
    axes[2].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle("SmolLM2-1.7B: Positional Bias Evolution During Training",
                fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    fig_path = _V3_DIR / "figures" / "jacobian_dynamics_smollm2.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    # Save results
    json_path = save_dir / "jacobian_dynamics_smollm2.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"Results saved: {json_path}")

    # Summary
    print("\n=== Summary ===")
    print(f"{'Checkpoint':<15} {'Step':>8} {'Primacy':>9} {'Recency':>9}")
    print("-" * 45)
    for rev, r in sorted(all_results.items(), key=lambda x: x[1]["step"]):
        print(f"{rev:<15} {r['step']:>8,} {r['primacy_ratio']:>8.2f}x {r['recency_ratio']:>8.2f}x")


if __name__ == "__main__":
    main()
