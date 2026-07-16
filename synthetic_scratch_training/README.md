# Synthetic Scratch Training

Tiny transformers (2-5 layers), trained **from scratch** on a fully synthetic
key-value interference task with a custom closed vocabulary — no pretrained
weights, no natural-language tokens. Distinct from `v3/` and
`lora_intervention/`, which fine-tune/probe real pretrained LLMs (Qwen,
Gemma). This line exists to get controlled, ground-truth mechanistic evidence
(the model's exact training data and capacity are known exactly) for the
theory pillar of the interference project.

## Conventions

- **One subfolder per experiment**, numbered (`01_`, `02_`, ...), named for
  what's distinct about it.
- **Every subfolder has a `setup.md`** — the definitive record of that
  experiment's assumptions, vocabulary, data-generation rules, splits, model
  config, and open/unresolved numbers. If a decision changes the setup
  fundamentally, update `setup.md` (or branch a new numbered subfolder) rather
  than letting the code and the doc drift apart.
- **Code reuse across subfolders is fine** — import/copy from an earlier
  experiment's folder directly. No shared `common/` library until there are
  enough experiments to justify factoring one out.
- **Everything is deterministic.** A single fixed global seed drives dataset
  generation, interleaving order, train/val/test split assignment, held-out
  cell selection, model init, and batch ordering. Splits are generated once
  and saved to disk under `data/` (not regenerated on the fly), so re-running
  analysis or resuming from a checkpoint always sees identical data.
- **Checkpoints are saved under `checkpoints/`** (periodic + best-val) so any
  point in training can be reloaded later and evaluated independently on the
  IID test set and any OOD/held-out test set, without retraining.
- **Results** (metrics, per-cell accuracy, eval outputs) go under `results/`.
