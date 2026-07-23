"""Example builder + streaming train generator + frozen eval set generator.

Sequence format (see ../../01_baseline_step_ood/setup.md):
  <BOS> <Key><Value> x K*N (interleaved)  <QUERY> <Key> <Step> <Value> <EOS>

Loss is computed only on the last two positions (predicting VALUE, then EOS) —
the K*N context tokens are i.i.d. random and unpredictable, so training on them
would just inject noise.

DIFFERS FROM experiment 01/02/03's data_gen.py in exactly one place:
`stream_training_batches` now samples query_step independently PER EXAMPLE within
a batch, instead of once for the whole batch. Sequence length depends only on
(K, N) — the query step and query key are always exactly one token each,
regardless of which step/key they represent — so mixing steps within a batch that
shares (K, N) doesn't break the no-padding batching. Before this change, every
micro-batch was homogeneous in role (all-first, all-last, or all-the-same-
intermediate-step); now every micro-batch has natural first/last/intermediate
mixing internally, so gradient diversity doesn't depend entirely on
accum_steps/DDP-world-size the way it used to. Kept as a separate copy (not a
shared-file edit) so experiments 01/02/03 stay exactly reproducible as originally
run.
"""

import random

import torch

import vocab
from grid import K_VALUES, N_VALUES, allowed_steps_for_training


def build_example(rng: random.Random, k: int, n: int, query_step: int):
    """Build one full example: sample active keys + histories, interleave, attach query.

    query_key is always chosen uniformly among the k active keys.
    Returns (full_token_ids, meta_dict).
    """
    active_keys = rng.sample(range(vocab.NUM_KEYS), k)
    query_key = rng.choice(active_keys)

    # per-key value history, i.i.d. with replacement
    histories = {key: [rng.randrange(vocab.NUM_VALUES) for _ in range(n)] for key in active_keys}

    # interleave: repeatedly pop the next event from a uniformly-chosen key that still has events left
    remaining = {key: n for key in active_keys}
    context_ids = []
    while remaining:
        key = rng.choice(list(remaining.keys()))
        step_idx = n - remaining[key]  # 0-indexed position within this key's own history
        value = histories[key][step_idx]
        context_ids.append(vocab.key_id(key))
        context_ids.append(vocab.value_id(value))
        remaining[key] -= 1
        if remaining[key] == 0:
            del remaining[key]

    target_value = histories[query_key][query_step - 1]
    duplicate_flag = any(
        v == target_value for i, v in enumerate(histories[query_key]) if i != query_step - 1
    )

    full_ids = (
        [vocab.BOS]
        + context_ids
        + [vocab.QUERY, vocab.key_id(query_key), vocab.step_id(query_step)]
        + [vocab.value_id(target_value), vocab.EOS]
    )
    meta = {
        "k": k,
        "n": n,
        "query_key": query_key,
        "query_step": query_step,
        "is_first": query_step == 1,
        "is_last": query_step == n,
        "duplicate": duplicate_flag,
    }
    return full_ids, meta


def stream_training_examples(seed: int, held_out_cells):
    """Infinite deterministic generator of training examples (flat/uniform over 36 cells)."""
    rng = random.Random(seed)
    while True:
        k = rng.choice(K_VALUES)
        n = rng.choice(N_VALUES)
        step = rng.choice(allowed_steps_for_training(n, held_out_cells))
        yield build_example(rng, k, n, step)


def build_frozen_eval_set(seed: int, triples, per_triple: int):
    """Generate a fixed list of examples for a list of (k,n,step) triples, per_triple each."""
    rng = random.Random(seed)
    examples = []
    for k, n, step in triples:
        for _ in range(per_triple):
            examples.append(build_example(rng, k, n, step))
    return examples


def stream_training_batches(seed: int, held_out_cells, batch_size: int):
    """Infinite generator of (examples, k, n) batches.

    One (k,n) pair per batch (still required for padding-free fixed-length batching),
    but query_step is now sampled independently per example within the batch — every
    micro-batch gets natural role diversity (first/last/intermediate all mixed in),
    which experiments 01/02/03's version didn't have (they fixed step per whole batch).
    """
    rng = random.Random(seed)
    while True:
        k = rng.choice(K_VALUES)
        n = rng.choice(N_VALUES)
        allowed_steps = allowed_steps_for_training(n, held_out_cells)
        batch = [build_example(rng, k, n, rng.choice(allowed_steps)) for _ in range(batch_size)]
        yield batch, k, n


def pack_split(examples):
    """Group a flat list of (full_ids, meta) into per-(k,n) tensors for eval."""
    groups = {}
    for full_ids, meta in examples:
        key = (meta["k"], meta["n"])
        groups.setdefault(key, {"input_ids": [], "meta": []})
        groups[key]["input_ids"].append(full_ids)
        groups[key]["meta"].append(meta)
    packed = {}
    for (k, n), g in groups.items():
        packed[(k, n)] = {"input_ids": torch.tensor(g["input_ids"], dtype=torch.long), "meta": g["meta"]}
    return packed
