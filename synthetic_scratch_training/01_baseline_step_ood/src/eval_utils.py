"""Shared evaluation: run a packed (k,n)-grouped split through the model, score VALUE-token
prediction accuracy, and aggregate by cell / step-position / duplicate-flag.

Loss/accuracy is scored only on the VALUE token (position -2 of each sequence) — that's
the actual answer; EOS (position -1) is tracked too but isn't the headline metric.
"""

import torch
import torch.nn.functional as F

EVAL_CHUNK = 512


@torch.no_grad()
def evaluate_split(model, packed, device):
    """packed: dict[(k,n)] -> {"input_ids": LongTensor[N,L], "meta": list[dict]}.

    Returns a flat list of per-example result dicts (meta + correct + loss), for the
    caller to aggregate however it wants (see aggregate_results below).
    """
    model.eval()
    results = []
    for (k, n), group in packed.items():
        input_ids_full = group["input_ids"].to(device)
        meta_list = group["meta"]
        num = input_ids_full.shape[0]
        for start in range(0, num, EVAL_CHUNK):
            chunk = input_ids_full[start:start + EVAL_CHUNK]
            inp = chunk[:, :-1]
            targets = chunk[:, -2:]  # [VALUE, EOS]
            logits, _ = model(inp)
            answer_logits = logits[:, -2:, :]
            loss = F.cross_entropy(
                answer_logits.reshape(-1, answer_logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            ).reshape(chunk.shape[0], 2)
            pred_value = answer_logits[:, 0, :].argmax(dim=-1)
            correct = (pred_value == targets[:, 0])
            for i in range(chunk.shape[0]):
                m = meta_list[start + i]
                results.append({
                    **m,
                    "correct": bool(correct[i].item()),
                    "value_loss": float(loss[i, 0].item()),
                })
    model.train()
    return results


def aggregate_results(results):
    """Overall / per-(k,n) / per-step-role / per-duplicate-flag accuracy."""
    def acc(rs):
        return sum(r["correct"] for r in rs) / len(rs) if rs else None

    agg = {"overall": acc(results), "n": len(results)}

    by_cell = {}
    for r in results:
        by_cell.setdefault((r["k"], r["n"]), []).append(r)
    agg["by_cell"] = {f"K{k}_N{n}": {"acc": acc(rs), "n": len(rs)} for (k, n), rs in by_cell.items()}

    by_role = {"first": [], "last": [], "intermediate": []}
    for r in results:
        if r["is_first"]:
            by_role["first"].append(r)
        elif r["is_last"]:
            by_role["last"].append(r)
        else:
            by_role["intermediate"].append(r)
    agg["by_role"] = {role: {"acc": acc(rs), "n": len(rs)} for role, rs in by_role.items()}

    # Intermediate-only per-step-index curve. A step value that's also a valid N in the
    # grid (e.g. step=4, N in {2,4,6,8,10,12}) is the "last" query when N==step but
    # "intermediate" when N>step — pooling by raw step value alone would mix those two
    # different query types together for every even step. by_role already covers
    # first/last, so by_step here excludes them to give a clean intermediate-only curve.
    by_step = {}
    for r in results:
        if r["is_first"] or r["is_last"]:
            continue
        by_step.setdefault(r["query_step"], []).append(r)
    agg["by_step"] = {str(s): {"acc": acc(rs), "n": len(rs)} for s, rs in sorted(by_step.items())}

    by_dup = {True: [], False: []}
    for r in results:
        by_dup[r["duplicate"]].append(r)
    agg["by_duplicate"] = {str(k): {"acc": acc(rs), "n": len(rs)} for k, rs in by_dup.items()}

    return agg
