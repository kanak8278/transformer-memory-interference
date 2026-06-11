#!/usr/bin/env python3
"""
Generate REAL multi-key interleaved task transcripts for the §1 demo.

Task = the paper's SEMANTIC_MULTI: K keys, each updated N times, interleaved so
no two consecutive updates share a key (greedy placement, matching data_gen.py).
"First/last value of X" is defined by position in the *presented stream* — the
model sees the text and we score against the first/last occurrence of the target
key in that text. We restrict to single-word values (clean chips, unambiguous
scoring). K=3 (target + 2 distractors): FVQ stays high, CVQ degrades with N.

For each scenario × N ∈ {5,10,15,20,30} we run Qwen2.5-3B-Instruct for both FVQ
and CVQ and record the real answer.

Output: web/data/demo_runs.json   (build_data.py curates → examples.json)
tqdm + JSON checkpoint after every scenario (resumable; delete file to rebuild).

Run: .venv/bin/python web/build_demo_data.py
"""
import sys
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mechanistic_probing_v2.core.dataset_configs import _get_pools  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM = ("You are a precise data extraction tool. Output ONLY a single word - "
          "the exact value requested. No other text, no explanation, no punctuation.")
N_VALUES = [5, 10, 15, 20, 30]
K = 3
OUT = Path(__file__).resolve().parent / "data" / "demo_runs.json"

# Preferred target keys (evocative, single-word values). Distractors drawn from
# the remaining single-word-eligible pool, deterministically per scenario.
PREFERRED_TARGETS = [
    "gemstone", "flower species", "bird species", "tree species",
    "wine variety", "sea creature", "martial art", "dance style",
]


def norm(s):
    return s.strip().lower().strip(".,!?;:\"'")


def greedy_interleave(items, rng):
    res, rem = [], items.copy()
    rng.shuffle(rem)
    last = None
    while rem:
        valid = [i for i, x in enumerate(rem) if x["category"] != last]
        if not valid:
            res.extend(rem); break
        i = rng.choice(valid)
        res.append(rem.pop(i)); last = res[-1]["category"]
    return res


@torch.no_grad()
def ask(model, tok, device, stream, target, qw):
    txt = "\n".join(f"{it['category']}: {it['value']}" for it in stream)
    msg = [{"role": "system", "content": SYSTEM},
           {"role": "user", "content":
            f"Read the following key-value stream. Each key gets updated multiple times.\n\n"
            f"{txt}\n\nWhat was the {qw} value of {target}?"}]
    p = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inp = tok(p, return_tensors="pt").to(device)
    o = model.generate(**inp, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0].strip()


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if device == "mps" else torch.float32,
        trust_remote_code=True).to(device).eval()

    pools = _get_pools("SEMANTIC_MULTI")
    sw = {c: [v for v in vs if " " not in v and "-" not in v] for c, vs in pools.items()}
    eligible = sorted([c for c, vs in sw.items() if len(vs) >= max(N_VALUES)])
    targets = [t for t in PREFERRED_TARGETS if t in eligible]
    print(f"{len(eligible)} single-word-eligible categories; {len(targets)} targets.")

    done = {}
    if OUT.exists():
        done = {s["id"]: s for s in json.load(open(OUT)).get("scenarios", [])}
        print(f"Resuming: {len(done)} scenarios done.")

    scenarios = []
    pbar = tqdm(total=len(targets) * len(N_VALUES) * 2, desc="generations")
    for si, target in enumerate(targets):
        sid = f"{si}_{target}".replace(" ", "_")
        if sid in done:
            scenarios.append(done[sid]); pbar.update(len(N_VALUES) * 2); continue
        drng = random.Random(500 + si)
        distractors = drng.sample([c for c in eligible if c != target], K - 1)
        cats = [target] + distractors

        by_n = {}
        for N in N_VALUES:
            rng = random.Random(7000 + si * 100 + N)
            vpc = {c: rng.sample(sw[c], N) for c in cats}
            items = [{"category": c, "value": v} for c in cats for v in vpc[c]]
            stream = greedy_interleave(items, rng)
            # stream-order target values + memory state
            tvals = [it["value"] for it in stream if it["category"] == target]
            memory_now = {}
            for c in cats:
                cv = [it["value"] for it in stream if it["category"] == c]
                memory_now[c] = cv[-1]
            fvq_pred = ask(model, tok, device, stream, target, "first"); pbar.update(1)
            cvq_pred = ask(model, tok, device, stream, target, "last"); pbar.update(1)
            by_n[str(N)] = {
                "stream": stream,
                "target_values": tvals,
                "memory_now": memory_now,
                "fvq": {"expected": tvals[0], "predicted": fvq_pred,
                        "correct": norm(fvq_pred) == norm(tvals[0])},
                "cvq": {"expected": tvals[-1], "predicted": cvq_pred,
                        "correct": norm(cvq_pred) == norm(tvals[-1])},
            }
        scen = {"id": sid, "target_key": target, "keys": cats, "by_n": by_n}
        scenarios.append(scen); done[sid] = scen
        json.dump({"model": "Qwen2.5-3B-Instruct", "K": K, "n_values": N_VALUES,
                   "scenarios": scenarios}, open(OUT, "w"), indent=1)
    pbar.close()
    print(f"\nWrote {OUT} — {len(scenarios)} scenarios.\n")
    for s in scenarios:
        line = " ".join(f"N{n}:F{'+' if s['by_n'][n]['fvq']['correct'] else '-'}"
                         f"C{'+' if s['by_n'][n]['cvq']['correct'] else '-'}"
                         for n in map(str, N_VALUES))
        print(f"  {s['target_key']:20s} {line}")


if __name__ == "__main__":
    main()
