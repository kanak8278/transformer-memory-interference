"""
Behavioral 3-way sweep: base_plain vs base_block vs lora_plain.

Pure accuracy numbers (NO mechanistics). Per position k (1..N) we ask the
position query and score correctness. k=1 is "first" (FVQ), k=N is "last" (CVQ),
interior k is ordinal "k-th" (IVQ) — so endpoints and intermediates are distinct
queries, exactly as in the reference dense-IVQ study.

SAME AS THE NAIVE-MODEL RUN: Plain prompts, seeds, stream construction (shuffle_nc),
ordinal phrasing, and matching rule are imported verbatim from
lora_intervention/evaluate_ivq.py (seed = hash(("ivq_eval",nk,nu,k,t))). base_plain
here is a method-identical extension of that study. base_block reuses the SAME
seed -> SAME (cats, values) as base_plain (paired), only re-rendered as
[Update j] round blocks with an "in Update k" query.

Metric: Wilson 95% CI with early stopping (min 30, cap 200, HW<=0.07) — the
converged() rule from evaluate.py. Greedy (temp 0), matching is prefix/contains
tolerant (is_correct). Generation is BATCHED (HF, left-padded) for feasibility;
Qwen is not Xet-only at inference, only download is affected.

Conditions run with two model loads: base (-> base_plain + base_block) then
LoRA-merged (-> lora_plain).
"""
import os, gc, sys, json, math, time, random, argparse
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

# Verbatim reuse of the naive-run Plain scheme (importing does NOT pull vLLM —
# evaluate_ivq imports vllm lazily inside VLLMEngine):
from lora_intervention.evaluate_ivq import (
    make_prompt as make_prompt_plain, ordinal, is_correct, SYSTEM_PROMPT,
)
from mechanistic_probing_v2.core.dataset_configs import (
    get_eligible_categories, generate_values_for_trial,
)

DATASET = "ARBITRARY_SINGLE"
MIN_TRIALS, MAX_TRIALS, CI_THRESHOLD, BATCH = 30, 200, 0.07, 16


def wilson_hw(n, k, z=1.96):
    if n == 0:
        return 1.0
    nt = n + z ** 2
    pt = (k + z ** 2 / 2) / nt
    return z * math.sqrt(pt * (1 - pt) / nt)


def make_prompt_block(tokenizer, nk, nu, k, seed):
    """Block rendering of the SAME trial as make_prompt_plain(seed) — identical
    cats/values (rng consumed in the same order: sample -> generate_values), only
    grouped by [Update j] with an 'in Update k' query. Position = round (k-th
    update), target = vals[test_cat][k-1]."""
    rng = random.Random(seed)
    elig = get_eligible_categories(DATASET, min_values=nu)
    cats = rng.sample(elig, nk)
    try:
        vals = generate_values_for_trial(DATASET, cats, nu, rng)
    except ValueError:
        return None, None
    test_cat = cats[seed % nk]
    if k > nu:
        return None, None
    lines = []
    for j in range(1, nu + 1):
        lines.append(f"[Update {j}]")
        for c in cats:
            lines.append(f"  {c}: {vals[c][j - 1]}")
    stream = "\n".join(lines)
    expected = vals[test_cat][k - 1]
    user_txt = ("Read the following key-value stream. Each key is updated multiple "
                f"times, grouped by update round.\n\n{stream}\n\n"
                f"What was the value of {test_cat} in Update {k}?")
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_txt}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return prompt, expected


def _model_class_for(base_id):
    """Gemma-3 ships as a VLM via Auto*; pick the text-only causal head."""
    from transformers import AutoModelForCausalLM
    if "gemma-3" in base_id.lower():
        try:
            from transformers import Gemma3ForCausalLM
            return Gemma3ForCausalLM
        except ImportError:
            pass
    return AutoModelForCausalLM


class HFEngine:
    def __init__(self, base_id, adapter=None):
        import torch
        from transformers import AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        m = _model_class_for(base_id).from_pretrained(
            base_id, torch_dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="sdpa", trust_remote_code=True)
        if adapter:
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
        m.eval()
        self.model = m

    def generate(self, prompts, max_new_tokens=8):
        torch = self.torch
        outs = []
        for i in range(0, len(prompts), BATCH):
            chunk = prompts[i:i + BATCH]
            enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=False).to(self.model.device)
            with torch.no_grad():
                g = self.model.generate(**enc, max_new_tokens=max_new_tokens,
                                        do_sample=False, pad_token_id=self.tok.pad_token_id)
            new = g[:, enc["input_ids"].shape[1]:]
            outs.extend(self.tok.decode(r, skip_special_tokens=True).strip() for r in new)
        return outs

    def close(self):
        del self.model
        gc.collect()
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass


def eval_position(engine, fmt, nk, nu, k):
    """Wilson early-stopping accuracy at one position under one format."""
    builder = make_prompt_block if fmt == "block" else make_prompt_plain
    results = []
    t = 0
    while t < MAX_TRIALS:
        prompts, exps = [], []
        for _ in range(BATCH):
            seed = hash(("ivq_eval", nk, nu, k, t)) % (2 ** 31)  # same scheme as naive run
            t += 1
            p, e = builder(engine.tok, nk, nu, k, seed)
            if p is not None:
                prompts.append(p); exps.append(e)
            if t >= MAX_TRIALS:
                break
        if prompts:
            for pred, e in zip(engine.generate(prompts), exps):
                results.append(1 if is_correct(pred, e) else 0)
        n = len(results)
        if n >= MIN_TRIALS and wilson_hw(n, sum(results)) <= CI_THRESHOLD:
            break
    n, c = len(results), sum(results)
    hw = wilson_hw(n, c)
    acc = c / n if n else 0.0
    return {"k": k, "acc": acc, "n": n, "correct": c,
            "ci_lo": max(0.0, acc - hw), "ci_hi": min(1.0, acc + hw),
            "wilson_hw": hw, "stopped_early": n < MAX_TRIALS}


def positions_for(cell):
    nk, nu, mode = cell["k"], cell["n"], cell["mode"]
    if mode == "endpoints":
        return [1, nu]
    ks = list(range(1, nu + 1))
    if mode == "alternate":
        ks = [k for k in ks if k == 1 or k == nu or k % 2 == 0]  # ends + every-other
    return ks


CELLS = [
    {"k": 5,  "n": 20,  "mode": "dense"},        # within training N
    {"k": 10, "n": 25,  "mode": "dense"},
    {"k": 10, "n": 50,  "mode": "dense"},        # primary divergence test, full res
    {"k": 5,  "n": 50,  "mode": "alternate"},
    {"k": 5,  "n": 75,  "mode": "alternate"},
    {"k": 2,  "n": 30,  "mode": "endpoints"},
    {"k": 20, "n": 20,  "mode": "endpoints"},    # reversal regime
    {"k": 10, "n": 100, "mode": "endpoints"},    # 5x N
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default=str(_ROOT / "lora_intervention" / "checkpoints" / "adapter"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "results.json"; logf = out_dir / "run.log"

    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        with open(logf, "a") as f:
            f.write(line + "\n"); f.flush()

    results = json.loads(res_path.read_text()) if res_path.exists() else {}

    def cell_key(c):
        return f"K{c['k']}N{c['n']}"

    def done(ck, cond):
        return ck in results and cond in results[ck]

    def save():
        res_path.write_text(json.dumps(results, indent=1))

    # base model -> base_plain + base_block ; then LoRA -> lora_plain
    plan = [("base", None, [("base_plain", "plain"), ("base_block", "block")]),
            ("lora", args.adapter, [("lora_plain", "plain")])]

    for tag, adapter, conds in plan:
        if all(done(cell_key(c), cn) for c in CELLS for cn, _ in conds):
            continue
        log(f"loading {tag}...")
        eng = HFEngine(args.base_model, adapter)
        try:
            for c in CELLS:
                ck = cell_key(c); ks = positions_for(c)
                for cond_name, fmt in conds:
                    if done(ck, cond_name):
                        continue
                    t0 = time.time()
                    per_pos = [eval_position(eng, fmt, c["k"], c["n"], k) for k in ks]
                    results.setdefault(ck, {})[cond_name] = {
                        "keys": c["k"], "updates": c["n"], "mode": c["mode"],
                        "positions": per_pos,
                        "fvq_acc": per_pos[0]["acc"], "cvq_acc": per_pos[-1]["acc"],
                    }
                    save()
                    log(f"{ck} {cond_name}: FVQ={per_pos[0]['acc']:.2f} "
                        f"CVQ={per_pos[-1]['acc']:.2f} ({len(ks)} pos, {time.time()-t0:.0f}s)")
        finally:
            eng.close()
    log("DONE")


if __name__ == "__main__":
    main()
