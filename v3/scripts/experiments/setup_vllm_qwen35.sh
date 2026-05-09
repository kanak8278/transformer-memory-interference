#!/bin/bash
# Setup vLLM environment with Qwen3.5 support.
#
# Qwen3.5 (0.8B, 4B, 9B) uses the new `qwen3_5` architecture, which requires:
#   - transformers from main branch (>= 4.58.0.dev, qwen3_5 merged May 2025)
#   - vLLM with qwen3_5 model support (check registry after upgrade)
#   - flash-attn for efficient inference on L40S
#
# Current environment: Python 3.12, torch 2.11.0, vLLM 0.20.1, transformers 4.57.6
# After this script: transformers from git main, vLLM latest
#
# WARNING: Upgrading transformers may affect transformer_lens and sae_lens.
# Verify existing experiments still work after running this.
#
# Usage:
#   export HF_TOKEN=<your_token>
#   bash setup_vllm_qwen35.sh           # full install
#   bash setup_vllm_qwen35.sh --check   # just verify Qwen3.5 loads
#
# After setup, run experiments with:
#   python stage1_sweep_vllm.py \
#     --model Qwen/Qwen3.5-0.8B,Qwen/Qwen3.5-4B,Qwen/Qwen3.5-9B \
#     --dataset ARBITRARY_SINGLE,SEMANTIC_MULTI --trials 100

set -e
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "============================================================"
echo "  Setup vLLM for Qwen3.5 Support"
echo "  $(date)"
echo "============================================================"

# ── CHECK MODE ───────────────────────────────────────────────────
if [ "$1" = "--check" ]; then
    echo ""
    echo "Checking Qwen3.5 support..."
    python -c "
import os; os.environ.setdefault('HF_TOKEN', '${HF_TOKEN}')
import transformers, vllm
print(f'transformers: {transformers.__version__}')
print(f'vllm: {vllm.__version__}')
from transformers import AutoConfig
for m in ['Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-4B', 'Qwen/Qwen3.5-9B']:
    try:
        cfg = AutoConfig.from_pretrained(m, trust_remote_code=True)
        print(f'  OK: {m} (ctx={cfg.max_position_embeddings}, hidden={cfg.hidden_size})')
    except Exception as e:
        print(f'  FAIL: {m}: {str(e)[:100]}')
" 2>&1 | grep -v "^2026\|^WARNING\|^E0000\|FutureWarning\|^To enable"
    exit 0
fi

# ── STEP 1: Record current versions ──────────────────────────────
echo ""
echo "=== Step 1: Current versions ==="
pip show transformers vllm torch 2>/dev/null | grep -E "^Name:|^Version:"

# ── STEP 2: Upgrade transformers from source ─────────────────────
# qwen3_5 architecture was added to transformers main in May 2025.
# PyPI only has 4.57.6 — needs dev version from git.
echo ""
echo "=== Step 2: Upgrading transformers from main branch ==="
pip install \
    "git+https://github.com/huggingface/transformers.git" \
    --quiet 2>&1 | tail -5

python -c "import transformers; print(f'transformers: {transformers.__version__}')"

# ── STEP 3: Upgrade vLLM if needed ───────────────────────────────
echo ""
echo "=== Step 3: Checking vLLM Qwen3.5 support ==="

QWEN35_WORKS=$(python -c "
import os; os.environ.setdefault('HF_TOKEN', '${HF_TOKEN}')
try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True)
    print('yes')
except:
    print('no')
" 2>/dev/null)

if [ "$QWEN35_WORKS" = "yes" ]; then
    echo "Transformers now supports Qwen3.5. Checking vLLM..."

    # Check if vLLM knows about qwen3_5 arch
    VLLM_OK=$(python -c "
try:
    # vLLM maps HF arch names to its own models
    from vllm.model_executor.models.registry import _MODELS
    keys = list(_MODELS.keys())
    has_qwen35 = any('qwen3_5' in k.lower() or 'Qwen35' in k for k in keys)
    print('yes' if has_qwen35 else 'no')
except:
    print('unknown')
" 2>/dev/null)

    echo "vLLM Qwen3.5 registry: $VLLM_OK"

    if [ "$VLLM_OK" != "yes" ]; then
        echo "vLLM does not yet have qwen3_5 — trying to install latest vLLM..."
        pip install "vllm" --upgrade --quiet 2>&1 | tail -5 || true
        python -c "import vllm; print(f'vllm: {vllm.__version__}')"
    fi
else
    echo "transformers upgrade failed — cannot verify Qwen3.5 support"
fi

# ── STEP 4: Update CONTEXT_LIMITS in model_loader.py ─────────────
echo ""
echo "=== Step 4: Adding Qwen3.5 to CONTEXT_LIMITS ==="

python -c "
from pathlib import Path

model_loader = Path('$REPO_ROOT/mechanistic_probing_v2/core/model_loader.py')
content = model_loader.read_text()

additions = '''
    # Qwen3.5 series (qwen3_5 architecture, May 2025)
    \"Qwen/Qwen3.5-0.8B\":               32_768,
    \"Qwen/Qwen3.5-4B\":                 32_768,
    \"Qwen/Qwen3.5-9B\":                 32_768,
    \"Qwen/Qwen3.5-0.8B-Base\":          32_768,
    \"Qwen/Qwen3.5-4B-Base\":            32_768,
    \"Qwen/Qwen3.5-9B-Base\":            32_768,'''

if 'Qwen3.5' in content:
    print('Qwen3.5 already in CONTEXT_LIMITS — no change needed')
else:
    # Insert before closing brace of CONTEXT_LIMITS
    # Find the last Qwen entry and insert after it
    import re
    match = re.search(r'(    \"Qwen/Qwen2\.5-3B-Instruct\".*?\n)', content)
    if match:
        new_content = content.replace(
            match.group(0),
            match.group(0) + additions + '\n'
        )
        model_loader.write_text(new_content)
        print('Added Qwen3.5 to CONTEXT_LIMITS')
    else:
        print('Could not find insertion point — add manually')
        print('Add to mechanistic_probing_v2/core/model_loader.py CONTEXT_LIMITS:')
        print(additions)
"

# ── STEP 5: Update MODEL_ENGINE_CONFIG in stage1_sweep_vllm.py ───
echo ""
echo "=== Step 5: Adding Qwen3.5 to stage1_sweep_vllm.py ==="

python -c "
from pathlib import Path

sweep = Path('$SCRIPT_DIR/stage1_sweep_vllm.py')
content = sweep.read_text()

additions = '''    # Qwen3.5 series (qwen3_5 arch, bfloat16 preferred)
    \"Qwen/Qwen3.5-0.8B\":               (0.95, 65536, 8192, \"bfloat16\"),
    \"Qwen/Qwen3.5-4B\":                 (0.92, 16384, 8192, \"bfloat16\"),
    \"Qwen/Qwen3.5-9B\":                 (0.90, 16384, 8192, \"bfloat16\"),
    \"Qwen/Qwen3.5-0.8B-Base\":          (0.95, 65536, 8192, \"bfloat16\"),
    \"Qwen/Qwen3.5-4B-Base\":            (0.92, 16384, 8192, \"bfloat16\"),
    \"Qwen/Qwen3.5-9B-Base\":            (0.90, 16384, 8192, \"bfloat16\"),'''

if 'Qwen3.5' in content:
    print('Qwen3.5 already in MODEL_ENGINE_CONFIG — no change needed')
else:
    # Insert before DEFAULT_ENGINE_CONFIG line
    new_content = content.replace(
        'DEFAULT_ENGINE_CONFIG =',
        additions + '\n}\nDEFAULT_ENGINE_CONFIG ='
    )
    # Handle double-closing brace issue
    new_content = new_content.replace('}\nDEFAULT_ENGINE_CONFIG', 'DEFAULT_ENGINE_CONFIG')

    # Better: insert before the closing } of MODEL_ENGINE_CONFIG
    import re
    # Find the last entry in MODEL_ENGINE_CONFIG and add after it
    match = re.search(r'(    \"google/gemma-3-4b-it\".*?\"bfloat16\"\),\n)', content)
    if match:
        new_content = content.replace(
            match.group(0),
            match.group(0) + additions + '\n'
        )
        sweep.write_text(new_content)
        print('Added Qwen3.5 to MODEL_ENGINE_CONFIG')
    else:
        print('Could not find insertion point — check manually')
        print('Add to MODEL_ENGINE_CONFIG in stage1_sweep_vllm.py:')
        print(additions)
"

# ── STEP 6: Final verification ────────────────────────────────────
echo ""
echo "=== Step 6: Final verification ==="
python -c "
import os; os.environ.setdefault('HF_TOKEN', '${HF_TOKEN}')
import transformers, vllm
print(f'transformers: {transformers.__version__}')
print(f'vllm: {vllm.__version__}')

from transformers import AutoConfig, AutoTokenizer
print()
print('Model configs:')
models = ['Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-4B', 'Qwen/Qwen3.5-9B']
for m in models:
    try:
        cfg = AutoConfig.from_pretrained(m, trust_remote_code=True)
        ctx = getattr(cfg, 'max_position_embeddings', '?')
        hidden = getattr(cfg, 'hidden_size', '?')
        layers = getattr(cfg, 'num_hidden_layers', '?')
        arch = getattr(cfg, 'model_type', '?')
        print(f'  OK: {m}')
        print(f'      arch={arch}, ctx={ctx}, hidden={hidden}, layers={layers}')
    except Exception as e:
        print(f'  FAIL: {m}: {str(e)[:100]}')

print()
print('Tokenizer + chat template:')
try:
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True)
    has_chat = tok.chat_template is not None
    print(f'  chat_template: {has_chat}')
    if has_chat:
        sample = tok.apply_chat_template(
            [{'role': 'user', 'content': 'test'}],
            tokenize=False, add_generation_prompt=True)
        print(f'  template preview: {sample[:100]}')
except Exception as e:
    print(f'  FAIL: {e}')
" 2>&1 | grep -v "^2026\|^WARNING\|^E0000\|FutureWarning\|^To enable"

echo ""
echo "============================================================"
echo "  Setup complete. $(date)"
echo ""
echo "  To run Qwen3.5 on Stage 1 sweep:"
echo "    cd $SCRIPT_DIR"
echo "    python stage1_sweep_vllm.py \\"
echo "      --model Qwen/Qwen3.5-0.8B,Qwen/Qwen3.5-4B,Qwen/Qwen3.5-9B \\"
echo "      --dataset ARBITRARY_SINGLE,SEMANTIC_MULTI --trials 100"
echo ""
echo "  Verify Qwen3.5 is supported:"
echo "    bash setup_vllm_qwen35.sh --check"
echo "============================================================"
