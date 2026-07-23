# Source this before any uv / python / training command in this project.
#   usage:  source synthetic_scratch_training/env.sh
#
# All tool caches, downloaded Pythons, and model downloads live under a SHARED
# cache root at the raka6003 level (also set in ~/.bashrc), so every project
# under raka6003/ reuses one cache instead of filling $HOME (~2.5G).
export RAKA_CACHE_ROOT="${RAKA_CACHE_ROOT:-/rnd_ai_datasets1/projects/raka6003/.cache}"
export UV_CACHE_DIR="$RAKA_CACHE_ROOT/uv"
export UV_PYTHON_INSTALL_DIR="$RAKA_CACHE_ROOT/uv-python"
export XDG_CACHE_HOME="$RAKA_CACHE_ROOT/xdg"
export PIP_CACHE_DIR="$RAKA_CACHE_ROOT/pip"
export HF_HOME="$RAKA_CACHE_ROOT/huggingface"
export TORCH_HOME="$RAKA_CACHE_ROOT/torch"
export TRITON_CACHE_DIR="$RAKA_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$RAKA_CACHE_ROOT/torchinductor"

# The .venv stays local to this project; only the wheel/download caches are shared.
export SST_ROOT="/rnd_ai_datasets1/projects/raka6003/transformer-memory-interference/synthetic_scratch_training"
export UV_PROJECT_ENVIRONMENT="$SST_ROOT/.venv"

mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" \
         "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export PATH="$HOME/.local/bin:$PATH"

# Python's bundled certifi lacks the enterprise root CA, so huggingface_hub /
# requests / uv all fail TLS (UnknownIssuer) even though curl works. Point Python
# at the system CA bundle so model downloads (e.g. pretrained GPT-2 for exp07) work.
if [ -f /etc/pki/tls/certs/ca-bundle.crt ]; then
  export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
  export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
fi
