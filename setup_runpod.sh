#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/root/projects/world_model_from_scratch"
VENV_DIR="/root/wm_env"

# Persistent storage
HF_HOME_GLOBAL="/workspace/huggingface"

# Local POSIX storage: avoids symlink problems on the Global Volume
HF_HOME_LOCAL="/root/hf_local"
NLTK_DATA_DIR="/root/nltk_data"

echo "========================================"
echo " World Models - RunPod setup"
echo "========================================"

cd "$REPO_DIR"

# ------------------------------------------------------------
# 1. System dependencies
# ------------------------------------------------------------

echo "[1/7] Installing system dependencies..."

apt-get update
apt-get install -y \
    libxcb1 \
    libgl1 \
    libglib2.0-0

# ------------------------------------------------------------
# 2. Python virtual environment
# ------------------------------------------------------------

echo "[2/7] Preparing Python environment..."

if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

# ------------------------------------------------------------
# 3. Python dependencies
# ------------------------------------------------------------

echo "[3/7] Installing project dependencies..."

pip install -r requirements.txt
pip install -e .
pip install -r requirements-chapter-02-gpu.txt

pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ------------------------------------------------------------
# 4. Hugging Face persistent cache
# ------------------------------------------------------------

echo "[4/7] Configuring Hugging Face..."

mkdir -p "$HF_HOME_GLOBAL"
mkdir -p "$HF_HOME_LOCAL"

export HF_HOME="$HF_HOME_GLOBAL"

# Reuse the authentication token stored on the persistent volume.
# This avoids having to log in again just because HF_HOME changes.
if [ -f "$HF_HOME_GLOBAL/token" ]; then
    cp "$HF_HOME_GLOBAL/token" "$HF_HOME_LOCAL/token"
    chmod 600 "$HF_HOME_LOCAL/token"
fi

# ------------------------------------------------------------
# 5. Fix Cosmos Guardrail / NLTK symlink issue
# ------------------------------------------------------------

echo "[5/7] Preparing local NLTK Guardrail data..."

rm -rf "$NLTK_DATA_DIR"
mkdir -p "$NLTK_DATA_DIR"

# Download only the NLTK resources into a normal local filesystem.
HF_HOME="$HF_HOME_LOCAL" hf download \
    nvidia/Cosmos-1.0-Guardrail \
    --include "blocklist/nltk_data/**"

GUARDRAIL_SNAPSHOT=$(find \
    "$HF_HOME_LOCAL/hub/models--nvidia--Cosmos-1.0-Guardrail/snapshots" \
    -mindepth 1 \
    -maxdepth 1 \
    -type d \
    | head -n 1)

if [ -z "$GUARDRAIL_SNAPSHOT" ]; then
    echo "ERROR: Cosmos Guardrail snapshot was not found."
    exit 1
fi

# -L dereferences symlinks:
# NLTK receives real files instead of Hugging Face symlinks.
cp -aL \
    "$GUARDRAIL_SNAPSHOT/blocklist/nltk_data/." \
    "$NLTK_DATA_DIR/"

export NLTK_DATA="$NLTK_DATA_DIR"

# ------------------------------------------------------------
# 6. Persist runtime environment variables
# ------------------------------------------------------------

echo "[6/7] Configuring future shell sessions..."

ENV_FILE="/root/world_models_env.sh"

cat > "$ENV_FILE" <<EOF
export HF_HOME="$HF_HOME_GLOBAL"
export NLTK_DATA="$NLTK_DATA_DIR"
source "$VENV_DIR/bin/activate"
EOF

# Add once to .bashrc
if ! grep -q "world_models_env.sh" /root/.bashrc; then
    echo 'source /root/world_models_env.sh' >> /root/.bashrc
fi

# ------------------------------------------------------------
# 7. Validation
# ------------------------------------------------------------

echo "[7/7] Running validation tests..."

python - <<'PY'
import torch
import nltk
import world_models

print()
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

tokens = nltk.word_tokenize(
    "The machinery keeps moving and the water keeps flowing."
)

print("NLTK:", tokens)
print("world_models: OK")
PY

echo
echo "========================================"
echo " Setup complete"
echo "========================================"
echo
echo "HF_HOME=$HF_HOME"
echo "NLTK_DATA=$NLTK_DATA"
echo
echo "Environment: $VENV_DIR"
echo "Persistent models: $HF_HOME_GLOBAL"
echo "Local NLTK data: $NLTK_DATA_DIR"