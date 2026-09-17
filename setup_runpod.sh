#!/usr/bin/env bash
set -e

cd /root/projects/world_model_from_scratch

python -m venv /root/wm_env
source /root/wm_env/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

apt-get update
apt-get install -y libxcb1 libgl1 libglib2.0-0

pip install -r requirements-chapter-02-gpu.txt

pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

export HF_HOME=/workspace/huggingface

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"