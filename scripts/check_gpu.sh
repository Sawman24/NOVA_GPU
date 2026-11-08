#!/usr/bin/env bash
# Checks that Docker containers can use the NVIDIA GPUs (needed by the NOVA stack).
# Run on the GPU box:  bash scripts/check_gpu.sh
set -u
echo "== Host driver"
if nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader; then :; else
  echo "nvidia-smi failed: install the NVIDIA driver first."; exit 1; fi
echo; echo "== GPU inside a container"
if docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L; then
  echo; echo "OK: containers can use the GPU. Deploy the stack in Portainer."
else
  cat <<'MSG'

Containers can't see the GPU yet. Install the NVIDIA Container Toolkit (Ubuntu/Debian):

  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker

Then run this check again.
MSG
  exit 1
fi
