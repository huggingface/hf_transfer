#!/bin/bash
set -ex

# Install NVIDIA driver + CUDA
sudo apt-get update -y
sudo apt-get install -y linux-headers-$(uname -r) build-essential
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update -y
sudo apt-get install -y cuda-drivers cuda-toolkit-12-4

# Set up PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install Python 3.10 deps
sudo apt-get install -y python3-pip python3-dev git curl

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Install PyTorch + vLLM
pip3 install --upgrade pip
pip3 install torch --index-url https://download.pytorch.org/whl/cu124
pip3 install vllm huggingface_hub

# Clone and build hf_transfer
cd ~
git clone -b feat/download-to-memory https://github.com/huggingface/hf_transfer.git
cd hf_transfer
pip3 install -e .

echo "=== Setup complete ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "Reboot needed for NVIDIA driver"
