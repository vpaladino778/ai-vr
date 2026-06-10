#!/usr/bin/env bash
# Bootstrap script for RunPod pods.
#
# Base image: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
# (torch 2.1.0 + CUDA 11.8 are pre-installed — this script adds everything else)
#
# Usage (first time on a fresh pod):
#   cd /workspace && git clone https://github.com/vpaladino778/ai-vr.git ai-video-generation
#   cd ai-video-generation && bash bootstrap.sh
#
# For fully unattended runs, set HUGGING_FACE_HUB_TOKEN in RunPod pod env vars.
# Re-running is safe: pip and weight download steps are idempotent.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="/workspace/models"

echo "=== [0/6] HuggingFace authentication ==="
# Install huggingface_hub first so huggingface-cli is available regardless of
# whether the rest of pip install has run yet.
pip install -q huggingface_hub

if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "HF token found in environment — skipping interactive login."
else
    echo ""
    echo "No HUGGING_FACE_HUB_TOKEN env var found."
    echo "You need a token with read access to download SVD (gated model)."
    echo "Get one at: https://huggingface.co/settings/tokens"
    echo "Accept the SVD license at: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1"
    echo ""
    huggingface-cli login
fi

echo "=== [1/6] System packages ==="
apt-get update -qq
apt-get install -y --no-install-recommends ffmpeg git-lfs
git lfs install

echo "=== [2/6] Initialise StereoCrafter submodule ==="
cd "$REPO_ROOT"
git submodule update --init --recursive

echo "=== [3/6] Python dependencies ==="
pip install --upgrade pip
pip install -r "$REPO_ROOT/requirements-gpu.txt"

# StereoCrafter's requirements.txt pins torch==2.0.1 which would downgrade the
# base image's torch 2.1.0. Strip torch/torchvision lines before installing.
grep -v -E "^torch(vision)?[>=<! ]|^torch(vision)?$" \
    "$REPO_ROOT/deps/StereoCrafter/requirements.txt" \
    | pip install -r /dev/stdin

echo "=== [4/6] Build Forward-Warp CUDA extension ==="
cd "$REPO_ROOT/deps/StereoCrafter/dependency/Forward-Warp"
bash install.sh
cd "$REPO_ROOT"

echo "=== [5/6] Download model weights (~12 GB, skips already-cached) ==="
echo "Disk available on /workspace:"
df -h /workspace | tail -1
python scripts/download_weights.py --dest "$WEIGHTS_DIR"

echo "=== [6/6] Verify ==="
echo "Weights directory contents:"
du -sh "$WEIGHTS_DIR"/* 2>/dev/null || echo "(empty — download may have failed)"

echo ""
echo "Bootstrap complete."
echo ""
echo "To run the pipeline:"
echo "  python scripts/run_pipeline.py --mode gpu --video /workspace/your_clip.mp4"
