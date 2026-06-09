#!/usr/bin/env bash
# Bootstrap script for RunPod pods.
#
# Base image: runpod/pytorch:2.0.1-py3.10-cuda11.8.0-dkms-ubuntu22.04
# (torch 2.0.1 + CUDA 11.8 are pre-installed — this script adds everything else)
#
# Usage (first time on a fresh pod):
#   cd /workspace && git clone <your-repo-url> ai-video-generation
#   cd ai-video-generation && bash bootstrap.sh
#
# Re-running is safe: pip and weight download steps are idempotent.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="/workspace/models"

echo "=== [1/5] System packages ==="
apt-get update -qq
apt-get install -y --no-install-recommends ffmpeg git-lfs
git lfs install

echo "=== [2/5] Initialise StereoCrafter submodule ==="
cd "$REPO_ROOT"
git submodule update --init --recursive

echo "=== [3/5] Python dependencies ==="
pip install --upgrade pip
pip install -r "$REPO_ROOT/requirements-gpu.txt"

# StereoCrafter has its own requirements (some may overlap — pip deduplicates)
pip install -r "$REPO_ROOT/deps/StereoCrafter/requirements.txt"

echo "=== [4/5] Build Forward-Warp CUDA extension ==="
cd "$REPO_ROOT/deps/StereoCrafter/dependency/Forward-Warp"
bash install.sh
cd "$REPO_ROOT"

echo "=== [5/5] Download model weights ==="
python scripts/download_weights.py --dest "$WEIGHTS_DIR"

echo ""
echo "Bootstrap complete."
echo ""
echo "To run the pipeline:"
echo "  python scripts/run_pipeline.py --mode gpu --video /workspace/your_clip.mp4"
