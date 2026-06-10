"""Download StereoCrafter model weights to a local directory.

Uses huggingface_hub.snapshot_download — no git-lfs required.
Safe to re-run: skips models that are already fully downloaded.

Usage:
    python scripts/download_weights.py                        # uses config weights_dir
    python scripts/download_weights.py --dest /workspace/models
"""
import argparse
import sys
from pathlib import Path

from common import get_logger, load_config, get_weights_dir

log = get_logger("download_weights")

MODELS = [
    {
        "repo_id": "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        "local_dir_name": "stable-video-diffusion-img2vid-xt-1-1",
        "desc": "SVD img2vid-xt-1-1 (image encoder + VAE) — ~7 GB",
    },
    {
        "repo_id": "tencent/DepthCrafter",
        "local_dir_name": "DepthCrafter",
        "desc": "DepthCrafter (depth estimation UNet) — ~2 GB",
    },
    {
        "repo_id": "TencentARC/StereoCrafter",
        "local_dir_name": "StereoCrafter",
        "desc": "StereoCrafter (stereo inpainting UNet) — ~3 GB",
    },
]


def is_downloaded(dest: Path) -> bool:
    """Rough check: directory exists and contains at least one .safetensors or .bin file."""
    if not dest.is_dir():
        return False
    return any(dest.rglob("*.safetensors")) or any(dest.rglob("*.bin"))


def main():
    parser = argparse.ArgumentParser(description="Download StereoCrafter model weights")
    parser.add_argument("--dest", default=None, help="Destination directory for weights")
    parser.add_argument(
        "--svd-repo",
        default=None,
        help="Override SVD repo ID (e.g. a non-gated community mirror). "
             "Default: stabilityai/stable-video-diffusion-img2vid-xt-1-1",
    )
    args = parser.parse_args()

    if args.svd_repo:
        MODELS[0]["repo_id"] = args.svd_repo
        log.info(f"SVD repo override: {args.svd_repo}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    cfg = load_config()
    dest_root = Path(args.dest) if args.dest else get_weights_dir(cfg)
    dest_root.mkdir(parents=True, exist_ok=True)
    log.info(f"Weights destination: {dest_root}")

    for model in MODELS:
        dest = dest_root / model["local_dir_name"]
        if is_downloaded(dest):
            log.info(f"Already present, skipping: {model['local_dir_name']}")
            continue
        log.info(f"Downloading {model['desc']} ...")
        snapshot_download(
            repo_id=model["repo_id"],
            local_dir=str(dest),
        )
        log.info(f"  -> {dest}")

    log.info("All weights ready.")


if __name__ == "__main__":
    main()
