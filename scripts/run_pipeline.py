"""Orchestrator: runs the full VR180 pipeline, stages 01 through 07, in order.

Each stage is a standalone script (also runnable individually).

Usage:
    # Full pipeline, CPU stereo (default)
    python scripts/run_pipeline.py

    # Full pipeline, GPU stereo (StereoCrafter) with a specific input video
    python scripts/run_pipeline.py --mode gpu --video /workspace/myclip.mp4

    # Resume from a specific stage
    python scripts/run_pipeline.py --mode gpu --start 06_compose_vr180_sbs
"""
import argparse
import importlib
import os
import sys
import time

from common import get_logger, load_config

log = get_logger("run_pipeline")

CPU_STAGES = [
    "01_extract_frames",
    "02_run_matting",
    "03_run_depth",
    "04_clean_depth",
    "05_depth_to_stereo",
    "06_compose_vr180_sbs",
    "07_encode",
]

# GPU path skips depth estimation (stages 03/04) — StereoCrafter does its own.
GPU_STAGES = [
    "01_extract_frames",
    "02_run_matting",
    "05_stereocrafter",
    "06_compose_vr180_sbs",
    "07_encode",
]


def main():
    parser = argparse.ArgumentParser(description="VR180 pipeline orchestrator")
    parser.add_argument(
        "--mode", choices=["cpu", "gpu"], default="cpu",
        help="cpu: depth-warp stereo (local). gpu: StereoCrafter diffusion stereo (RunPod).",
    )
    parser.add_argument(
        "--video", default=None,
        help="Override source video path from config (e.g. /workspace/input.mp4).",
    )
    parser.add_argument(
        "--start", default=None,
        help="Resume from this stage name (e.g. 06_compose_vr180_sbs).",
    )
    args = parser.parse_args()

    # Inject video path override via environment variable so all stages see it
    # without needing to modify the config file on disk.
    if args.video:
        os.environ["PIPELINE_SOURCE_VIDEO"] = args.video
        log.info(f"Source video override: {args.video}")

    stages = GPU_STAGES if args.mode == "gpu" else CPU_STAGES
    log.info(f"Mode: {args.mode} — stages: {stages}")

    start_index = 0
    if args.start:
        matches = [i for i, s in enumerate(stages) if s == args.start or s.startswith(args.start)]
        if not matches:
            log.error(f"Unknown stage '{args.start}'. Valid stages for {args.mode} mode: {stages}")
            sys.exit(1)
        start_index = matches[0]

    overall_start = time.time()
    for stage_name in stages[start_index:]:
        module = importlib.import_module(stage_name)
        module.main()

    log.info(f"Pipeline complete in {time.time() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
