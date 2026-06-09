"""Shared helpers for the VR180 MVP pipeline: config loading, frame I/O, logging."""
import glob
import logging
import shutil
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline_config.yaml"


def load_config():
    import os
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    # Resolve every path in cfg["paths"] to an absolute path under PROJECT_ROOT.
    cfg["paths"] = {k: str(PROJECT_ROOT / v) for k, v in cfg["paths"].items()}
    # Allow CLI --video override without editing config on disk.
    override = os.environ.get("PIPELINE_SOURCE_VIDEO")
    if override:
        cfg["paths"]["source_video"] = str(Path(override).resolve())
    return cfg


def get_logger(name):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def list_frames(directory, pattern_suffix=".png"):
    """Return sorted list of frame file paths in a directory."""
    return sorted(Path(directory).glob(f"*{pattern_suffix}"))


def frame_path(directory, index, frame_pattern="%06d.png"):
    return Path(directory) / (frame_pattern % index)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_weights_dir(cfg=None):
    """Return the model weights directory.

    Priority: cfg["weights_dir"] → $WEIGHTS_DIR env var → PROJECT_ROOT/models (local fallback).
    """
    import os
    if cfg and cfg.get("weights_dir"):
        return Path(cfg["weights_dir"])
    env = os.environ.get("WEIGHTS_DIR")
    if env:
        return Path(env)
    return PROJECT_ROOT / "models"


def resolve_binary(name, configured_value):
    """Resolve an ffmpeg/ffprobe binary path.

    Tries, in order: the configured value if it points to an existing file,
    the binary on PATH, then a search under the winget package install dir
    (winget often modifies PATH only for new shells).
    """
    if configured_value and Path(configured_value).is_file():
        return configured_value

    found = shutil.which(configured_value or name)
    if found:
        return found

    winget_root = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    matches = glob.glob(str(winget_root / "Gyan.FFmpeg*" / "*" / "bin" / f"{name}.exe"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not locate '{name}'. Install it (e.g. 'winget install Gyan.FFmpeg') "
        f"or set 'ffmpeg_bin'/'ffprobe_bin' in configs/pipeline_config.yaml to its full path."
    )


class StageTimer:
    """Context manager that logs how long a pipeline stage took."""

    def __init__(self, logger, stage_name):
        self.logger = logger
        self.stage_name = stage_name

    def __enter__(self):
        self.start = time.time()
        self.logger.info(f"=== Starting: {self.stage_name} ===")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        if exc_type is None:
            self.logger.info(f"=== Finished: {self.stage_name} in {elapsed:.1f}s ===")
        else:
            self.logger.info(f"=== Failed: {self.stage_name} after {elapsed:.1f}s ===")
        return False
