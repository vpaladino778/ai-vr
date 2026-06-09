"""Step 1: Extract RGB frames from the source video, preserving its FPS.

Writes frames/rgb/000001.png, 000002.png, ... and a frames/fps.txt file recording
the source FPS so later stages (and the final encode) stay in sync.
"""
import json
import subprocess
import sys
from pathlib import Path

from common import PROJECT_ROOT, get_logger, load_config, resolve_binary, ensure_dir, StageTimer

log = get_logger("01_extract_frames")


def probe_fps(ffprobe_bin, video_path):
    result = subprocess.run(
        [
            ffprobe_bin,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "json",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    rate_str = data["streams"][0]["r_frame_rate"]  # e.g. "30/1" or "24000/1001"
    num, den = rate_str.split("/")
    fps = float(num) / float(den)
    return fps, rate_str


def main():
    cfg = load_config()
    ffmpeg_bin = resolve_binary("ffmpeg", cfg.get("ffmpeg_bin"))
    ffprobe_bin = resolve_binary("ffprobe", cfg.get("ffprobe_bin"))

    source_video = Path(cfg["paths"]["source_video"])
    rgb_dir = Path(cfg["paths"]["rgb_frames"])

    if not source_video.is_file():
        log.error(f"Source video not found: {source_video}")
        sys.exit(1)

    with StageTimer(log, "extract RGB frames"):
        fps, rate_str = probe_fps(ffprobe_bin, source_video)
        log.info(f"Source FPS: {rate_str} ({fps:.3f})")

        ensure_dir(rgb_dir)
        # Clear any stale frames from a previous run.
        for f in rgb_dir.glob("*.png"):
            f.unlink()

        pattern = str(rgb_dir / cfg["frame_pattern"])
        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(source_video),
            "-vsync", "0",
            pattern,
        ]
        log.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        frame_count = len(list(rgb_dir.glob("*.png")))
        log.info(f"Extracted {frame_count} frames to {rgb_dir}")

        # Persist FPS for downstream stages (notably the final encode).
        fps_file = PROJECT_ROOT / "frames" / "fps.txt"
        fps_file.write_text(rate_str)
        log.info(f"Wrote source FPS '{rate_str}' to {fps_file}")


if __name__ == "__main__":
    main()
