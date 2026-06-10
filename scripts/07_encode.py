"""Step 7: Encode the composed SBS PNG sequence into the final HEVC MP4.

Wraps the ffmpeg command from the handover doc: libx265, yuv420p, hvc1 tag,
source FPS, and either CRF (test quality) or fixed-bitrate mode per config.
"""
import shutil
import subprocess
from pathlib import Path

from common import PROJECT_ROOT, get_logger, load_config, ensure_dir, resolve_binary, StageTimer

log = get_logger("07_encode")


def main():
    cfg = load_config()
    ffmpeg_bin = resolve_binary("ffmpeg", cfg.get("ffmpeg_bin"))

    sbs_dir = Path(cfg["paths"]["sbs_frames"])
    output_video = Path(cfg["paths"]["output_video"])
    ensure_dir(output_video.parent)

    fps_file = PROJECT_ROOT / "frames" / "fps.txt"
    if not fps_file.is_file():
        raise RuntimeError(f"{fps_file} not found. Run 01_extract_frames.py first.")
    rate_str = fps_file.read_text().strip()

    sbs_frames = sorted(sbs_dir.glob("sbs_*.png"))
    if not sbs_frames:
        raise RuntimeError(f"No SBS frames found in {sbs_dir}. Run 06_compose_vr180_sbs.py first.")

    pattern = str(sbs_dir / "sbs_%06d.png")

    cmd = [
        ffmpeg_bin, "-y",
        "-framerate", rate_str,
        "-i", pattern,
        "-c:v", "libx265",
        "-pix_fmt", "yuv420p",
        "-tag:v", "hvc1",
        "-r", rate_str,
    ]

    mode = cfg.get("encode_mode", "crf")
    if mode == "bitrate":
        cmd += [
            "-b:v", str(cfg["encode_bitrate"]),
            "-maxrate", str(cfg["encode_maxrate"]),
            "-bufsize", str(cfg["encode_bufsize"]),
        ]
    else:
        cmd += ["-crf", str(cfg["encode_crf"])]

    cmd += [str(output_video)]

    with StageTimer(log, f"encode H.265 MP4 ({mode} mode)"):
        log.info(f"Encoding {len(sbs_frames)} frames at {rate_str} fps")
        log.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        log.info(f"Wrote {output_video}")

        log.info("Removing SBS frame sequence to free disk ...")
        shutil.rmtree(str(sbs_dir), ignore_errors=True)


if __name__ == "__main__":
    main()
