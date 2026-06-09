"""Stage 5 (GPU path): StereoCrafter diffusion-based stereo synthesis.

Replaces 05_depth_to_stereo.py on RunPod. Calls StereoCrafter's two inference
scripts via subprocess, then post-processes their SBS output into the left/right
RGBA frame format expected by stage 06.

Pipeline:
  frames/rgb/*.png  →  (re-encode to temp MP4)
                    →  depth_splatting_inference.py   (stage 1: depth + warp)
                    →  inpainting_inference.py         (stage 2: fill disocclusions)
                    →  <save_dir>/*_sbs.mp4            (SBS: left=original, right=inpainted)
                    →  split SBS frames into left/right
                    +  frames/alpha/*.png              (alpha matte from stage 02)
                    →  stereo/left_rgba/*.png
                    →  stereo/right_rgba/*.png

Stage 06 is completely unchanged — it receives the same RGBA frame format.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from common import (
    PROJECT_ROOT,
    StageTimer,
    ensure_dir,
    get_logger,
    get_weights_dir,
    list_frames,
    load_config,
    resolve_binary,
)

log = get_logger("05_stereocrafter")

# StereoCrafter lives as a git submodule at deps/StereoCrafter/
STEREOCRAFTER_DIR = PROJECT_ROOT / "deps" / "StereoCrafter"


def _check_stereocrafter():
    if not (STEREOCRAFTER_DIR / "depth_splatting_inference.py").is_file():
        raise RuntimeError(
            f"StereoCrafter not found at {STEREOCRAFTER_DIR}. "
            "Run: git submodule update --init --recursive"
        )


def _frames_to_video(rgb_dir: Path, output_path: Path, fps: str, ffmpeg_bin: str):
    """Re-encode the RGB frame sequence to a temporary MP4 for StereoCrafter input."""
    pattern = str(rgb_dir / "%06d.png")
    cmd = [
        ffmpeg_bin, "-y",
        "-framerate", fps,
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "0",        # lossless to avoid quality loss before diffusion
        str(output_path),
    ]
    log.info(f"Re-encoding frames to temp video: {output_path}")
    subprocess.run(cmd, check=True)


def _run_depth_splatting(input_video: Path, output_video: Path, weights_dir: Path, cfg: dict):
    """Call StereoCrafter's depth_splatting_inference.py via subprocess."""
    svd_path = weights_dir / "stable-video-diffusion-img2vid-xt-1-1"
    depth_crafter_path = weights_dir / "DepthCrafter"

    for p in (svd_path, depth_crafter_path):
        if not p.is_dir():
            raise RuntimeError(f"Weights not found: {p}. Run: python scripts/download_weights.py")

    cmd = [
        sys.executable,
        str(STEREOCRAFTER_DIR / "depth_splatting_inference.py"),
        "--pre_trained_path", str(svd_path),
        "--unet_path", str(depth_crafter_path),
        "--input_video_path", str(input_video),
        "--output_video_path", str(output_video),
        "--max_disp", str(cfg.get("stereocrafter_max_disp", 20)),
    ]
    log.info("Running StereoCrafter stage 1 (depth splatting) ...")
    subprocess.run(cmd, check=True, cwd=str(STEREOCRAFTER_DIR))


def _run_inpainting(input_video: Path, save_dir: Path, weights_dir: Path, cfg: dict):
    """Call StereoCrafter's inpainting_inference.py via subprocess."""
    svd_path = weights_dir / "stable-video-diffusion-img2vid-xt-1-1"
    stereocrafter_path = weights_dir / "StereoCrafter"

    for p in (svd_path, stereocrafter_path):
        if not p.is_dir():
            raise RuntimeError(f"Weights not found: {p}. Run: python scripts/download_weights.py")

    ensure_dir(save_dir)
    cmd = [
        sys.executable,
        str(STEREOCRAFTER_DIR / "inpainting_inference.py"),
        "--pre_trained_path", str(svd_path),
        "--unet_path", str(stereocrafter_path),
        "--input_video_path", str(input_video),
        "--save_dir", str(save_dir),
        "--tile_num", str(cfg.get("stereocrafter_tile_num", 1)),
        "--frames_chunk", str(cfg.get("stereocrafter_frames_chunk", 23)),
    ]
    log.info("Running StereoCrafter stage 2 (inpainting) ...")
    subprocess.run(cmd, check=True, cwd=str(STEREOCRAFTER_DIR))


def _find_sbs_video(save_dir: Path) -> Path:
    """Find the *_sbs.mp4 output from inpainting_inference.py."""
    matches = list(save_dir.glob("*_sbs.mp4"))
    if not matches:
        raise RuntimeError(f"No *_sbs.mp4 found in {save_dir}. Inpainting may have failed.")
    return sorted(matches)[-1]


def _split_sbs_and_apply_alpha(
    sbs_video: Path,
    alpha_dir: Path,
    left_dir: Path,
    right_dir: Path,
    dilate_px: int,
    blur_radius: float,
):
    """Split each SBS frame into left/right halves, apply alpha matte, write RGBA PNGs.

    StereoCrafter's SBS layout: left half = original (depth-splatted) view,
    right half = inpainted stereo view. Both are RGB — we add the alpha matte
    from stage 02 so stage 06 can composite them onto the black VR180 canvas.
    """
    cap = cv2.VideoCapture(str(sbs_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open SBS video: {sbs_video}")

    alpha_frames = list_frames(alpha_dir)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info(f"Splitting {total} SBS frames and applying alpha ...")

    if len(alpha_frames) != total:
        log.warning(
            f"Alpha frame count ({len(alpha_frames)}) != SBS frame count ({total}). "
            "Using whichever is smaller — check that stages 01 and 02 ran on the same source."
        )

    for i in range(1, total + 1):
        ret, sbs_bgr = cap.read()
        if not ret:
            break

        name = f"{i:06d}.png"
        h, w_sbs = sbs_bgr.shape[:2]
        w = w_sbs // 2

        left_bgr = sbs_bgr[:, :w]
        right_bgr = sbs_bgr[:, w:]

        # Load matching alpha matte (stage 02 output)
        alpha_path = alpha_dir / name
        if alpha_path.is_file():
            alpha_u8 = cv2.imread(str(alpha_path), cv2.IMREAD_GRAYSCALE)
            if alpha_u8.shape != (h, w):
                alpha_u8 = cv2.resize(alpha_u8, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            log.warning(f"Alpha frame missing: {alpha_path} — using full-frame opaque mask")
            alpha_u8 = np.full((h, w), 255, dtype=np.uint8)

        alpha_u8 = _soften_alpha(alpha_u8, dilate_px, blur_radius)

        left_rgba = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2BGRA)
        left_rgba[:, :, 3] = alpha_u8
        right_rgba = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2BGRA)
        right_rgba[:, :, 3] = alpha_u8

        cv2.imwrite(str(left_dir / name), left_rgba)
        cv2.imwrite(str(right_dir / name), right_rgba)

        if i % 25 == 0 or i == total:
            log.info(f"  processed {i}/{total} frames")

    cap.release()


def _soften_alpha(alpha_u8: np.ndarray, dilate_px: int, blur_radius: float) -> np.ndarray:
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        alpha_u8 = cv2.dilate(alpha_u8, kernel)
    if blur_radius > 0:
        ksize = max(3, int(round(blur_radius * 2)) | 1)
        alpha_u8 = cv2.GaussianBlur(alpha_u8, (ksize, ksize), blur_radius)
    return alpha_u8


def main():
    _check_stereocrafter()

    cfg = load_config()
    rgb_dir = Path(cfg["paths"]["rgb_frames"])
    alpha_dir = Path(cfg["paths"]["alpha_frames"])
    left_dir = Path(cfg["paths"]["left_rgba"])
    right_dir = Path(cfg["paths"]["right_rgba"])
    weights_dir = get_weights_dir(cfg)

    ffmpeg_bin = resolve_binary("ffmpeg", cfg.get("ffmpeg_bin"))

    rgb_frames = list_frames(rgb_dir)
    if not rgb_frames:
        raise RuntimeError(f"No RGB frames in {rgb_dir}. Run 01_extract_frames.py first.")

    fps_file = PROJECT_ROOT / "frames" / "fps.txt"
    if not fps_file.is_file():
        raise RuntimeError(f"{fps_file} not found. Run 01_extract_frames.py first.")
    fps = fps_file.read_text().strip()

    ensure_dir(left_dir)
    ensure_dir(right_dir)
    for d in (left_dir, right_dir):
        for f in list_frames(d):
            f.unlink()

    with StageTimer(log, "StereoCrafter stereo synthesis"):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            input_video = tmp / "input.mp4"
            splatted_video = tmp / "splatted.mp4"
            inpaint_dir = tmp / "inpaint_out"

            _frames_to_video(rgb_dir, input_video, fps, ffmpeg_bin)
            _run_depth_splatting(input_video, splatted_video, weights_dir, cfg)
            _run_inpainting(splatted_video, inpaint_dir, weights_dir, cfg)

            sbs_video = _find_sbs_video(inpaint_dir)
            log.info(f"SBS video: {sbs_video}")

            _split_sbs_and_apply_alpha(
                sbs_video,
                alpha_dir,
                left_dir,
                right_dir,
                dilate_px=cfg["alpha_dilate_px"],
                blur_radius=cfg["alpha_blur_radius"],
            )

    log.info(f"Wrote {len(list_frames(left_dir))} left/right RGBA frame pairs")


if __name__ == "__main__":
    main()
