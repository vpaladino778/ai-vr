"""Step 2: Generate an alpha matte for the person using Robust Video Matting (RVM).

Loads the RVM MobileNetV3 model via torch.hub, runs it frame-by-frame while
carrying its recurrent state forward (RVM is a temporally-recurrent matting
network -- frames must be processed in order), and writes cleaned alpha mattes
(white = person, black = background) to frames/alpha/.

Cleanup applied per the handover doc: slight dilation, slight blur, and removal
of tiny speckles.
"""
import cv2
import numpy as np
import torch

from common import get_logger, list_frames, load_config, ensure_dir, StageTimer

log = get_logger("02_run_matting")


def remove_speckles(alpha_u8, min_area=64):
    """Zero out small connected components in the alpha mask (mask noise)."""
    _, binary = cv2.threshold(alpha_u8, 10, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = alpha_u8.copy()
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < min_area:
            cleaned[labels == label] = 0
    return cleaned


def clean_alpha(alpha_u8, dilate_px, blur_radius):
    cleaned = remove_speckles(alpha_u8)
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        cleaned = cv2.dilate(cleaned, kernel)
    if blur_radius > 0:
        ksize = max(3, int(round(blur_radius * 2)) | 1)  # odd kernel size
        cleaned = cv2.GaussianBlur(cleaned, (ksize, ksize), blur_radius)
    return cleaned


def main():
    cfg = load_config()
    rgb_dir = cfg["paths"]["rgb_frames"]
    alpha_dir = cfg["paths"]["alpha_frames"]
    downsample_ratio = cfg["matting_downsample_ratio"]
    dilate_px = cfg["alpha_dilate_px"]
    blur_radius = cfg["alpha_blur_radius"]

    frames = list_frames(rgb_dir)
    if not frames:
        raise RuntimeError(f"No RGB frames found in {rgb_dir}. Run 01_extract_frames.py first.")

    ensure_dir(alpha_dir)
    for f in list_frames(alpha_dir):
        f.unlink()

    with StageTimer(log, "alpha matting (RVM)"):
        log.info("Loading RVM MobileNetV3 model via torch.hub (downloads on first run)...")
        torch.set_num_threads(max(1, torch.get_num_threads()))
        model = torch.hub.load(
            "PeterL1n/RobustVideoMatting", "mobilenetv3", trust_repo=True
        ).eval()

        rec = [None] * 4  # recurrent states: must persist across frames in order
        for i, frame_path in enumerate(frames, start=1):
            bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            src = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0

            with torch.no_grad():
                fgr, pha, *rec = model(src, *rec, downsample_ratio=downsample_ratio)

            alpha_u8 = (pha[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8)
            alpha_u8 = clean_alpha(alpha_u8, dilate_px, blur_radius)

            out_path = frame_path.parent.parent / "alpha" / frame_path.name
            cv2.imwrite(str(out_path), alpha_u8)

            if i % 25 == 0 or i == len(frames):
                log.info(f"Matted {i}/{len(frames)} frames")

        log.info(f"Wrote {len(frames)} alpha mattes to {alpha_dir}")


if __name__ == "__main__":
    main()
