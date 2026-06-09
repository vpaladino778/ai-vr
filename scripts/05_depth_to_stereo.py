"""Step 5 (doc's "Convert 2D Person to Stereo Left/Right"): depth-warp into eyes.

For each frame:
  - load RGB, alpha, cleaned person-depth (already normalized 0-1 inside the mask)
  - convert depth -> per-pixel horizontal disparity, scaled by max_disparity_px
  - warp RGB+alpha for the left eye with -disparity/2 and the right eye with +disparity/2
  - dilate/blur the warped alpha edges to soften seams left by the warp
  - save left/right RGBA frames (background stays transparent -- no inpainting in the MVP,
    per the doc's "keep background black for MVP" guidance; the black canvas is added
    in the compose step)

Sign convention: closer (depth=1) pixels get pushed further apart (more disparity),
which is the standard "near things shift more" stereo cue. If the result looks
inverted/concave in the headset, flip the sign of DISPARITY_SIGN below or swap the
left/right outputs (see the doc's debugging checklist).
"""
from pathlib import Path

import cv2
import numpy as np

from common import get_logger, list_frames, load_config, ensure_dir, StageTimer

log = get_logger("05_depth_to_stereo")

DISPARITY_SIGN = -1.0  # close objects shift rightward in left eye, leftward in right eye


def warp_horizontal(rgba, shift_map):
    """Warp an RGBA image horizontally by a per-pixel float shift (in pixels)."""
    h, w = rgba.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = (xs + shift_map).astype(np.float32)
    map_y = ys
    warped = cv2.remap(
        rgba, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return warped


def soften_alpha_edges(alpha_u8, dilate_px, blur_radius):
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        alpha_u8 = cv2.dilate(alpha_u8, kernel)
    if blur_radius > 0:
        ksize = max(3, int(round(blur_radius * 2)) | 1)
        alpha_u8 = cv2.GaussianBlur(alpha_u8, (ksize, ksize), blur_radius)
    return alpha_u8


def main():
    cfg = load_config()
    rgb_dir = cfg["paths"]["rgb_frames"]
    alpha_dir = cfg["paths"]["alpha_frames"]
    person_depth_dir = cfg["paths"]["person_depth_frames"]
    left_dir = cfg["paths"]["left_rgba"]
    right_dir = cfg["paths"]["right_rgba"]

    max_disparity = float(cfg["max_disparity_px"])
    dilate_px = cfg["alpha_dilate_px"]
    blur_radius = cfg["alpha_blur_radius"]

    rgb_frames = list_frames(rgb_dir)
    if not rgb_frames:
        raise RuntimeError(f"No RGB frames found in {rgb_dir}. Run 01_extract_frames.py first.")

    ensure_dir(left_dir)
    ensure_dir(right_dir)
    for d in (left_dir, right_dir):
        for f in list_frames(d):
            f.unlink()

    with StageTimer(log, "depth-to-stereo warp"):
        for i, rgb_path in enumerate(rgb_frames, start=1):
            name = rgb_path.name
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            alpha_u8 = cv2.imread(str(rgb_path.parent.parent / "alpha" / name), cv2.IMREAD_GRAYSCALE)
            depth_u16 = cv2.imread(str(rgb_path.parent.parent / "person_depth" / name), cv2.IMREAD_UNCHANGED)

            if alpha_u8 is None or depth_u16 is None:
                raise RuntimeError(f"Missing alpha or person-depth frame for {name}. Run steps 02 and 04 first.")

            rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = alpha_u8

            depth_f = depth_u16.astype(np.float32) / 65535.0  # already 0-1, 0 outside person
            # Power curve spreads midrange depth values, making subtle within-face depth differences
            # (nose vs. eye socket vs. cheek) more pronounced in the stereo output.
            depth_curved = np.where(depth_f > 0, np.power(depth_f, 0.65), 0.0).astype(np.float32)
            disparity = DISPARITY_SIGN * (depth_curved - 0.5) * max_disparity

            left_rgba = warp_horizontal(rgba, -disparity / 2.0)
            right_rgba = warp_horizontal(rgba, +disparity / 2.0)

            left_rgba[:, :, 3] = soften_alpha_edges(left_rgba[:, :, 3], dilate_px, blur_radius)
            right_rgba[:, :, 3] = soften_alpha_edges(right_rgba[:, :, 3], dilate_px, blur_radius)

            cv2.imwrite(str(Path(left_dir) / name), left_rgba)
            cv2.imwrite(str(Path(right_dir) / name), right_rgba)

            if i % 25 == 0 or i == len(rgb_frames):
                log.info(f"Warped stereo for {i}/{len(rgb_frames)} frames")

        log.info(f"Wrote {len(rgb_frames)} left/right RGBA frame pairs")


if __name__ == "__main__":
    main()
