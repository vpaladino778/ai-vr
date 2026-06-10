"""Step 6: Compose warped stereo RGBA frames into VR180 SBS with equirectangular projection.

THE CORE PROBLEM THIS SOLVES:
The VR180 format uses half-equirectangular projection — the 1920×1920 canvas maps
onto the inside of a hemisphere. If you paste a flat rectangular image directly onto
this canvas, the VR player will bend it to conform to the sphere surface. The person
ends up wrapped around the sphere, distorting as they move through the frame.

THE FIX:
Apply the INVERSE equirectangular projection before compositing. For each pixel on
the output canvas we compute the sphere direction it represents, project that direction
onto the virtual flat plane where the person "stands", and sample the person image at
that point. The VR player then re-applies the equirectangular transform, and the two
operations cancel out — the person appears as a flat plane in 3D space.

KEY CONFIG PARAMETERS:
  virtual_fov_v_deg            -- how many degrees of vertical arc the source frame spans
  virtual_center_elevation_deg -- where the center of the source frame sits vertically
                                   (negative = below horizon, e.g. -15 means the frame
                                    center is 15° below eye level, which is natural for
                                    a person standing/sitting in front of the viewer)
  virtual_center_azimuth_deg   -- horizontal offset from straight ahead (default 0 = centered)
"""
import shutil
from pathlib import Path

import cv2
import numpy as np

from common import get_logger, list_frames, load_config, ensure_dir, StageTimer

log = get_logger("06_compose_vr180_sbs")


def build_equirect_remap(canvas_size, source_h, source_w,
                          virtual_fov_v_deg, virtual_center_elevation_deg,
                          virtual_center_azimuth_deg=0.0):
    """Precompute remap maps: for each canvas pixel, which source pixel to sample.

    Implements the inverse half-equirectangular → rectilinear (flat plane) projection.
    The maps are floats suitable for cv2.remap with INTER_LINEAR.

    Canvas coordinate convention (standard equirectangular):
      x = 0      → looking 90° left
      x = W/2    → looking straight ahead
      x = W      → looking 90° right
      y = 0      → looking straight up
      y = H/2    → looking at the horizon
      y = H      → looking straight down
    """
    half_fov_v = np.radians(virtual_fov_v_deg / 2.0)
    aspect = source_w / float(source_h)
    # Horizontal FoV follows naturally from aspect ratio + vertical FoV on a flat plane.
    half_fov_h = np.arctan(np.tan(half_fov_v) * aspect)

    center_elev = np.radians(virtual_center_elevation_deg)
    center_az = np.radians(virtual_center_azimuth_deg)

    # Build full grid of canvas pixel coordinates.
    cy, cx = np.mgrid[0:canvas_size, 0:canvas_size].astype(np.float32)

    # Convert canvas pixels to sphere angles.
    # Each axis spans π radians (the 180° half-sphere).
    theta = (cx / canvas_size - 0.5) * np.pi   # horizontal: -π/2 .. +π/2
    phi   = (0.5 - cy / canvas_size) * np.pi    # vertical: +π/2 (top) .. -π/2 (bottom)

    # Project sphere direction onto the virtual flat plane (tangent-space).
    flat_x = np.tan(theta) - np.tan(center_az)
    flat_y = np.tan(phi)   - np.tan(center_elev)

    # Map flat-plane coords to source image pixels.
    # flat_x ∈ [-tan(half_fov_h), +tan(half_fov_h)] → source column [0, source_w]
    # flat_y ∈ [-tan(half_fov_v), +tan(half_fov_v)] → source row    [source_h, 0]
    src_col = ( flat_x / np.tan(half_fov_h) * 0.5 + 0.5) * source_w
    src_row = (0.5 - flat_y / np.tan(half_fov_v) * 0.5) * source_h

    return src_col.astype(np.float32), src_row.astype(np.float32)


def alpha_composite(canvas_bgr, rgba_overlay):
    """Alpha-composite a full-canvas BGRA overlay onto a BGR canvas in-place."""
    alpha = rgba_overlay[:, :, 3:4].astype(np.float32) / 255.0
    fg = rgba_overlay[:, :, :3].astype(np.float32)
    bg = canvas_bgr.astype(np.float32)
    return (fg * alpha + bg * (1.0 - alpha)).astype(np.uint8)


def main():
    cfg = load_config()
    left_dir = cfg["paths"]["left_rgba"]
    right_dir = cfg["paths"]["right_rgba"]
    sbs_dir = cfg["paths"]["sbs_frames"]

    canvas_size = cfg["canvas_size"]
    final_width = cfg["final_width"]
    final_height = cfg["final_height"]
    virtual_fov_v_deg = cfg["virtual_fov_v_deg"]
    virtual_center_elevation_deg = cfg["virtual_center_elevation_deg"]
    virtual_center_azimuth_deg = cfg.get("virtual_center_azimuth_deg", 0.0)
    bg = cfg["background_color"]
    bg_color_bgr = (bg["b"], bg["g"], bg["r"])

    assert final_width == 2 * canvas_size, "final_width must equal 2 * canvas_size"
    assert final_height == canvas_size, "final_height must equal canvas_size"

    left_frames = list_frames(left_dir)
    right_frames = list_frames(right_dir)
    if not left_frames or not right_frames:
        raise RuntimeError("No stereo frames found. Run 05_depth_to_stereo.py first.")
    if len(left_frames) != len(right_frames):
        raise RuntimeError("Left/right frame counts differ — re-run 05_depth_to_stereo.py.")

    ensure_dir(sbs_dir)
    for f in list_frames(sbs_dir):
        f.unlink()

    with StageTimer(log, "compose VR180 SBS (equirectangular projection)"):
        # Remap maps depend only on canvas size and source dimensions — precompute once.
        probe = cv2.imread(str(left_frames[0]), cv2.IMREAD_UNCHANGED)
        source_h, source_w = probe.shape[:2]

        log.info(
            f"Source: {source_w}×{source_h}  |  "
            f"Virtual FoV: {virtual_fov_v_deg}°V  |  "
            f"Center elevation: {virtual_center_elevation_deg}°  |  "
            f"Canvas: {canvas_size}×{canvas_size}"
        )

        map_x, map_y = build_equirect_remap(
            canvas_size, source_h, source_w,
            virtual_fov_v_deg, virtual_center_elevation_deg,
            virtual_center_azimuth_deg,
        )

        for i, (left_path, right_path) in enumerate(zip(left_frames, right_frames), start=1):
            left_rgba = cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED)
            right_rgba = cv2.imread(str(right_path), cv2.IMREAD_UNCHANGED)

            # Project each flat eye frame onto its equirectangular canvas.
            remap_flags = cv2.INTER_LINEAR
            remap_border = (cv2.BORDER_CONSTANT, cv2.BORDER_CONSTANT)

            left_proj = cv2.remap(
                left_rgba, map_x, map_y, remap_flags,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
            )
            right_proj = cv2.remap(
                right_rgba, map_x, map_y, remap_flags,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
            )

            left_canvas = np.full((canvas_size, canvas_size, 3), bg_color_bgr, dtype=np.uint8)
            right_canvas = np.full((canvas_size, canvas_size, 3), bg_color_bgr, dtype=np.uint8)

            left_canvas = alpha_composite(left_canvas, left_proj)
            right_canvas = alpha_composite(right_canvas, right_proj)

            sbs = np.hstack([left_canvas, right_canvas])
            assert sbs.shape[1] == final_width and sbs.shape[0] == final_height

            out_path = Path(sbs_dir) / f"sbs_{left_path.stem}.png"
            cv2.imwrite(str(out_path), sbs)

            if i % 25 == 0 or i == len(left_frames):
                log.info(f"Composed {i}/{len(left_frames)} SBS frames")

        log.info(f"Wrote {len(left_frames)} SBS frames ({final_width}×{final_height}) to {sbs_dir}")

        log.info("Removing intermediate stereo RGBA frames to free disk ...")
        shutil.rmtree(left_dir, ignore_errors=True)
        shutil.rmtree(right_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
