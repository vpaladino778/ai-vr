"""Step 4 (doc's "person depth"): mask, normalize, clamp and smooth depth.

For each frame:
  - mask the raw depth to the person region (alpha matte)
  - normalize depth to 0-1 using percentile clamping computed ONLY inside the mask
  - smooth spatially with an edge-preserving bilateral filter
  - smooth temporally with an exponential moving average across frames
  - write the cleaned person-depth (16-bit grayscale, 0 outside the person)

This directly implements the "do not trust raw monocular depth directly" guidance:
raw Depth-Anything output is masked and renormalized per-person rather than used as-is.
"""
import cv2
import numpy as np

from common import get_logger, list_frames, load_config, ensure_dir, StageTimer

log = get_logger("04_clean_depth")


def main():
    cfg = load_config()
    rgb_dir = cfg["paths"]["rgb_frames"]
    alpha_dir = cfg["paths"]["alpha_frames"]
    depth_dir = cfg["paths"]["depth_frames"]
    person_depth_dir = cfg["paths"]["person_depth_frames"]

    lo_pct = cfg["depth_clamp_percentile_low"]
    hi_pct = cfg["depth_clamp_percentile_high"]
    sigma_color = cfg["depth_spatial_smooth_sigma_color"]
    sigma_space = cfg["depth_spatial_smooth_sigma_space"]
    ema_alpha = cfg["depth_temporal_smooth_alpha"]

    rgb_frames = list_frames(rgb_dir)
    if not rgb_frames:
        raise RuntimeError(f"No RGB frames found in {rgb_dir}. Run 01_extract_frames.py first.")

    ensure_dir(person_depth_dir)
    for f in list_frames(person_depth_dir):
        f.unlink()

    with StageTimer(log, "depth cleanup (mask, normalize, smooth)"):
        prev_smoothed = None  # for temporal EMA
        for i, rgb_path in enumerate(rgb_frames, start=1):
            name = rgb_path.name
            alpha_u8 = cv2.imread(str(rgb_path.parent.parent / "alpha" / name), cv2.IMREAD_GRAYSCALE)
            depth_u16 = cv2.imread(str(rgb_path.parent.parent / "depth" / name), cv2.IMREAD_UNCHANGED)

            if alpha_u8 is None or depth_u16 is None:
                raise RuntimeError(f"Missing alpha or depth frame for {name}. Run steps 02 and 03 first.")

            mask = alpha_u8 > 10  # boolean person mask
            depth_f = depth_u16.astype(np.float32) / 65535.0

            if not mask.any():
                cleaned_u16 = np.zeros_like(depth_u16)
            else:
                person_values = depth_f[mask]
                lo = np.percentile(person_values, lo_pct)
                hi = np.percentile(person_values, hi_pct)
                span = max(hi - lo, 1e-6)

                normalized = np.clip((depth_f - lo) / span, 0.0, 1.0)

                # Edge-preserving spatial smoothing (bilateral keeps person silhouette crisp).
                smoothed = cv2.bilateralFilter(
                    normalized.astype(np.float32), d=-1,
                    sigmaColor=sigma_color, sigmaSpace=sigma_space,
                )

                # Temporal smoothing via EMA against the previous cleaned frame.
                if prev_smoothed is not None:
                    smoothed = ema_alpha * smoothed + (1.0 - ema_alpha) * prev_smoothed
                prev_smoothed = smoothed

                cleaned = np.where(mask, smoothed, 0.0)
                cleaned_u16 = (np.clip(cleaned, 0.0, 1.0) * 65535).astype(np.uint16)

            out_path = rgb_path.parent.parent / "person_depth" / name
            cv2.imwrite(str(out_path), cleaned_u16)

            if i % 25 == 0 or i == len(rgb_frames):
                log.info(f"Cleaned depth for {i}/{len(rgb_frames)} frames")

        log.info(f"Wrote {len(rgb_frames)} cleaned person-depth maps to {person_depth_dir}")


if __name__ == "__main__":
    main()
