"""Step 3: Estimate per-frame depth using Depth Anything V2 (Small).

This produces raw monocular depth maps (frames/depth/). They are NOT temporally
stable on their own -- per-frame normalization, clamping and temporal smoothing
happen in 04_clean_depth.py, which is also where the depth gets restricted to the
person region using the alpha matte. Output here is a 16-bit grayscale PNG so we
keep full precision for the cleanup stage.
"""
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from common import get_logger, list_frames, load_config, ensure_dir, StageTimer

log = get_logger("03_run_depth")

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def main():
    cfg = load_config()
    rgb_dir = cfg["paths"]["rgb_frames"]
    depth_dir = cfg["paths"]["depth_frames"]

    frames = list_frames(rgb_dir)
    if not frames:
        raise RuntimeError(f"No RGB frames found in {rgb_dir}. Run 01_extract_frames.py first.")

    ensure_dir(depth_dir)
    for f in list_frames(depth_dir):
        f.unlink()

    with StageTimer(log, "depth estimation (Depth Anything V2 Small)"):
        log.info(f"Loading {MODEL_ID} via transformers (downloads on first run)...")
        depth_pipe = pipeline(task="depth-estimation", model=MODEL_ID, device=-1)  # CPU

        for i, frame_path in enumerate(frames, start=1):
            image = Image.open(frame_path).convert("RGB")
            result = depth_pipe(image)
            depth = result["predicted_depth"]  # torch tensor, relative depth (higher = closer)

            if isinstance(depth, torch.Tensor):
                depth = depth.squeeze().cpu().numpy()
            depth = depth.astype(np.float32)

            # Resize back to source resolution if the model output a different size.
            if depth.shape[::-1] != image.size:
                depth = cv2.resize(depth, image.size, interpolation=cv2.INTER_CUBIC)

            # Save as 16-bit grayscale, normalized over the full frame's own range.
            # (Person-region-specific normalization happens in the cleanup stage,
            # which has access to the alpha matte; this just preserves precision.)
            d_min, d_max = float(depth.min()), float(depth.max())
            span = max(d_max - d_min, 1e-6)
            depth_u16 = ((depth - d_min) / span * 65535).astype(np.uint16)

            out_path = frame_path.parent.parent / "depth" / frame_path.name
            cv2.imwrite(str(out_path), depth_u16)

            if i % 25 == 0 or i == len(frames):
                log.info(f"Estimated depth for {i}/{len(frames)} frames")

        log.info(f"Wrote {len(frames)} depth maps to {depth_dir}")


if __name__ == "__main__":
    main()
