# AI Video → VR180 Pipeline

## Project Purpose

Convert a short 2D AI-generated video (single centered person, static camera) into a VR180 stereo video playable on Meta Quest via DeoVR. This is a synthesis pipeline — it produces a believable stereo illusion, not physically correct geometry.

**Architecture doc:** `docs/PIPELINE_ARCHITECTURE.md` — update it whenever pipeline stages, config parameters, or the projection approach changes.

---

## Tech Stack

- **Language:** Python 3.11+
- **Deep learning:** PyTorch (CPU), HuggingFace `transformers`
- **Models:** RVM MobileNetV3 (matting, via `torch.hub`), Depth Anything V2 Small (depth, via HuggingFace)
- **Image/video:** OpenCV, Pillow, ffmpeg + ffprobe (libx265)
- **Config:** YAML (`configs/pipeline_config.yaml`)

---

## Setup

```bash
# Install ffmpeg with libx265 (Windows)
winget install Gyan.FFmpeg

# Python environment
py -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

Models download automatically on first run from `torch.hub` and HuggingFace.

---

## Running the Pipeline

```bash
# Full pipeline (source video must be at source/source_video.mp4)
.venv/Scripts/python scripts/run_pipeline.py

# Resume from a specific stage (useful after tuning config)
.venv/Scripts/python scripts/run_pipeline.py 05_depth_to_stereo
```

**Output:** `output/vr180_person_sbs.mp4` — 3840×1920 HEVC, 180° SBS stereo.  
**Playback:** DeoVR on Quest → projection: 180° SBS, stereo: Left/Right.

---

## Pipeline Stages

| Script | Input → Output | Notes |
|--------|---------------|-------|
| `01_extract_frames.py` | MP4 → `frames/rgb/` | Lossless PNG, saves FPS to `frames/fps.txt` |
| `02_run_matting.py` | rgb → `frames/alpha/` | RVM temporal matting; white=person |
| `03_run_depth.py` | rgb → `frames/depth/` | Depth Anything V2; 16-bit, relative scale |
| `04_clean_depth.py` | depth + alpha → `frames/person_depth/` | Mask to person, percentile clamp, bilateral + EMA smoothing |
| `05_depth_to_stereo.py` | rgb + person_depth → `stereo/left_rgba/`, `stereo/right_rgba/` | `disparity = -1.0 * (depth - 0.5) * max_disparity_px` |
| `06_compose_vr180_sbs.py` | stereo RGBA → `compose/sbs_frames/` | Inverse equirectangular projection onto 3840×1920 canvas |
| `07_encode.py` | sbs_frames + fps.txt → `output/vr180_person_sbs.mp4` | libx265, CRF 18, hvc1 tag |

**Orchestrator:** `scripts/run_pipeline.py` runs all stages in sequence; supports resuming from any stage name.  
**Shared utilities:** `scripts/common.py` — config loading, logging, frame I/O helpers, `StageTimer`.

---

## Key Config Parameters (`configs/pipeline_config.yaml`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `max_disparity_px` | 16 | Stereo depth intensity. 720p: 8–20, 1080p: 12–30 |
| `virtual_fov_v_deg` | 60 | Vertical arc the person spans in headset. Lower = smaller/further |
| `virtual_center_elevation_deg` | -15 | Vertical placement; -15° = natural standing person |
| `virtual_center_azimuth_deg` | 0.0 | Horizontal offset; 0 = centered |
| `depth_temporal_smooth_alpha` | 0.4 | EMA smoothing strength (0=no smoothing, 1=freeze) |
| `matting_downsample_ratio` | 0.4 | Lower = faster CPU matting |
| `encode_crf` | 18 | Lower = better quality, larger file |

---

## Debugging Stereo

- **Wrong eye (person appears recessed, not popping out):** Swap left/right in DeoVR, or flip the sign of `max_disparity_px` in stage 5.
- **Depth flickering:** Increase `depth_temporal_smooth_alpha` (toward 0.6–0.8).
- **Person too small/large:** Adjust `virtual_fov_v_deg`.
- **Person too high/low:** Adjust `virtual_center_elevation_deg`.

Full stereo debugging checklist in `docs/PIPELINE_ARCHITECTURE.md`.

---

## MVP Scope (Known Limitations)

- No background (black void behind person)
- No VR180 spatial metadata injected into MP4
- No disocclusion inpainting (holes at warped edges are transparent)
- No audio pass-through
- CPU-only inference (~0.7s/frame for depth on 720p)
