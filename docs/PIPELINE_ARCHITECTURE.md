# VR180 Pipeline Architecture

## Project goal

Take a short 2D AI-generated video of a single person and produce a headset-testable stereoscopic VR180 video. The output should feel plausibly present in a Quest-class headset — the face should appear to exist in 3D space in front of the viewer.

This is not a mathematically perfect VR capture. It is a synthesis pipeline: we infer depth from a 2D image, use that depth to generate two slightly-offset eye views, and composite them into the VR180 side-by-side format. The result is a credible stereo illusion, not ground-truth capture.

---

## Why VR180, not 360

VR180 only covers the front 180° hemisphere. This makes it the right fit for a stationary subject (a person standing/sitting in front of the viewer) because:

- The viewer's natural attention stays forward
- No background or back-hemisphere content is needed
- Each eye gets a full square canvas (1920×1920) rather than a squeezed slice of a 360° equirectangular
- File sizes and encoding complexity are lower

The final SBS frame is 3840×1920 — two 1920×1920 half-equirectangular eye panels side by side.

---

## Input assumptions

The pipeline is designed around a specific class of source video:

```
- Duration:     5–10 seconds
- Subject:      single person, centered, facing camera
- Camera:       completely static, no zoom, no cuts
- Background:   simple/neutral (plain wall, studio-style)
- Motion:       subject can move, but no fast or erratic motion
- Lighting:     consistent throughout, no hard strobing
- Resolution:   720p minimum; 1080p preferred for headset quality
- Origin:       any AI video generator (WAN, Kling, Runway, Luma, etc.)
```

The pipeline does not care what generator produced the source. It treats it as a plain MP4.

---

## Pipeline overview

```
source_video.mp4
│
├─ 01: Extract RGB frames (ffmpeg, lossless PNG sequence)
│
├─ 02: Alpha matte the person (Robust Video Matting)
│       → frames/alpha/  — white=person, black=background
│
├─ 03: Estimate depth per frame (Depth Anything V2 Small)
│       → frames/depth/  — 16-bit grayscale, raw monocular depth
│
├─ 04: Clean person depth
│       → frames/person_depth/  — depth masked to person, normalized,
│                                  spatially and temporally smoothed
│
├─ 05: Depth-warp into stereo left/right eye frames
│       → stereo/left_rgba/
│       → stereo/right_rgba/
│
├─ 06: Compose onto VR180 SBS canvas
│       → compose/sbs_frames/  — 3840×1920 per frame
│
└─ 07: Encode HEVC MP4
        → output/vr180_person_sbs.mp4
```

Every stage reads from and writes to named frame directories. Stages are individually re-runnable, which matters for tuning: stages 5–7 are fast (seconds to minutes) and can be iterated without re-running the slow matting and depth estimation (stages 2–3).

---

## Stage-by-stage detail

### Stage 1 — Frame extraction

Tool: ffmpeg.

Extracts every frame as a lossless PNG (`000001.png`, `000002.png`, ...). Also probes the source FPS via ffprobe and writes it to `frames/fps.txt` — this propagates to the final encode so output FPS always matches source.

Frame numbering (`%06d`) is consistent across all stage directories. Every downstream script uses this numbering to match alpha, depth, and RGB frames by filename.

---

### Stage 2 — Alpha matting (Robust Video Matting)

Model: **RVM MobileNetV3**, loaded via `torch.hub`.

RVM is a recurrent matting network — it maintains hidden state across frames and uses temporal context to produce a stable, temporally-coherent matte. This is critical: frame-independent segmentation (like SAM or basic GrabCut) produces flickery mattes that make the stereo warp shimmer.

Implementation details:
- Frames are processed **in order** and recurrent state (`rec = [None]*4`) is carried forward. Shuffling frames or processing them independently would break temporal coherence.
- `downsample_ratio = 0.4` — the model internally upsamples; lowering this ratio reduces CPU time significantly with minimal quality loss on clean single-subject videos.
- After matting: small connected components are removed (speckle filter via `cv2.connectedComponentsWithStats`), slight dilation and Gaussian blur are applied to soften the alpha edge and avoid hard-edge artifacts when compositing onto the black canvas.

Output: 8-bit grayscale alpha frames. White (255) = fully person, black (0) = background. Values in between are soft transitions at edges.

---

### Stage 3 — Depth estimation (Depth Anything V2)

Model: **Depth Anything V2 Small** (`depth-anything/Depth-Anything-V2-Small-hf`), loaded via HuggingFace `transformers` pipeline.

This produces per-frame **relative monocular depth** maps. "Relative" means the scale is arbitrary per-frame — a depth value of 1000 does not mean 1 meter. The only guarantee is that higher values correspond to objects closer to the camera.

This is the dominant CPU bottleneck: ~0.7 seconds per frame at 720p on a modern CPU. The small model variant (ViT-S) is used for speed; accuracy is acceptable for the person-only depth we need.

Output: 16-bit grayscale PNGs preserving the full dynamic range of the raw model output. Per-frame normalization at this stage is over the whole frame — person-specific normalization happens in stage 4.

---

### Stage 4 — Depth cleanup

This stage exists because raw monocular depth cannot be used directly for stereo synthesis:

1. **Scale jumps between frames.** The raw depth for frame N might span [120, 980] and frame N+1 might span [50, 1200]. Direct use produces visible depth "popping" in the stereo output.

2. **Background depth noise.** The model estimates depth for the background too, which we don't need and which can contaminate the person's depth range.

3. **Edge noise at boundaries.** Depth transitions at the person silhouette are often unreliable.

What stage 4 does, per frame:

1. **Mask to person.** Uses the alpha frame to restrict depth to pixels where `alpha > 10`. Everything outside the person gets depth = 0.

2. **Percentile normalization inside the mask.** Computes the 1st and 99th percentile of depth values within the person region, then linearly maps this range to [0, 1]. This removes extreme outliers from influencing the normalization.

3. **Spatial smoothing.** `cv2.bilateralFilter` — edge-preserving spatial smoothing. Reduces noise within the person silhouette while keeping the person/background boundary sharp.

4. **Temporal smoothing (EMA).** Each frame's smoothed depth is blended with the previous frame: `smoothed = alpha * current + (1 - alpha) * previous`. `alpha = 0.4` keeps the current frame dominant while damping inter-frame flickering. This is the temporal stability mechanism that substitutes for Video-Depth-Anything's more sophisticated temporal consistency approach.

Output: 16-bit grayscale, 0 outside the person, 0–65535 inside scaled to the normalized [0, 1] range.

---

### Stage 5 — Depth to stereo warp

This is the core stereo synthesis step. The goal is to produce two slightly-different views of the person — one for each eye — that, when fused by the brain in a headset, create the perception of depth.

**Depth to disparity:**

```
disparity = DISPARITY_SIGN * (depth_f - 0.5) * max_disparity_px
```

`depth_f` is the normalized depth (0–1, higher = closer). Subtracting 0.5 centers disparity around zero — pixels at the midpoint depth appear in the same position in both eyes; closer pixels shift outward, further pixels shift inward. `max_disparity_px` is the tuning parameter that controls stereo intensity.

**Sign convention:**

`DISPARITY_SIGN = -1.0`. This ensures that close objects (depth_f → 1.0) produce **negative parallax** — the object appears in front of the screen plane:
- Left eye: close pixels shift rightward (+disparity/2)
- Right eye: close pixels shift leftward (−disparity/2)

This is the correct convention for "pop-out" VR stereo (subject appearing in front of the viewer). An inverted sign would produce a concave, "sunken" appearance.

**Warp implementation:**

`cv2.remap` with a per-pixel horizontal shift map derived from the disparity. The shift is applied only horizontally — vertical disparity is zero (standard for horizontal-baseline stereo).

**Alpha edge handling:**

After warping, the alpha channel has small disocclusion gaps at the shifted edges (pixels that moved away, leaving transparent holes). These are softened with the same dilation + blur from stage 2 rather than inpainted — sufficient for the MVP because the background is plain black.

---

### Stage 6 — VR180 SBS composition

This stage does not generate stereo. It takes the already-warped left/right RGBA frames and places them into the VR180 format using a correct equirectangular projection.

**Canvas structure:**

```
[ left eye: 1920×1920 ] [ right eye: 1920×1920 ]
└─────────────────────── 3840×1920 ─────────────────────────┘
```

Each 1920×1920 canvas is a half-equirectangular projection of the VR180 hemisphere for one eye. The entire canvas spans 180° horizontally and 180° vertically — canvas center = looking straight ahead, top edge = looking straight up, sides = 90° left/right.

**Why naive pasting fails:**

Placing a flat person rectangle directly onto this canvas causes the VR player to bend it onto the sphere surface. The person appears to wrap around the hemisphere, distorting as they move through the frame. This was the first symptom observed in headset testing.

**The fix — inverse equirectangular projection:**

For each output canvas pixel, we compute the sphere direction it represents, project that direction onto the virtual flat plane where the person "stands", and sample the person image at that point. Using `cv2.remap` for efficiency (maps precomputed once per resolution):

```
# Canvas pixel → sphere angles
theta = (cx / W - 0.5) * π       # horizontal: −π/2 to +π/2
phi   = (0.5 - cy / H) * π       # vertical: +π/2 (top) to −π/2 (bottom)

# Sphere angles → flat plane coordinates
flat_x = tan(theta) − tan(center_azimuth)
flat_y = tan(phi)   − tan(center_elevation)

# Flat plane → source image pixel
src_col = (flat_x / tan(half_fov_h) * 0.5 + 0.5) * source_w
src_row = (0.5 − flat_y / tan(half_fov_v) * 0.5) * source_h
```

The horizontal FoV is derived from the vertical FoV via the source frame's aspect ratio:
`half_fov_h = arctan(tan(half_fov_v) × aspect)`. This preserves the correct aspect on the virtual plane.

When the VR player renders the equirectangular canvas onto the hemisphere, it re-applies the forward equirectangular transform — which exactly cancels the inverse we applied. The result is a person who appears as a flat plane in 3D space.

**Placement parameters:**

```yaml
virtual_fov_v_deg: 60          # vertical angular extent of the source frame
                                # lower = person appears further/smaller (less distortion)
                                # higher = person appears closer/larger
virtual_center_elevation_deg: -15   # degrees below horizon for source frame center
                                     # −15° = person slightly below eye level (natural)
virtual_center_azimuth_deg: 0.0     # horizontal offset from straight ahead
```

To tune size: adjust `virtual_fov_v_deg`. To tune vertical position: adjust `virtual_center_elevation_deg`. These only require re-running stages 6–7 (~55s).

---

### Stage 7 — HEVC encode

Tool: ffmpeg with libx265.

```
codec:      libx265 (H.265/HEVC)
pixel fmt:  yuv420p
tag:        hvc1   (required for compatibility with Apple/Quest players)
fps:        matches source (from frames/fps.txt)
mode:       CRF 18 (default, high quality) or fixed bitrate 45Mbps
```

The `hvc1` tag is important — without it, some Quest players treat the stream as unsupported even though they can decode HEVC.

---

## Stereo sign and headset debugging

The stereo sign was determined empirically from the first real-video run. Measurement method: compute the horizontal center-of-mass of bright pixels in left vs. right eye panels on several scanlines through the face, and check the sign of (right_cx − left_cx).

Correct result: **negative shift** (~−10 to −11px at face, for `max_disparity_px = 16` on a 720p source). This means the right eye sees the face slightly to the left compared to the left eye — which is the correct pop-out stereo cue.

If testing in the headset produces a concave/inverted appearance despite a negative measured shift, the cause is likely that the VR player is swapping eye assignment. The fix is either to swap the left/right frame directories or to set `DISPARITY_SIGN = 1.0`.

---

## What this pipeline does NOT do (MVP scope)

- **No background.** Both eye canvases are plain black. The person floats in void.
- **No VR180 metadata.** The MP4 has no spatial media tags. Apps require manual projection selection.
- **No inpainting.** Disocclusion holes from the warp are left transparent (invisible against black background).
- **No audio passthrough.** Audio from the source video is not included in the output.
- **No upscaling.** Output quality is bounded by source resolution. A 720p source in a 1920×1920 canvas will look soft in the headset.

---

## Future upgrade path

In rough priority order:

1. **Add audio** — pass `-map 0:a` through the encode stage.
2. **Add VR180 metadata** — embed `st3d`/`sv3d` boxes (Google Spatial Media tools) for auto-detection in players.
3. **Higher-resolution source** — run the AI generator at 1080p or 4K; the pipeline handles any resolution.
4. **Better temporal depth** — swap stage 3 for Video-Depth-Anything (full temporal-consistency model) if EMA smoothing produces visible depth flicker on complex motion.
5. **Background environment** — add a stereo 3D room/scene behind the person; composite the person over it in stage 6.
6. **Inpainting for disocclusions** — fill warp holes with content-aware fill rather than transparency.
7. **Disparity refinement** — apply a diffusion-based stereo refinement pass (StereoCrafter-style) for photorealistic left/right divergence rather than pure horizontal shift.
8. **Upscaling** — run a video super-resolution model on the SBS frames before encode (target: 5760×2880 or higher).

---

## Configuration reference

All tunable parameters live in `configs/pipeline_config.yaml`.

```yaml
# --- Stereo geometry ---
max_disparity_px: 16           # stereo intensity; scale with source resolution
                                # 720p: 8–20, 1080p: 12–30
# --- Canvas layout ---
canvas_size: 1920               # each eye is this × this pixels

# --- Angular placement (controls how the person appears in the headset) ---
virtual_fov_v_deg: 60           # vertical angular extent of the source frame
                                 # lower = smaller/further, higher = larger/closer
virtual_center_elevation_deg: -15  # degrees below horizon for source frame center
virtual_center_azimuth_deg: 0.0    # horizontal offset from center (0 = centered)

# --- Matting ---
alpha_dilate_px: 2             # edge dilation after RVM
alpha_blur_radius: 1.5         # edge softness

# --- Depth cleanup ---
depth_clamp_percentile_low: 1.0
depth_clamp_percentile_high: 99.0
depth_spatial_smooth_sigma_color: 0.05
depth_spatial_smooth_sigma_space: 9.0
depth_temporal_smooth_alpha: 0.4   # EMA weight; 0.4 = moderate smoothing

# --- Matting runtime ---
matting_downsample_ratio: 0.4      # lower = faster CPU inference

# --- Encoding ---
encode_mode: "crf"             # "crf" for quality, "bitrate" for target size
encode_crf: 18                 # lower = better quality, larger file
```

---

## Playback (Meta Quest)

Transfer `output/vr180_person_sbs.mp4` to the headset via USB-C. Open in **DeoVR** (free, Meta Quest store). In DeoVR's settings for this file, manually set:

- Projection: **180° SBS** (or VR180)
- Stereo: **Left/Right**

The file does not auto-detect in the MVP because VR180 spatial metadata has not been embedded. This is a known limitation to address in a future iteration.
