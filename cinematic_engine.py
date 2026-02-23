"""
Cinematic Engine — DiaryOfJazbaat
Color grading + slow zoom + PIL text burn → 9:16 Short

TEXT STRATEGY: PIL draws text directly onto each frame (numpy array).
  ✓ Zero ImageMagick dependency
  ✓ Pure PIL compositing — no MoviePy layer bugs
  ✓ Works on Windows, macOS, Linux identically
  ✓ Single VideoClip.make_frame — no layer compositing failures
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, VideoClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from proglog import ProgressBarLogger

TARGET_H = 1920
TARGET_W = 1080
DURATION = 8.0
FPS      = 24

GRADE_PARAMS = {
    "Pain":        {"brightness": -0.12, "contrast": 1.25, "saturation": 0.70},
    "Realization": {"brightness":  0.08, "contrast": 1.05, "saturation": 1.10},
    "Growth":      {"brightness":  0.15, "contrast": 1.10, "saturation": 1.20},
    "Ego":         {"brightness": -0.05, "contrast": 1.40, "saturation": 0.85},
    "Calm":        {"brightness":  0.05, "contrast": 0.90, "saturation": 0.85},
}

# ─── FONT LOADER ──────────────────────────────────────────────────────────────

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """
    Try every known system font path across Windows / macOS / Linux.
    Never raises — falls back to PIL's built-in bitmap font if all else fails.
    """
    candidates = [
        # ── Windows ────────────────────────────────────────────────────────────
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\trebucbd.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
        # ── macOS ──────────────────────────────────────────────────────────────
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/SFNSDisplay-Bold.otf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        # ── Linux ──────────────────────────────────────────────────────────────
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/liberation-sans/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                f = ImageFont.truetype(path, size)
                print(f"  [font] Loaded: {path}")
                return f
            except Exception:
                continue

    # Last resort: PIL bitmap font (always present, small but functional)
    print("  [font] WARNING: No TrueType font found — using PIL default bitmap font. "
          "Text will be small. Install fonts or set FONT_PATH env var.", file=sys.stderr)
    return ImageFont.load_default()


# Cache font objects so we don't reload per-frame
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}

def _get_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _FONT_CACHE:
        # Allow override via environment variable
        custom = os.getenv("JAZBAAT_FONT_PATH", "")
        if custom and os.path.exists(custom):
            try:
                _FONT_CACHE[size] = ImageFont.truetype(custom, size)
                print(f"  [font] Custom font: {custom}")
            except Exception:
                _FONT_CACHE[size] = _load_font(size)
        else:
            _FONT_CACHE[size] = _load_font(size)
    return _FONT_CACHE[size]


# ─── TEXT RENDERER — Premium Cinematic Caption Style ─────────────────────────
#
# Target aesthetic: High-end Instagram Reel / film typography
#   • Warm white text (#f4efe6) — NOT pure white
#   • Serif font (Cormorant/Playfair/Georgia) — elegant, not UI
#   • Soft drop shadow — depth, not glow
#   • Very subtle 1px dark stroke for crisp edges
#   • Cinematic placement zones (lower-third variants, not always dead-center)
#   • Line1 slightly larger, Line2 slightly smaller
#   • Float-up animation: subtle upward drift as text fades in
#   • Soft vignette baked into every frame
#   • Gradient underlay beneath caption zone
#   • Max caption width 72% of frame — negative space preserved

# ── Warm white palette ────────────────────────────────────────────────────────
CAPTION_COLORS = {
    "Pain":        (242, 233, 220),   # warm ivory
    "Realization": (245, 241, 232),   # golden cream
    "Growth":      (240, 248, 238),   # cool pearl
    "Ego":         (244, 239, 230),   # warm off-white
    "Calm":        (238, 242, 248),   # cool silk
}

# ── Cinematic placement zones (x_frac, y_frac, align) ────────────────────────
# x_frac = centre-x as fraction of width
# y_frac = baseline of block as fraction of height
# align  = "left" | "center" | "right"
CAPTION_ZONES = [
    (0.50, 0.74, "center"),   # centered bottom  — most common
    (0.50, 0.80, "center"),   # slightly lower
    (0.30, 0.76, "left"),     # lower-third left
    (0.70, 0.76, "right"),    # lower-third right
    (0.50, 0.70, "center"),   # slightly above bottom
]

# Fixed zone per render (chosen once from emotion hash so it's deterministic)
def _pick_zone(emotion: str, line1: str) -> tuple:
    idx = (hash(emotion + line1[:6])) % len(CAPTION_ZONES)
    return CAPTION_ZONES[idx]

from proglog import ProgressBarLogger

class RenderLogger(ProgressBarLogger):
    def __init__(self, job_id, jobs_dict):
        super().__init__()
        self.job_id = job_id
        self.jobs_dict = jobs_dict

    def callback(self, **changes):
        if "progress" in changes:
            pct = int(changes["progress"] * 100)

            # Clamp safely
            pct = max(0, min(100, pct))

            if self.job_id in self.jobs_dict:
                self.jobs_dict[self.job_id]["progress"] = pct
                self.jobs_dict[self.job_id]["status"] = "rendering"

def _measure_text(font, text: str):
    try:
        bb = font.getbbox(text)
        return bb[2] - bb[0], bb[3] - bb[1]
    except AttributeError:
        return font.getsize(text)


def _load_serif_font(size: int) -> ImageFont.FreeTypeFont:
    """Try to load a cinematic serif font; fall back to sans-serif; then bitmap."""
    serif_candidates = [
        # Windows — Palatino Linotype (elegant serif)
        r"C:\Windows\Fonts\pala.ttf",
        r"C:\Windows\Fonts\palabi.ttf",
        r"C:\Windows\Fonts\georgia.ttf",
        r"C:\Windows\Fonts\georgiab.ttf",
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\timesbi.ttf",
        # macOS
        "/Library/Fonts/Palatino.ttc",
        "/System/Library/Fonts/Supplemental/Palatino.ttc",
        "/Library/Fonts/Georgia.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        # Linux
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/TTF/DejaVuSerif.ttf",
    ]
    for path in serif_candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # Fall back to sans-serif via existing loader
    return _load_font(size)


_SERIF_CACHE: dict = {}

def _get_serif_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _SERIF_CACHE:
        custom = os.getenv("JAZBAAT_SERIF_FONT_PATH", "")
        if custom and os.path.exists(custom):
            try:
                _SERIF_CACHE[size] = ImageFont.truetype(custom, size)
            except Exception:
                _SERIF_CACHE[size] = _load_serif_font(size)
        else:
            _SERIF_CACHE[size] = _load_serif_font(size)
    return _SERIF_CACHE[size]


def _apply_vignette(frame: np.ndarray) -> np.ndarray:
    """Burn a soft elliptical vignette into the frame. Pure numpy — fast."""
    h, w = frame.shape[:2]
    # Build vignette mask once and cache
    ys = np.linspace(-1.0, 1.0, h)
    xs = np.linspace(-1.0, 1.0, w)
    xv, yv = np.meshgrid(xs, ys)
    # Ellipse — tighter vertically for cinematic feel
    dist = np.clip(xv**2 * 0.55 + yv**2 * 0.65, 0, 1)
    # Smooth quintic falloff — no hard edge
    mask = 1.0 - dist**2 * (3 - 2 * dist)   # smoothstep
    mask = np.clip(mask, 0.42, 1.0)           # floor at 0.42 so corners aren't pitch black
    vignette = mask[:, :, np.newaxis]
    f = frame.astype(np.float32) * vignette
    return np.clip(f, 0, 255).astype(np.uint8)


def _draw_caption_gradient(img_rgba: Image.Image, x0: int, y0: int, x1: int, y1: int, alpha: int):
    """Draw a very subtle dark-to-transparent gradient beneath the caption zone."""
    grad_h = int((y1 - y0) * 2.2)
    grad_y0 = max(0, y0 - grad_h // 2)
    grad_y1 = min(img_rgba.height, y1 + 20)
    if grad_y1 <= grad_y0:
        return
    grad = Image.new("RGBA", (img_rgba.width, grad_y1 - grad_y0), (0, 0, 0, 0))
    draw = ImageDraw.Draw(grad)
    steps = grad_y1 - grad_y0
    max_a = int(60 * alpha / 255)   # very subtle — 60 max opacity
    for i in range(steps):
        frac = i / max(steps - 1, 1)
        # Peaks in the middle of gradient, zero at top and bottom
        a = int(max_a * 4 * frac * (1 - frac))
        draw.line([(0, i), (img_rgba.width, i)], fill=(0, 0, 0, a))
    img_rgba.paste(grad, (0, grad_y0), grad)

from PIL import ImageFilter

def _apply_grain(frame: np.ndarray, strength: float = 0.035) -> np.ndarray:
    noise = np.random.normal(0, 255 * strength, frame.shape).astype(np.float32)
    grain = frame.astype(np.float32) + noise
    return np.clip(grain, 0, 255).astype(np.uint8)

def _apply_bloom(frame: np.ndarray, threshold: int = 215, blur_radius: int = 5) -> np.ndarray:
    img = Image.fromarray(frame)
    bright_mask = img.convert("L").point(lambda p: 255 if p > threshold else 0)
    glow = img.filter(ImageFilter.GaussianBlur(blur_radius))
    blended = Image.composite(glow, img, bright_mask)
    return np.asarray(blended, dtype=np.uint8)

def _render_text_onto_frame(
    frame:       np.ndarray,
    line1:       str,
    line2:       str,
    t:           float,
    emotion:     str  = "Pain",
    fade1_start: float = 0.30,
    fade2_start: float = 3.00,   # Line2 appears after 3 seconds
    fade_dur:    float = 0.45,
    float_px:    int  = 6,       # upward float distance in pixels
) -> np.ndarray:
    """
    Burn premium cinematic captions onto frame.
    - Warm white serif text, not pure white
    - Drop shadow (soft, medium blur simulated via multi-offset)
    - 1px dark stroke
    - Cinematic zone placement
    - Float-up animation
    - Vignette
    - Gradient underlay under caption
    """
    h, w = frame.shape[:2]

    # Vignette first (modifies frame pixels directly)
    #frame = _apply_vignette(frame)

    # ── Font sizes ──────────────────────────────────────────────────────────
    size1 = max(36, h // 46)          # Line1 slightly larger
    size2 = max(30, int(h / 52))      # Line2 slightly smaller
    font1 = _get_serif_font(size1)
    font2 = _get_serif_font(size2)

    # ── Max width = 72% of frame ────────────────────────────────────────────
    max_w_px = int(w * 0.72)

    # ── Text color with emotion tint ────────────────────────────────────────
    text_rgb = CAPTION_COLORS.get(emotion, (244, 239, 230))

    # ── Cinematic zone ──────────────────────────────────────────────────────
    zone_x_frac, zone_y_frac, align = _pick_zone(emotion, line1)
    block_cx  = int(w * zone_x_frac)
    block_y   = int(h * zone_y_frac)
    safe_margin = int(w * 0.06)        # 6% safe margin from edges

    # ── Fade + float calculation ────────────────────────────────────────────
    def ease_out(start: float, dur: float = fade_dur) -> float:
        if t < start: return 0.0
        p = min(1.0, (t - start) / dur)
        return 1.0 - (1.0 - p) ** 3   # ease-out cubic [0..1]

    prog1 = ease_out(fade1_start)
    prog2 = ease_out(fade2_start)
    alpha1 = int(242 * prog1)          # 95% max opacity (warm white)
    alpha2 = int(242 * prog2)

    # Float offset: starts float_px below final position, rises to 0
    float1 = int(float_px * (1.0 - prog1))
    float2 = int(float_px * (1.0 - prog2))

    # ── Build overlay ───────────────────────────────────────────────────────
    img     = Image.fromarray(frame, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    def wrap_to_width(text: str, font) -> list:
        """Word-wrap text to fit within max_w_px. Returns list of lines."""
        words = text.split()
        lines_out = []
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            tw, _ = _measure_text(font, test)
            if tw <= max_w_px:
                current = test
            else:
                if current:
                    lines_out.append(current)
                current = word
        if current:
            lines_out.append(current)
        return lines_out or [text]

    def draw_text_block(text: str, font, alpha: int, float_off: int,
                        y_top: int, is_line1: bool) -> int:
        """Render one wrapped text block. Returns bottom y of block."""
        if alpha <= 0:
            return y_top
        lines_out = wrap_to_width(text, font)
        line_h = int(_measure_text(font, "Ag")[1] * 1.45)   # 1.45× line height
        total_h = len(lines_out) * line_h

        y_cur = y_top + float_off

        for ln in lines_out:
            tw, th = _measure_text(font, ln)

            # X position based on zone alignment
            if align == "center":
                tx = block_cx - tw // 2
            elif align == "left":
                tx = max(safe_margin, block_cx - tw // 2)
            else:  # right
                tx = min(w - safe_margin - tw, block_cx - tw // 2)

            ty = y_cur

            # ── Drop shadow (multi-offset at low alpha for soft blur sim) ──
            shadow_alpha = int(alpha * 0.55)
            shadow_offsets = [(3, 4), (4, 5), (2, 3), (5, 6), (1, 2)]
            for sdx, sdy in shadow_offsets:
                a = int(shadow_alpha * (1 - (sdx - 1) / 6))
                draw.text((tx + sdx, ty + sdy), ln, font=font, fill=(0, 0, 0, a))

            # ── 1px dark stroke (charcoal at 50% opacity) ──────────────────
            stroke_alpha = int(alpha * 0.50)
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                draw.text((tx+dx, ty+dy), ln, font=font, fill=(17, 17, 17, stroke_alpha))

            # ── Main warm-white text ────────────────────────────────────────
            draw.text((tx, ty), ln, font=font, fill=(*text_rgb, alpha))

            y_cur += line_h

        return y_cur   # bottom of this block

    # ── Measure block heights to anchor at zone_y_frac ──────────────────────
    lines1 = wrap_to_width(line1, font1)
    lines2 = wrap_to_width(line2, font2)
    lh1 = int(_measure_text(font1, "Ag")[1] * 1.45)
    lh2 = int(_measure_text(font2, "Ag")[1] * 1.45)
    block1_h = len(lines1) * lh1
    block_gap = int(lh2 * 0.6)
    block2_h = len(lines2) * lh2
    total_block_h = block1_h + block_gap + block2_h

    # Place so the bottom of the whole block lands at zone_y_frac
    block_y_top = block_y - total_block_h
    block_y_top = max(int(h * 0.05), block_y_top)   # safe top margin

    # ── Gradient underlay ──────────────────────────────────────────────────
    max_alpha_so_far = max(alpha1, alpha2)
    if max_alpha_so_far > 0:
        _draw_caption_gradient(overlay,
                                0, block_y_top - 20,
                                w, block_y_top + total_block_h + 20,
                                max_alpha_so_far)

    # ── Draw Line1 ──────────────────────────────────────────────────────────
    bottom1 = draw_text_block(line1, font1, alpha1, float1, block_y_top, True)

    # ── Draw Line2 ──────────────────────────────────────────────────────────
    draw_text_block(line2, font2, alpha2, float2, bottom1 + block_gap, False)

    result = Image.alpha_composite(img, overlay).convert("RGB")
    final = np.asarray(result, dtype=np.uint8)
    # ── Emotion-based visual polish ────────────────────────────────
    if emotion == "Pain":
        final = _apply_grain(final, 0.05)
        
    elif emotion == "Ego":
        final = _apply_grain(final, 0.045)
        
    elif emotion == "Calm":
        final = _apply_grain(final, 0.025)
        final = _apply_bloom(final, 220, 4)
        
    elif emotion == "Growth":
        final = _apply_bloom(final, 215, 6)
        final = _apply_grain(final, 0.03)
        
    elif emotion == "Realization":
        final = _apply_bloom(final, 210, 5)
        final = _apply_grain(final, 0.035)
        
    else:
        final = _apply_grain(final, 0.035)
        
    return final


# ─── COLOR GRADE ──────────────────────────────────────────────────────────────

def _apply_grade(frame: np.ndarray, params: dict) -> np.ndarray:
    f   = frame.astype(np.float32) / 255.0
    # Brightness
    f  += params["brightness"]
    # Contrast (pivot around mid-grey)
    f   = (f - 0.5) * params["contrast"] + 0.5
    # Saturation via per-pixel luminance
    lum = (0.299*f[...,0] + 0.587*f[...,1] + 0.114*f[...,2])[..., np.newaxis]
    f   = lum + params["saturation"] * (f - lum)
    return np.clip(f * 255.0, 0, 255).astype(np.uint8)


# ─── SLOW ZOOM ────────────────────────────────────────────────────────────────

def _zoom_and_grade(frame: np.ndarray, t: float, params: dict) -> np.ndarray:
    """Scale frame up slightly (slow zoom-in) then center-crop back to original size."""
    safe_t = max(0.0, min(float(t), DURATION))
    scale  = 1.0 + 0.03 * (safe_t / DURATION)
    h, w   = frame.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)

    img  = Image.fromarray(frame, mode="RGB").resize((new_w, new_h), Image.LANCZOS)
    arr  = np.asarray(img, dtype=np.uint8)
    y0   = (new_h - h) // 2
    x0   = (new_w - w) // 2
    return _apply_grade(arr[y0:y0+h, x0:x0+w], params)


# ─── MAIN RENDER ──────────────────────────────────────────────────────────────

def render_short(
    video_path: str,
    line1: str,
    line2: str,
    emotion: str,
    output_path: str = "output/final_short.mp4",
    job_id: str = None,
    jobs_dict: dict = None,
) -> str:
    """
    Complete render pipeline.

    1. Load source video
    2. Trim to DURATION
    3. Resize + center-crop to TARGET_W × TARGET_H (1080 × 1920)
    4. Per-frame: slow zoom → color grade → PIL text burn
    5. Write H.264 MP4 (no audio)

    Pure PIL per-frame rendering. No MoviePy compositing layers.
    """
    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    params = GRADE_PARAMS.get(emotion, GRADE_PARAMS["Calm"])
    print(f"  [render] Loading: {video_path}")
    print(f"  [render] Emotion: {emotion} | Grade: {params}")

    # ── Load & trim ───────────────────────────────────────────────────────────
    src = VideoFileClip(str(video_path), audio=False)
    clip_dur = min(DURATION, src.duration)
    src = src.subclip(0, clip_dur)
    print(f"  [render] Source: {src.w}×{src.h}  dur={clip_dur:.1f}s  fps={src.fps}")

    # ── Compute crop box once (PIL resize per-frame — avoids ANTIALIAS bug) ──
    # moviepy's clip.resize() internally uses Image.ANTIALIAS which was removed
    # in Pillow 10. We do all resizing ourselves with PIL.LANCZOS.
    src_ratio    = src.w / src.h
    target_ratio = TARGET_W / TARGET_H   # 0.5625

    if src_ratio > target_ratio:
        # Wider than 9:16 → fit height, crop width
        _new_h = TARGET_H
        _new_w = int(src.w * (TARGET_H / src.h))
    else:
        # Taller / equal → fit width, crop height
        _new_w = TARGET_W
        _new_h = int(src.h * (TARGET_W / src.w))

    # Crop offsets inside the scaled frame
    _cx = (_new_w - TARGET_W) // 2
    _cy = (_new_h - TARGET_H) // 2

    print(f"  [render] Will scale raw frames to {_new_w}x{_new_h}, "
          f"then crop to {TARGET_W}x{TARGET_H}")

    # ── Single-pass frame function ────────────────────────────────────────────
    frame_count = int(DURATION * FPS)
    def make_frame(t: float) -> np.ndarray:
        if job_id and jobs_dict:
            frame_index = int(t * FPS)
            pct = int((frame_index / frame_count) * 100)
            jobs_dict[job_id]["progress"] = pct
            jobs_dict[job_id]["status"] = "rendering"
            
        raw = src.get_frame(min(t, clip_dur - 1/FPS))
        pil = Image.fromarray(raw, mode="RGB").resize((_new_w, _new_h), Image.LANCZOS)
        cropped = np.asarray(pil, dtype=np.uint8)[_cy:_cy+TARGET_H, _cx:_cx+TARGET_W]
            
        zoomed = _zoom_and_grade(cropped, t, params)
        final  = _render_text_onto_frame(zoomed, line1, line2, t)
        return final

    out_clip = VideoClip(make_frame, duration=DURATION)
    out_clip = out_clip.set_fps(FPS)



    # ── Export ────────────────────────────────────────────────────────────────
    logger = None
    if job_id and jobs_dict:
        logger = RenderLogger(job_id, jobs_dict)
    
    out_clip.write_videofile(
    str(output_path),
    fps=FPS,
    codec="libx264",
    audio=False,
    preset="fast",
    ffmpeg_params=["-crf", "20", "-pix_fmt", "yuv420p"],
    logger=logger,
    threads=2,
)

    src.close()
    out_clip.close()
    print(f"  [render] ✓ Saved → {output_path}")
    return str(output_path)



# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python cinematic_engine.py <video_path> [emotion]")
        sys.exit(1)
    vp  = sys.argv[1]
    emo = sys.argv[2] if len(sys.argv) > 2 else "Pain"
    out = render_short(vp, "I stopped chasing them.", "That's when they noticed me.", emo,
                       output_path="output/test_short.mp4")
    print("Output:", out)