from __future__ import annotations

from typing import Optional, Tuple

import os
import json

import time
import psutil
from win32gui import GetForegroundWindow
from win32process import GetWindowThreadProcessId
import threading
import mss
import cv2
import numpy as np
try:
    import pydirectinput as pdi  # from pydirectinput-rgx
    pdi.PAUSE = 0.0
except Exception:
    pdi = None
# --- auto-elevate as Administrator ---





# Crop configuration via external config only
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "crop_config.json")

# Brightness correction (applied to screenshots and recordings)
# gain < 1.0 will darken linearly; gamma > 1.0 will also darken mid/highlights
BRIGHTNESS_GAIN = 0.75
BRIGHTNESS_GAMMA = 1.25

# Preview toggle (True to show debug preview window, False for headless)
PREVIEW_ENABLED = True

# Multi-scale matching sweep
SCALE_MIN = 0.80
SCALE_MAX = 1.10
SCALE_STEP = 0.05

# Per-second best score accumulator
WINDOW_BEST = {"score": 0.0, "name": None, "loc": None, "scale": 1.0}
WINDOW_START_TS = time.time()

# Enable key actions after successful matches
KEY_ACTIONS_ENABLED = True


def apply_brightness_correction(bgr_image: np.ndarray) -> np.ndarray:
    """Darken image using linear gain and gamma to reduce perceived brightness."""

    if bgr_image is None or bgr_image.size == 0:
        return bgr_image
    img = bgr_image.astype(np.float32) / 255.0
    # Linear gain
    img *= float(BRIGHTNESS_GAIN)
    # Gamma correction (gamma>1 darkens)
    img = np.power(np.clip(img, 0.0, 1.0), float(BRIGHTNESS_GAMMA))
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)

 
def _preprocess_gray_for_match(gray: np.ndarray) -> np.ndarray:
    # No preprocessing needed for masked direct comparison (kept for compatibility)
    return gray


def _load_match_templates() -> list:
    """Load templates as BGR and use full-background masks for matching/overlay.

    - Keep actual BGR pixels (with their backgrounds)
    - Do not hollow out backgrounds; avoid Otsu masking to reduce false matches
    - Returned entries: (name, template_bgr, mask_uint8_full)
    """

    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    candidate_files = [
        os.path.join(templates_dir, "slice.png"),
        os.path.join(templates_dir, "defend.png"),
        os.path.join(templates_dir, "execute.png"),
    ]
    BLACK_THRESHOLD = 20  # kept for potential future use
    loaded = []  # entries: (name, tmpl_bgr, mask)
    for path in candidate_files:
        if os.path.exists(path):
            name = os.path.basename(path).lower()
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None or img.size == 0:
                continue
            # construct BGR template and full mask (keep background)
            if img.ndim == 3 and img.shape[2] == 4:
                tmpl_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif img.ndim == 3:
                tmpl_bgr = img.copy()
            else:
                tmpl_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            h, w = tmpl_bgr.shape[:2]
            mask = np.full((h, w), 255, dtype=np.uint8)
            loaded.append((os.path.basename(path), tmpl_bgr, mask))
    print(f"templates_loaded: {len(loaded)} from {templates_dir}")
    return loaded


def _match_templates_on_frame(
    bgr_frame: np.ndarray,
    loaded_templates: list,
    threshold: float = 0.6,
    debug: bool = True,
) -> tuple[dict, Optional[dict], dict]:
    """Run template matching per-template and return scaled templates for preview.

    - defend.png: fixed scale=0.6, threshold=0.6
    - exe.png: fixed scale=1.0 (current setting)
    - others: multi-scale exploration, use provided threshold
    Returns a dict name->scaled_template (grayscale, masked) for unified preview.
    """

    if not loaded_templates:
        if debug:
            print("match_debug: no templates loaded, skip")
        return {}, None, {}

    # Use raw color frame and gradient magnitude for matching (robust to brightness)
    frame_u8 = bgr_frame
    frame_gray = cv2.cvtColor(frame_u8, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(frame_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(frame_gray, cv2.CV_32F, 0, 1, ksize=3)
    frame_grad = cv2.magnitude(gx, gy)
    # Normalize to 0-255 uint8 for template matching
    frame_grad_u8 = cv2.normalize(frame_grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if debug:
        # match_debug disabled per request
        pass

    # Track best score across all templates/scales for this frame
    scaled_for_preview: dict[str, np.ndarray] = {}
    best_score: float = 0.0
    best_name: Optional[str] = None
    best_loc: Optional[Tuple[int, int]] = None
    best_scale: float = 1.0
    best_overlay: Optional[dict] = None
    per_template_best: dict[str, dict] = {}

    for entry in loaded_templates:
        name, tmpl_bgr, tmpl_mask = entry
        th, tw = tmpl_bgr.shape[:2]
        # Evaluate defend/execute/slice; apply per-template overrides from config
        nm = name.lower()
        if nm not in ("defend.png", "execute.png", "slice.png"):
            continue
        scale_val = None
        thresh_val = None
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f) or {}
            m_all = cfg.get("matching", {})
            key = (
                "defend" if nm == "defend.png" else
                ("execute" if nm == "execute.png" else
                 ("slice" if nm == "slice.png" else None))
            )
            if key:
                m = m_all.get(key, {})
                if isinstance(m.get("scale"), (int, float)):
                    scale_val = float(m["scale"])
                if isinstance(m.get("threshold"), (int, float)):
                    thresh_val = float(m["threshold"])
                hotkey = m.get("hotkey")
        except Exception:
            pass
        if nm == "defend.png":
            scales = [scale_val if scale_val is not None else 0.53]
            local_threshold = thresh_val if thresh_val is not None else 0.8
        elif nm == "execute.png":
            scales = [scale_val if scale_val is not None else 1.0]
            local_threshold = thresh_val if thresh_val is not None else 0.8
        else:  # slice.png
            scales = [scale_val if scale_val is not None else 1.0]
            local_threshold = thresh_val if thresh_val is not None else 0.8

        for s in scales:
            # Scale based on template's own dimensions (not frame ratio)
            scale = max(float(s), 1e-3)
            new_w = max(1, int(round(tw * scale)))
            new_h = max(1, int(round(th * scale)))
            work_tmpl = cv2.resize(tmpl_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # Build grayscale preview masked to content regions
            gray_preview = cv2.cvtColor(work_tmpl, cv2.COLOR_BGR2GRAY)
            # Compute gradient magnitude for template
            tgx = cv2.Sobel(gray_preview, cv2.CV_32F, 1, 0, ksize=3)
            tgy = cv2.Sobel(gray_preview, cv2.CV_32F, 0, 1, ksize=3)
            tmpl_grad = cv2.magnitude(tgx, tgy)
            tmpl_grad_u8 = cv2.normalize(tmpl_grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            mask_resized = cv2.resize(tmpl_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            # Make masked-out regions white (255) for better visibility in preview
            gray_preview_bgwhite = gray_preview.copy()
            # With full mask, preview is unchanged; keep logic for consistency
            if mask_resized is not None:
                gray_preview_bgwhite[mask_resized == 0] = 255
            # Store latest scaled preview image (grayscale)
            scaled_for_preview[name] = gray_preview_bgwhite
            if new_h > frame_u8.shape[0] or new_w > frame_u8.shape[1]:
                continue
            # Defend uses direct color matching on BGR with TM_CCORR_NORMED
            work_tmpl_bgr = work_tmpl
            if work_tmpl_bgr.ndim == 2:
                work_tmpl_bgr = cv2.cvtColor(work_tmpl_bgr, cv2.COLOR_GRAY2BGR)
            res = cv2.matchTemplate(frame_u8, work_tmpl_bgr, cv2.TM_CCOEFF_NORMED)
            _, maxVal, _, maxLoc = cv2.minMaxLoc(res)

            # Update current best score for this frame
            if maxVal > best_score:
                best_score = maxVal
                best_name = name
                best_loc = maxLoc
                best_scale = s
                best_overlay = {
                    "name": name,
                    "loc": maxLoc,
                    "scale": s,
                    "tmpl": work_tmpl,
                    "mask": mask_resized,
                    "score": maxVal,
                }
            # Track best per-template
            prev = per_template_best.get(name)
            if prev is None or maxVal > prev.get("score", 0.0):
                per_template_best[name] = {
                    "name": name,
                    "loc": maxLoc,
                    "scale": s,
                    "tmpl": work_tmpl,
                    "mask": mask_resized,
                    "score": maxVal,
                }
            if debug:
                # print disabled per request
                pass

            # Threshold-based: trigger only when >= local_threshold
            if maxVal >= local_threshold:
                if KEY_ACTIONS_ENABLED:
                    # Use configured hotkey if present, else fall back by template name
                    if 'hotkey' in locals() and isinstance(hotkey, str) and len(hotkey) > 0:
                        _trigger_key_action_async(name, hotkey)
                    else:
                        _trigger_key_action_async(name)

    # Aggregate per-second best and print
    global WINDOW_START_TS, WINDOW_BEST
    now_ts = time.time()
    if best_name is not None and best_score > WINDOW_BEST.get("score", 0.0):
        WINDOW_BEST = {"score": best_score, "name": best_name, "loc": best_loc, "scale": best_scale}
    if now_ts - WINDOW_START_TS >= 1.0:
        b = WINDOW_BEST
        if b.get("name") is not None:
            print(f"best: {b['score']:.3f} ({b['name']}) at {b['loc']} scale={b['scale']:.2f}")
        WINDOW_BEST = {"score": 0.0, "name": None, "loc": None, "scale": 1.0}
        WINDOW_START_TS = now_ts
    # Return data; 仅暴露当前 best 的模板数据用于左侧叠加
    best_only = {best_name: per_template_best.get(best_name)} if best_name in per_template_best else {}
    return scaled_for_preview, best_overlay, best_only


def _trigger_key_action_async(name: str, override_key: Optional[str] = None) -> None:
    if pdi is None:
        return
    n = name.lower()
    k = (override_key or ("e" if n == "defend.png" else "f")).lower()
    def worker():
        try:
            print(f"press: {k}")
            pdi.press(k, _pause=False)
        except Exception:
            pass
    threading.Thread(target=worker, daemon=True).start()


def _compose_unified_preview(frame_bgr: np.ndarray, scaled: dict) -> np.ndarray:
    """Stack video frame with scaled templates into one preview image. Hide slice."""
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr
    tiles: list[np.ndarray] = []
    for name, tmpl in scaled.items():
        # convert tmpl (gray) to BGR for stacking
        if tmpl.ndim == 2:
            tile = cv2.cvtColor(tmpl, cv2.COLOR_GRAY2BGR)
        else:
            tile = tmpl
        tiles.append(tile)

    # If no templates, just return the frame
    if not tiles:
        return frame_bgr

    # Make a simple vertical stack of templates on the right with padding
    max_w = max(t.shape[1] for t in tiles)
    total_h = sum(t.shape[0] for t in tiles)
    pad = 6
    total_h += pad * (len(tiles) - 1)
    # Create a white canvas for right column
    right = np.full((total_h, max_w, 3), 255, dtype=np.uint8)
    y = 0
    for t in tiles:
        h, w = t.shape[:2]
        right[y:y+h, 0:w] = t
        y += h + pad

    # Match heights by padding the shorter one
    h_left, w_left = frame_bgr.shape[:2]
    h_right, w_right = right.shape[:2]
    H = max(h_left, h_right)
    def pad_to_h(img: np.ndarray, H: int) -> np.ndarray:
        if img.shape[0] == H:
            return img
        pad_h = H - img.shape[0]
        bottom = np.full((pad_h, img.shape[1], img.shape[2]), 255, dtype=img.dtype)
        return np.vstack([img, bottom])

    left_padded = pad_to_h(frame_bgr, H)
    right_padded = pad_to_h(right, H)
    # Add a small white separator
    sep = np.full((H, 4, 3), 255, dtype=np.uint8)
    combined = np.hstack([left_padded, sep, right_padded])
    return combined


def _overlay_best_match(frame_bgr: np.ndarray, overlay: Optional[dict]) -> np.ndarray:
    """Blend best-match template onto frame at detected location for debugging.

    - Draws a green rectangle around the match area
    - Alpha blends the template (masked) onto the frame for visual verification
    """
    if overlay is None or frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr
    x, y = overlay.get("loc", (0, 0))
    tmpl: np.ndarray = overlay.get("tmpl")
    mask: np.ndarray = overlay.get("mask")
    if tmpl is None or mask is None:
        return frame_bgr
    h, w = tmpl.shape[:2]
    H, W = frame_bgr.shape[:2]
    if x < 0 or y < 0 or x + w > W or y + h > H:
        return frame_bgr

    out = frame_bgr.copy()
    # rectangle + label with scale and score if present
    score = overlay.get("score") if overlay else None
    # Green by default; Red when score >= threshold (0.7)
    color = (0, 0, 255) if (isinstance(score, (int, float)) and score >= 0.7) else (0, 255, 0)
    cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    label = None
    if overlay.get("scale") is not None or score is not None:
        sc = overlay.get("scale")
        sv = score
        if sc is not None and sv is not None:
            label = f"{sc:.2f} | {sv:.3f}"
        elif sc is not None:
            label = f"{sc:.2f}"
        elif sv is not None:
            label = f"{sv:.3f}"
    if label:
        cv2.putText(out, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    # alpha blend masked region
    roi = out[y:y+h, x:x+w]
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    blended = (roi.astype(np.float32) * (1.0 - 0.5 * alpha) + tmpl.astype(np.float32) * (0.5 * alpha)).astype(np.uint8)
    out[y:y+h, x:x+w] = blended
    return out


def _overlay_per_template(frame_bgr: np.ndarray, per_template: dict) -> np.ndarray:
    """Draw colored boxes for each template's best match and lightly blend template.

    Colors:
    - slice.png: blue (255, 0, 0)
    - defend.png: green (0, 255, 0)
    - execute.png: red (0, 0, 255)
    """
    if frame_bgr is None or frame_bgr.size == 0 or not per_template:
        return frame_bgr
    out = frame_bgr.copy()
    H, W = out.shape[:2]
    for name, data in per_template.items():
        x, y = data.get("loc", (0, 0))
        tmpl: np.ndarray = data.get("tmpl")
        mask: np.ndarray = data.get("mask")
        if tmpl is None or mask is None:
            continue
        h, w = tmpl.shape[:2]
        if x < 0 or y < 0 or x + w > W or y + h > H:
            continue
        # Green by default; Red when score >= threshold (0.7)
        score = data.get("score")
        color = (0, 0, 255) if (isinstance(score, (int, float)) and score >= 0.7) else (0, 255, 0)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        # very light blend to preserve FPS
        roi = out[y:y+h, x:x+w]
        alpha = (mask.astype(np.float32) / 255.0)[..., None] * 0.3
        roi[:] = (roi.astype(np.float32) * (1.0 - alpha) + tmpl.astype(np.float32) * alpha).astype(np.uint8)
    return out

def load_crop_config() -> Tuple[float, float, float]:
    """Load crop config; prefer current resolution match, else default."""
    # Minimal inline fallback (used only if config missing)
    size_frac = 0.09
    vertical_bias = 0.11
    height_frac = 2

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f) or {}
    except Exception:
        # Print active parameters with no config
        try:
            with mss.mss() as sct:
                mon = sct.monitors[1]
                mon_w, mon_h = int(mon.get("width", 0)), int(mon.get("height", 0))
                print(f"resolution: {mon_w}x{mon_h}")
                print("message: No preset file found. Loaded default config.")
                print(f"size_fraction: {size_frac}")
                print(f"vertical_bias: {vertical_bias}")
                print(f"height_fraction: {height_frac}")
        except Exception:
            pass
        return size_frac, vertical_bias, height_frac

    default = cfg.get("default", {})
    res_map = cfg.get("resolutions", {})

    # Detect current primary monitor resolution via mss later in call site
    def resolve_for(mon_w: int, mon_h: int) -> Tuple[float, float, float]:
        key = f"{mon_w}x{mon_h}"
        res_cfg = res_map.get(key) or {}
        base = res_cfg if res_cfg else default
        params = (
            float(base.get("size_fraction", size_frac)),
            float(base.get("vertical_bias", vertical_bias)),
            float(base.get("height_fraction", height_frac)),
        )
        # Print which group is applied at load-time
        if res_cfg:
            print(f"resolution: {mon_w}x{mon_h}")
            print("message: Loaded resolution preset config.")
        else:
            print(f"resolution: {mon_w}x{mon_h}")
            print("message: Preset not found for resolution. Loaded default config.")
        print(f"size_fraction: {params[0]}")
        print(f"vertical_bias: {params[1]}")
        print(f"height_fraction: {params[2]}")
        return params

    return size_frac, vertical_bias, height_frac, cfg, resolve_for



def get_foreground_pid() -> Optional[int]:
    """Return PID of the foreground window's process, or None if unavailable."""

    hwnd = GetForegroundWindow()
    if not hwnd:
        return None
    try:
        _, pid = GetWindowThreadProcessId(hwnd)
        return int(pid) if pid else None
    except Exception:
        return None


def get_process_image_name(pid: int) -> Optional[str]:
    """Return the basename (lowercase) of the process image for the given PID."""

    try:
        proc = psutil.Process(pid)
        return proc.name().lower()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None


def any_process_named(process_name: str) -> bool:
    """Return True if any running process has the given image name (case-insensitive)."""

    target = process_name.lower()
    for proc in psutil.process_iter(["name"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if name == target:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return False


def is_process_foreground(process_name: str) -> bool:
    """Return True if the foreground process image name matches process_name."""

    pid = get_foreground_pid()
    if pid is None:
        return False
    image = get_process_image_name(pid)
    if image is None:
        return False
    return image == process_name.lower()


def nested_check_yysls() -> Tuple[bool, bool]:
    """Nested check: (exists_running, is_foreground)."""

    name = "yysls.exe"
    exists_running = any_process_named(name)
    if not exists_running:
        return False, False

    active_foreground = is_process_foreground(name)
    return True, active_foreground


def show_screenshot_window() -> None:
    """Capture the primary monitor and display the screenshot in a window."""

    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary monitor
        mon_left = int(monitor.get("left", 0))
        mon_top = int(monitor.get("top", 0))
        mon_w = int(monitor.get("width", 0))
        mon_h = int(monitor.get("height", 0))

        # Load config and resolve params for current resolution
        _, _, _, _, resolve_for = load_crop_config()
        size_f, vertical_b, height_f = resolve_for(mon_w, mon_h)

        # Compute crop width and height, with biased center
        side = int(max(1, min(mon_w, mon_h) * float(size_f)))
        crop_h = int(max(1, side * float(height_f)))
        center_x = mon_left + mon_w // 2
        center_y = mon_top + mon_h // 2 + int(mon_h * float(vertical_b))

        # Top-left of the rectangle, clamped to monitor bounds
        left = int(center_x - side // 2)
        top = int(center_y - crop_h // 2)
        left = max(mon_left, min(left, mon_left + mon_w - side))
        top = max(mon_top, min(top, mon_top + mon_h - crop_h))

        region = {"left": left, "top": top, "width": side, "height": crop_h}

        shot = sct.grab(region)
        bgr8 = np.array(shot)
        bgr8 = cv2.cvtColor(bgr8, cv2.COLOR_BGRA2BGR)
        bgr8 = apply_brightness_correction(bgr8)

        # Save raw screenshot before processing
        os.makedirs("results", exist_ok=True)
        raw_path = os.path.join("results", "raw_screenshot.png")
        cv2.imwrite(raw_path, bgr8)

        # Show and save raw only (no HDR processing)
        cv2.imshow("Screenshot", bgr8)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def record_screenshot_video(
    duration_seconds: Optional[int] = None,
    fps: int = 20,
) -> None:
    """
    Debug helper: preview the configured screen region as a video-like window only.
    Does not save to disk. Press 'q' in the preview window to stop.
    """

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        mon_left = int(monitor.get("left", 0))
        mon_top = int(monitor.get("top", 0))
        mon_w = int(monitor.get("width", 0))
        mon_h = int(monitor.get("height", 0))

        _, _, _, _, resolve_for = load_crop_config()
        size_f, vertical_b, height_f = resolve_for(mon_w, mon_h)
        side = int(max(1, min(mon_w, mon_h) * float(size_f)))
        crop_h = int(max(1, side * float(height_f)))
        center_x = mon_left + mon_w // 2
        center_y = mon_top + mon_h // 2 + int(mon_h * float(vertical_b))

        left = int(center_x - side // 2)
        top = int(center_y - crop_h // 2)
        left = max(mon_left, min(left, mon_left + mon_w - side))
        top = max(mon_top, min(top, mon_top + mon_h - crop_h))

        region = {"left": left, "top": top, "width": side, "height": crop_h}

        # Load templates once
        loaded_templates = _load_match_templates()

        match_threshold = 0.8
        match_interval_sec = 0.2
        last_match_time = 0.0

        try:
            if duration_seconds is None:
                # Record until 'q' pressed or interrupted
                while True:
                    frame_bgra = sct.grab(region)
                    frame = np.array(frame_bgra)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    frame = apply_brightness_correction(frame)

                    # Template matching at fixed interval
                    now = time.time()
                    if loaded_templates and (now - last_match_time) >= match_interval_sec:
                        last_match_time = now
                        latest_scaled, best_overlay, per_template = _match_templates_on_frame(frame, loaded_templates, match_threshold, debug=True)
                    else:
                        best_overlay = None
                        per_template = {}

                    frame_with_overlay = _overlay_best_match(frame, best_overlay)
                    # Show only the current best template in the right preview column
                    if best_overlay and best_overlay.get("name") in latest_scaled:
                        preview_scaled = {best_overlay.get("name"): latest_scaled[best_overlay.get("name")]}
                    else:
                        preview_scaled = {}
                    preview = _compose_unified_preview(frame_with_overlay, preview_scaled)
                    cv2.imshow("Preview", preview)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                total_frames = int(max(1, duration_seconds) * max(1, fps))
                for _ in range(total_frames):
                    frame_bgra = sct.grab(region)
                    frame = np.array(frame_bgra)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    frame = apply_brightness_correction(frame)

                    now = time.time()
                    if loaded_templates and (now - last_match_time) >= match_interval_sec:
                        last_match_time = now
                        latest_scaled, best_overlay, per_template = _match_templates_on_frame(frame, loaded_templates, match_threshold, debug=True)
                    else:
                        best_overlay = None
                        per_template = {}

                    frame_with_overlay = _overlay_best_match(frame, best_overlay)
                    if best_overlay and best_overlay.get("name") in latest_scaled:
                        preview_scaled = {best_overlay.get("name"): latest_scaled[best_overlay.get("name")]}
                    else:
                        preview_scaled = {}
                    preview = _compose_unified_preview(frame_with_overlay, preview_scaled)
                    cv2.imshow("Preview", preview)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            cv2.destroyAllWindows()

# Headless detection loop (no preview). Triggers actions on matches
def detect_loop(duration_seconds: Optional[int] = None, preview: Optional[bool] = None) -> None:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        mon_left = int(monitor.get("left", 0))
        mon_top = int(monitor.get("top", 0))
        mon_w = int(monitor.get("width", 0))
        mon_h = int(monitor.get("height", 0))

        _, _, _, _, resolve_for = load_crop_config()
        size_f, vertical_b, height_f = resolve_for(mon_w, mon_h)
        side = int(max(1, min(mon_w, mon_h) * float(size_f)))
        crop_h = int(max(1, side * float(height_f)))
        center_x = mon_left + mon_w // 2
        center_y = mon_top + mon_h // 2 + int(mon_h * float(vertical_b))

        left = int(center_x - side // 2)
        top = int(center_y - crop_h // 2)
        left = max(mon_left, min(left, mon_left + mon_w - side))
        top = max(mon_top, min(top, mon_top + mon_h - crop_h))

        region = {"left": left, "top": top, "width": side, "height": crop_h}

        loaded_templates = _load_match_templates()
        match_threshold = 0.7
        match_interval_sec = 0.1
        last_match_time = 0.0
        latest_scaled: dict = {}
        if preview is None:
            preview = PREVIEW_ENABLED

        start = time.time()
        while True:
            frame_bgra = sct.grab(region)
            frame = np.array(frame_bgra)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame = apply_brightness_correction(frame)

            now = time.time()
            if loaded_templates and (now - last_match_time) >= match_interval_sec:
                last_match_time = now
                latest_scaled, best_overlay, per_template = _match_templates_on_frame(frame, loaded_templates, match_threshold, debug=False)

            # show preview window for debugging (video + scaled templates)
            if preview:
                frame_with_overlay = _overlay_per_template(frame, per_template)
                preview_img = _compose_unified_preview(frame_with_overlay, latest_scaled)
                cv2.imshow("Preview", preview_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if duration_seconds is not None and (now - start) >= duration_seconds:
                break
        if preview:
            cv2.destroyAllWindows()

# (HDR-related code removed by request)


if __name__ == "__main__":
    exists, active = nested_check_yysls()
    print(exists)
    print(active)
    # record_screenshot_video()  # preview disabled per request
    detect_loop()
    # show_screenshot_window()


