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
import pydirectinput as pdi
import pystray
from PIL import Image, ImageDraw
import sys
pdi.PAUSE = 0.0

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "crop_config.json")

# Application state class
class AppState:
    def __init__(self):
        self.config = {}
        self.preview_enabled = False
        self.key_actions_enabled = True
        self.running = True
        self.tray_icon = None
        self.detection_thread = None
        self.template_cache = {}

# Global application state instance
app_state = AppState()



def apply_brightness_correction(bgr_image: np.ndarray) -> np.ndarray:
    """Apply linear gain and gamma to reduce perceived brightness.

    Parameters
    - bgr_image: Input image in BGR8 format.

    Returns
    - Brightness-adjusted BGR8 image.
    """

    if bgr_image is None or bgr_image.size == 0:
        return bgr_image
    img = bgr_image.astype(np.float32) / 255.0
    settings = (app_state.config.get("settings") or {}) if isinstance(app_state.config, dict) else {}
    gain = float(settings.get("brightness_gain", 1.0))
    gamma = float(settings.get("brightness_gamma", 1.0))
    img *= gain
    img = np.power(np.clip(img, 0.0, 1.0), gamma)
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)

 
def _load_match_templates() -> list:
    """Load available templates as BGR and 8-bit masks.

    Returns
    - List of tuples: (filename, template_bgr, mask_uint8)
    """

    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    candidate_files = [
        os.path.join(templates_dir, "slice.png"),
        os.path.join(templates_dir, "defend.png"),
        os.path.join(templates_dir, "execute.png"),
    ]
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
    return loaded


## Pre-scaled per-template cache (built once per run based on config)

def _build_template_cache(loaded_templates: list) -> None:
    """Build and cache per-template resized assets and parameters.

    Caches for each template name:
    - tmpl_bgr_resized: resized BGR template for overlay
    - tmpl_gray: resized grayscale template for matching
    - mask: resized uint8 mask
    - preview: grayscale preview with background set to white
    - scale, threshold, hotkey: parameters from config (with defaults)
    """
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f) or {}
    except Exception:
        cfg = {}
    m_all = cfg.get("matching", {})

    for entry in loaded_templates:
        name, tmpl_bgr, tmpl_mask = entry
        nm = name.lower()
        if nm not in ("defend.png", "execute.png", "slice.png"):
            continue
        key = (
            "defend" if nm == "defend.png" else
            ("execute" if nm == "execute.png" else
             ("slice" if nm == "slice.png" else None))
        )
        m = m_all.get(key, {}) if key else {}
        default_scale = 0.53 if nm == "defend.png" else 1.0
        scale = float(m.get("scale", default_scale))
        threshold = float(m.get("threshold", 0.8))
        hotkey = m.get("hotkey")

        th, tw = tmpl_bgr.shape[:2]
        new_w = max(1, int(round(tw * max(scale, 1e-3))))
        new_h = max(1, int(round(th * max(scale, 1e-3))))

        resized_bgr = cv2.resize(tmpl_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(tmpl_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        tmpl_gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
        gray_preview = tmpl_gray.copy()
        if resized_mask is not None:
            gray_preview[resized_mask == 0] = 255

        app_state.template_cache[name] = {
            "tmpl_bgr_resized": resized_bgr,
            "tmpl_gray": tmpl_gray,
            "mask": resized_mask,
            "preview": gray_preview,
            "scale": scale,
            "threshold": threshold,
            "hotkey": hotkey,
        }

def _match_templates_on_frame(
    bgr_frame: np.ndarray,
    loaded_templates: list,
) -> tuple[dict, Optional[dict], dict]:
    """Match configured templates on a single BGR frame using grayscale CCOEFF_NORMED.

    Returns:
    - scaled_for_preview: dict[name -> gray template preview]
    - best_overlay: optional metadata for the globally best match
    - per_template_best_best_only: dict[name -> best match metadata] (best only)
    """

    if not loaded_templates:
        return {}, None, {}

    frame_u8 = bgr_frame
    frame_gray = cv2.cvtColor(frame_u8, cv2.COLOR_BGR2GRAY)

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
        nm = name.lower()
        if nm not in ("defend.png", "execute.png", "slice.png"):
            continue
        # Ensure cache exists for this template
        cache = app_state.template_cache.get(name)
        if cache is None:
            _build_template_cache([entry])
            cache = app_state.template_cache.get(name)
        if cache is None:
            continue

        work_tmpl_bgr = cache["tmpl_bgr_resized"]
        work_tmpl_gray = cache["tmpl_gray"]
        mask_resized = cache["mask"]
        local_threshold = float(cache.get("threshold", 0.8))
        hotkey = cache.get("hotkey")

        h_t, w_t = work_tmpl_gray.shape[:2]
        scaled_for_preview[name] = cache["preview"]
        if h_t > frame_gray.shape[0] or w_t > frame_gray.shape[1]:
            continue

        # Grayscale normalized correlation
        res = cv2.matchTemplate(frame_gray, work_tmpl_gray, cv2.TM_CCOEFF_NORMED)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(res)

        # Update current best score for this frame
        if maxVal > best_score:
            best_score = maxVal
            best_name = name
            best_loc = maxLoc
            best_scale = float(cache.get("scale", 1.0))
            best_overlay = {
                "name": name,
                "loc": maxLoc,
                "scale": best_scale,
                "tmpl": work_tmpl_bgr,
                "mask": mask_resized,
                "score": maxVal,
                "threshold": local_threshold,
            }
        # Track best per-template
        prev = per_template_best.get(name)
        if prev is None or maxVal > prev.get("score", 0.0):
            per_template_best[name] = {
                "name": name,
                "loc": maxLoc,
                "scale": best_scale,
                "tmpl": work_tmpl_bgr,
                "mask": mask_resized,
                "score": maxVal,
                "threshold": local_threshold,
            }
        
        # Threshold-based key action
        if maxVal >= local_threshold:
            if app_state.key_actions_enabled:
                if 'hotkey' in locals() and isinstance(hotkey, str) and len(hotkey) > 0:
                    _trigger_key_action_async(name, hotkey)
                else:
                    _trigger_key_action_async(name)

    # 仅暴露当前 best 的模板数据用于左侧叠加
    best_only = {best_name: per_template_best.get(best_name)} if best_name in per_template_best else {}
    return scaled_for_preview, best_overlay, best_only


def _trigger_key_action_async(name: str, override_key: Optional[str] = None) -> None:
    """Send a configured key asynchronously when yysls.exe is foreground.

    Parameters
    - name: Template filename identifying the action.
    - override_key: Optional explicit key to send; falls back by template.
    """
    if pdi is None:
        return
    if not is_process_foreground("yysls.exe"):
        return
    n = name.lower()
    k = (override_key or ("e" if n == "defend.png" else "f")).lower()
    def worker():
        try:
            if not is_process_foreground("yysls.exe"):
                return
            print(f"press: {k}")
            pdi.press(k, _pause=False)
        except Exception:
            pass
    threading.Thread(target=worker, daemon=True).start()


def _compose_unified_preview(frame_bgr: np.ndarray, scaled: dict) -> np.ndarray:
    """Compose frame and scaled template previews into a single image."""
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


def _overlay_per_template(frame_bgr: np.ndarray, per_template: dict) -> np.ndarray:
    """Render per-template best match rectangles and light template blends."""
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
        score = data.get("score")
        thr = data.get("threshold", 0.7)
        color = (0, 0, 255) if (isinstance(score, (int, float)) and isinstance(thr, (int, float)) and score >= float(thr)) else (0, 255, 0)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        # very light blend to preserve FPS
        roi = out[y:y+h, x:x+w]
        alpha = (mask.astype(np.float32) / 255.0)[..., None] * 0.3
        roi[:] = (roi.astype(np.float32) * (1.0 - alpha) + tmpl.astype(np.float32) * alpha).astype(np.uint8)
    return out

def load_crop_config() -> Tuple[float, float, float]:
    """Load crop parameters and provide a resolver for current resolution."""
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

    # settings overrides
    settings = cfg.get("settings", {})
    app_state.config = cfg
    app_state.preview_enabled = bool(settings.get("preview_enabled", False))
    app_state.key_actions_enabled = bool(settings.get("key_actions_enabled", True))
    fps = int(settings.get("fps", 10))

    res_map = cfg.get("resolutions", {})
    default = res_map.get("default", {})

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

    return size_frac, vertical_bias, height_frac, cfg, resolve_for, fps



def get_foreground_pid() -> Optional[int]:
    """Get PID for the foreground window process, if available."""

    hwnd = GetForegroundWindow()
    if not hwnd:
        return None
    try:
        _, pid = GetWindowThreadProcessId(hwnd)
        return int(pid) if pid else None
    except Exception:
        return None


def get_process_image_name(pid: int) -> Optional[str]:
    """Get lowercase image name for the given PID, or None on failure."""

    try:
        proc = psutil.Process(pid)
        return proc.name().lower()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None


def any_process_named(process_name: str) -> bool:
    """Check if any running process matches the given image name (case-insensitive)."""

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
    """Determine whether the foreground process name equals the provided name."""

    pid = get_foreground_pid()
    if pid is None:
        return False
    image = get_process_image_name(pid)
    if image is None:
        return False
    return image == process_name.lower()


def nested_check_yysls() -> Tuple[bool, bool]:
    """Check whether yysls.exe exists and whether it is foreground."""

    name = "yysls.exe"
    exists_running = any_process_named(name)
    if not exists_running:
        return False, False

    active_foreground = is_process_foreground(name)
    return True, active_foreground


def create_tray_icon():
    """Create system tray icon"""
    # Create a simple icon
    image = Image.new('RGB', (64, 64), color='blue')
    draw = ImageDraw.Draw(image)
    draw.ellipse([16, 16, 48, 48], fill='white', outline='black')
    draw.text((20, 20), "自动QTE", fill='black')
    
    # Create menu with dynamic status
    menu = pystray.Menu(
        pystray.MenuItem("状态: 运行中", lambda: None),
        pystray.MenuItem(lambda text: f"切换预览 ({'开' if app_state.preview_enabled else '关'})", toggle_preview),
        pystray.MenuItem(lambda text: f"切换按键操作 ({'开' if app_state.key_actions_enabled else '关'})", toggle_key_actions),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("退出", quit_app)
    )
    
    return pystray.Icon("yysls-opencv-template", image, "YYSLS OpenCV Template", menu)

def toggle_preview(icon, item):
    """Toggle preview window"""
    app_state.preview_enabled = not app_state.preview_enabled
    print(f"预览窗口: {'开启' if app_state.preview_enabled else '关闭'}")
    # Update menu to reflect new status
    update_tray_menu()

def toggle_key_actions(icon, item):
    """Toggle key actions"""
    app_state.key_actions_enabled = not app_state.key_actions_enabled
    print(f"按键操作: {'开启' if app_state.key_actions_enabled else '关闭'}")
    # Update menu to reflect new status
    update_tray_menu()

def update_tray_menu():
    """Update tray menu to reflect current status"""
    if app_state.tray_icon:
        # Create new menu with updated status
        menu = pystray.Menu(
            pystray.MenuItem("状态: 运行中", lambda: None),
            pystray.MenuItem(lambda text: f"切换预览 ({'开' if app_state.preview_enabled else '关'})", toggle_preview),
            pystray.MenuItem(lambda text: f"切换按键操作 ({'开' if app_state.key_actions_enabled else '关'})", toggle_key_actions),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出", quit_app)
        )
        app_state.tray_icon.menu = menu

def quit_app(icon, item):
    """Quit application"""
    app_state.running = False
    icon.stop()
    sys.exit(0)

def detect_loop_tray() -> None:
    """Run the main capture/match loop with tray control."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        mon_left = int(monitor.get("left", 0))
        mon_top = int(monitor.get("top", 0))
        mon_w = int(monitor.get("width", 0))
        mon_h = int(monitor.get("height", 0))

        _, _, _, _, resolve_for, fps = load_crop_config()
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
        _build_template_cache(loaded_templates)
        interval = 1.0 / max(1, fps)
        last_match_time = 0.0
        latest_scaled: dict = {}
        per_template: dict = {}

        start = time.time()
        while app_state.running:
            frame_bgra = sct.grab(region)
            frame = np.array(frame_bgra)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame = apply_brightness_correction(frame)

            loop_start = time.time()
            now = loop_start
            if loaded_templates and (now - last_match_time) >= interval:
                last_match_time = now
                latest_scaled, best_overlay, per_template = _match_templates_on_frame(frame, loaded_templates)

            # show preview window for debugging (video + scaled templates)
            if app_state.preview_enabled:
                frame_with_overlay = _overlay_per_template(frame, per_template)
                preview_img = _compose_unified_preview(frame_with_overlay, latest_scaled)
                cv2.imshow("Preview", preview_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # pacing both capture and matching to the same fps
            sleep_s = interval - (time.time() - loop_start)
            if sleep_s > 0:
                time.sleep(sleep_s)
        
        if app_state.preview_enabled:
            cv2.destroyAllWindows()

def detect_loop(duration_seconds: Optional[int] = None, preview: Optional[bool] = None) -> None:
    """Run the main capture/match loop with optional preview window."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        mon_left = int(monitor.get("left", 0))
        mon_top = int(monitor.get("top", 0))
        mon_w = int(monitor.get("width", 0))
        mon_h = int(monitor.get("height", 0))

        _, _, _, _, resolve_for, fps = load_crop_config()
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
        _build_template_cache(loaded_templates)
        interval = 1.0 / max(1, fps)
        last_match_time = 0.0
        latest_scaled: dict = {}
        per_template: dict = {}
        if preview is None:
            preview = app_state.preview_enabled

        start = time.time()
        while True:
            frame_bgra = sct.grab(region)
            frame = np.array(frame_bgra)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame = apply_brightness_correction(frame)

            loop_start = time.time()
            now = loop_start
            if loaded_templates and (now - last_match_time) >= interval:
                last_match_time = now
                latest_scaled, best_overlay, per_template = _match_templates_on_frame(frame, loaded_templates)

            # show preview window for debugging (video + scaled templates)
            if preview:
                frame_with_overlay = _overlay_per_template(frame, per_template)
                preview_img = _compose_unified_preview(frame_with_overlay, latest_scaled)
                cv2.imshow("Preview", preview_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if duration_seconds is not None and (now - start) >= duration_seconds:
                break
            # pacing both capture and matching to the same fps
            sleep_s = interval - (time.time() - loop_start)
            if sleep_s > 0:
                time.sleep(sleep_s)
        if preview:
            cv2.destroyAllWindows()

def show_startup_notification():
    """Show system notification when application starts"""
    try:
        # Use Windows message box for notification
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0, 
            "自动QTE检测已启动，可通过系统托盘控制", 
            "YYSLS OpenCV Template", 
            0x40  # MB_ICONINFORMATION
        )
    except Exception as e:
        print(f"通知发送失败: {e}")
        # Fallback to simple print
        print("🔔 YYSLS OpenCV Template - 自动QTE检测已启动，可通过系统托盘控制")
        print("📝 版本: 1.0.1 - 测试GitHub Actions构建")

def main():
    """Main function with tray icon"""
    # Show startup notification
    show_startup_notification()
    
    # Create and start tray icon
    app_state.tray_icon = create_tray_icon()
    
    # Start detection in a separate thread
    app_state.detection_thread = threading.Thread(target=detect_loop_tray, daemon=True)
    app_state.detection_thread.start()
    
    # Run tray icon (this blocks until quit)
    app_state.tray_icon.run()

if __name__ == "__main__":
    main()



