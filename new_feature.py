"""
新功能模块：实现新的捕捉区域和可视化预览
捕捉区域：(550,1760) 到 (3300,2000)
"""

import os
import time
import threading
import mss
import cv2
import numpy as np
import sys
import random
import pydirectinput as pdi
from typing import Optional, Tuple

# 设置pydirectinput参数
pdi.PAUSE = 0.0
pdi.FAILSAFE = False  # 禁用安全模式


class NewFeatureState:
    """新功能状态管理类"""
    def __init__(self):
        self.running = False
        self.preview_enabled = False  # 默认关闭可视化
        self.detection_threads = []  # 改为列表存储多个线程
        self.fps = 10
        self.template_cache = {}
        self.test_region = "S"  # 默认测试区域
        self.last_match_state = {}  # 存储每个区域的上一个匹配状态
        self.key_states = {}  # 存储每个区域的按键状态
        self.key_actions_enabled = True  # 按键操作开关


# 全局状态实例
new_feature_state = NewFeatureState()


def apply_brightness_correction(bgr_image: np.ndarray) -> np.ndarray:
    """应用亮度校正，复用主程序的逻辑"""
    if bgr_image is None or bgr_image.size == 0:
        return bgr_image
    
    img = bgr_image.astype(np.float32) / 255.0
    # 使用默认的亮度设置
    gain = 1.0
    gamma = 1.0
    img *= gain
    img = np.power(np.clip(img, 0.0, 1.0), gamma)
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def _load_play_templates() -> list:
    """加载play.png、play_hold.png和play_holding.png模板"""
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    template_files = ["play.png", "play_hold.png", "play_holding.png"]
    
    loaded = []
    for template_file in template_files:
        template_path = os.path.join(templates_dir, template_file)
        if os.path.exists(template_path):
            img = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
            if img is not None and img.size > 0:
                # 构造BGR模板和完整掩码
                if img.ndim == 3 and img.shape[2] == 4:
                    tmpl_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                elif img.ndim == 3:
                    tmpl_bgr = img.copy()
                else:
                    tmpl_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                h, w = tmpl_bgr.shape[:2]
                mask = np.full((h, w), 255, dtype=np.uint8)
                
                loaded.append((template_file, tmpl_bgr, mask))
    return loaded


def _build_play_template_cache(loaded_templates: list) -> None:
    """构建play、play_hold和play_holding模板缓存"""
    for entry in loaded_templates:
        name, tmpl_bgr, tmpl_mask = entry
        if name.lower() not in ["play.png", "play_hold.png", "play_holding.png"]:
            continue
            
        # 使用默认参数
        scale = 1.0
        threshold = 0.8
        
        th, tw = tmpl_bgr.shape[:2]
        new_w = max(1, int(round(tw * max(scale, 1e-3))))
        new_h = max(1, int(round(th * max(scale, 1e-3))))
        
        resized_bgr = cv2.resize(tmpl_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(tmpl_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        tmpl_gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
        gray_preview = tmpl_gray.copy()
        if resized_mask is not None:
            gray_preview[resized_mask == 0] = 255
        
        new_feature_state.template_cache[name] = {
            "tmpl_bgr_resized": resized_bgr,
            "tmpl_gray": tmpl_gray,
            "mask": resized_mask,
            "preview": gray_preview,
            "scale": scale,
            "threshold": threshold,
        }


def _calculate_color_similarity(frame_roi: np.ndarray, template_bgr: np.ndarray) -> float:
    """计算颜色相似度"""
    if frame_roi.shape != template_bgr.shape:
        return 0.0
    
    # 计算BGR通道的平均差异
    diff = cv2.absdiff(frame_roi, template_bgr)
    mean_diff = np.mean(diff)
    
    # 转换为相似度分数 (0-1，1表示完全相似)
    similarity = max(0.0, 1.0 - (mean_diff / 255.0))
    return similarity


def _trigger_key_action_async(region_name: str, action_type: str) -> None:
    """异步触发按键操作"""
    if not new_feature_state.key_actions_enabled:
        return
    
    # 区域名称到按键的映射
    key_mapping = {
        'S': 's', 'D': 'd', 'F': 'f', 
        'J': 'j', 'K': 'k', 'L': 'l'
    }
    
    key = key_mapping.get(region_name)
    if not key:
        return
    
    def worker():
        try:
            if action_type == "press":
                # 按下按键
                # 使用press函数，随机延迟0.1-0.4秒
                duration = random.uniform(0.1, 0.4)
                pdi.press(key, duration=duration, _pause=False)
                
            elif action_type == "hold":
                # 按住按键（不抬起）
                pdi.keyDown(key, _pause=False)
                
            elif action_type == "release":
                # 抬起按键
                pdi.keyUp(key, _pause=False)
                
        except Exception as e:
            print(f"按键操作失败: {e}")
    
    threading.Thread(target=worker, daemon=True).start()


def _handle_key_actions(region_name: str, per_template: dict) -> None:
    """处理按键操作逻辑"""
    if not per_template:
        return
    
    # 获取当前按键状态
    current_key_state = new_feature_state.key_states.get(region_name, {
        "is_pressed": False,
        "last_action": None,
        "action_time": 0
    })
    
    current_time = time.time()
    
    # 检查每个模板的匹配状态
    for name, data in per_template.items():
        score = data.get("score", 0.0)
        threshold = data.get("threshold", 0.8)
        template_type = data.get("template_type", "")
        
        if score >= threshold:
            if template_type == "play":
                # Play图标：直接按下对应按键
                if current_key_state["last_action"] != "play" or (current_time - current_key_state["action_time"]) > 0.3:
                    if current_key_state["is_pressed"]:
                        # 如果之前有按键按下，先抬起
                        _trigger_key_action_async(region_name, "release")
                    
                    # 直接按下按键
                    _trigger_key_action_async(region_name, "press")
                    
                    # 更新状态
                    new_feature_state.key_states[region_name] = {
                        "is_pressed": False,
                        "last_action": "play",
                        "action_time": current_time
                    }
                    
            elif template_type == "play_hold":
                # Hold图标：按下按键但不抬起
                if not current_key_state["is_pressed"]:
                    _trigger_key_action_async(region_name, "hold")
                    new_feature_state.key_states[region_name] = {
                        "is_pressed": True,
                        "last_action": "hold",
                        "action_time": current_time
                    }
                    
            elif template_type == "play_holding":
                # Holding图标：保持按键按下状态
                if not current_key_state["is_pressed"]:
                    _trigger_key_action_async(region_name, "hold")
                    new_feature_state.key_states[region_name] = {
                        "is_pressed": True,
                        "last_action": "holding",
                        "action_time": current_time
                    }
                elif current_key_state["last_action"] != "holding":
                    # 如果状态从hold变为holding，更新状态但不重复打印
                    new_feature_state.key_states[region_name] = {
                        "is_pressed": True,
                        "last_action": "holding",
                        "action_time": current_time
                    }
        else:
            # 没有匹配到任何图标，如果之前有按键按下则抬起
            if current_key_state["is_pressed"] and current_key_state["last_action"] in ["hold", "holding"]:
                _trigger_key_action_async(region_name, "release")
                new_feature_state.key_states[region_name] = {
                    "is_pressed": False,
                    "last_action": "release",
                    "action_time": current_time
                }


def _match_play_template_on_frame(bgr_frame: np.ndarray, loaded_templates: list, region_name: str) -> tuple[dict, Optional[dict], dict]:
    """在帧上匹配play和play_hold模板（使用彩色匹配）"""
    if not loaded_templates:
        return {}, None, {}
    
    scaled_for_preview: dict[str, np.ndarray] = {}
    best_score: float = 0.0
    best_name: Optional[str] = None
    best_loc: Optional[Tuple[int, int]] = None
    best_scale: float = 1.0
    best_overlay: Optional[dict] = None
    per_template_best: dict[str, dict] = {}
    
    # 获取该区域的上一个匹配状态
    last_state = new_feature_state.last_match_state.get(region_name, {})
    
    for entry in loaded_templates:
        name, tmpl_bgr, tmpl_mask = entry
        if name.lower() not in ["play.png", "play_hold.png", "play_holding.png"]:
            continue
            
        # 确保缓存存在
        cache = new_feature_state.template_cache.get(name)
        if cache is None:
            _build_play_template_cache([entry])
            cache = new_feature_state.template_cache.get(name)
        if cache is None:
            continue
        
        work_tmpl_bgr = cache["tmpl_bgr_resized"]
        mask_resized = cache["mask"]
        local_threshold = float(cache.get("threshold", 0.8))
        
        h_t, w_t = work_tmpl_bgr.shape[:2]
        # 使用彩色模板作为预览
        scaled_for_preview[name] = work_tmpl_bgr.copy()
        
        if h_t > bgr_frame.shape[0] or w_t > bgr_frame.shape[1]:
            continue
        
        # 直接使用BGR彩色模板匹配
        res = cv2.matchTemplate(bgr_frame, work_tmpl_bgr, cv2.TM_CCOEFF_NORMED)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(res)
        
        # 判断当前匹配状态
        current_matches = maxVal >= local_threshold
        if "play.png" in name:
            template_type = "play"
        elif "play_hold.png" in name:
            template_type = "play_hold"
        elif "play_holding.png" in name:
            template_type = "play_holding"
        else:
            template_type = "unknown"
        
        # 检查holding状态：当前是hold且上一个状态也是hold
        is_holding = False
        if template_type == "play_hold" and current_matches:
            last_hold_state = last_state.get("play_hold", False)
            is_holding = last_hold_state
        
        # 更新当前最佳分数
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
                "is_holding": is_holding,
                "template_type": template_type,
            }
        
        # 跟踪每个模板的最佳匹配
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
                "is_holding": is_holding,
                "template_type": template_type,
            }
    
    # 更新该区域的匹配状态
    new_feature_state.last_match_state[region_name] = {
        "play": per_template_best.get("play.png", {}).get("score", 0.0) >= per_template_best.get("play.png", {}).get("threshold", 0.8),
        "play_hold": per_template_best.get("play_hold.png", {}).get("score", 0.0) >= per_template_best.get("play_hold.png", {}).get("threshold", 0.8),
        "play_holding": per_template_best.get("play_holding.png", {}).get("score", 0.0) >= per_template_best.get("play_holding.png", {}).get("threshold", 0.8),
    }
    
    # 仅暴露当前最佳模板数据用于叠加
    best_only = {best_name: per_template_best.get(best_name)} if best_name in per_template_best else {}
    return scaled_for_preview, best_overlay, best_only


def _compose_unified_preview(frame_bgr: np.ndarray, scaled: dict) -> np.ndarray:
    """组合帧和缩放模板预览为单一图像"""
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr
    tiles: list[np.ndarray] = []
    for name, tmpl in scaled.items():
        # 模板现在是彩色的，直接使用
        tiles.append(tmpl)
    
    # 如果没有模板，直接返回帧
    if not tiles:
        return frame_bgr
    
    # 在右侧创建简单的垂直堆叠模板，带填充
    max_w = max(t.shape[1] for t in tiles)
    total_h = sum(t.shape[0] for t in tiles)
    pad = 6
    total_h += pad * (len(tiles) - 1)
    # 为右列创建白色画布
    right = np.full((total_h, max_w, 3), 255, dtype=np.uint8)
    y = 0
    for t in tiles:
        h, w = t.shape[:2]
        right[y:y+h, 0:w] = t
        y += h + pad
    
    # 通过填充较短的一边来匹配高度
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
    # 添加小的白色分隔符
    sep = np.full((H, 4, 3), 255, dtype=np.uint8)
    combined = np.hstack([left_padded, sep, right_padded])
    return combined


def _overlay_per_template(frame_bgr: np.ndarray, per_template: dict) -> np.ndarray:
    """渲染每个模板的最佳匹配矩形和轻量模板混合"""
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
        # 非常轻的混合以保持FPS
        roi = out[y:y+h, x:x+w]
        alpha = (mask.astype(np.float32) / 255.0)[..., None] * 0.3
        roi[:] = (roi.astype(np.float32) * (1.0 - alpha) + tmpl.astype(np.float32) * alpha).astype(np.uint8)
    return out


def get_new_capture_regions() -> list:
    """获取六个分割的捕捉区域配置"""
    regions = []
    
    # 区域名称从左至右：S D F J K L
    region_names = ['S', 'D', 'F', 'J', 'K', 'L']
    
    # 第一个S区域：左上角(550,1440)，右下角(790,1790) - 上移200像素
    s_left = 550
    s_top = 1640 - 200  # 1440
    s_width = 790 - 550  # 240
    s_height = 1990 - 1640  # 350
    
    regions.append({
        "left": s_left,
        "top": s_top,
        "width": s_width,
        "height": s_height,
        "name": "S"
    })
    
    # 计算距离
    length = 960 - 790  # 170 (S到D的距离)
    mid = 2230 - 1620  # 610 (F到J的距离)
    
    # D区域：S右侧 + length距离
    d_left = 790 + length  # 960
    regions.append({
        "left": d_left,
        "top": s_top,
        "width": s_width,
        "height": s_height,
        "name": "D"
    })
    
    # F区域：D右侧 + length距离
    f_left = d_left + s_width + length  # 960 + 240 + 170 = 1370
    regions.append({
        "left": f_left,
        "top": s_top,
        "width": s_width,
        "height": s_height,
        "name": "F"
    })
    
    # J区域：F右侧 + mid距离
    j_left = f_left + s_width + mid  # 1370 + 240 + 610 = 2220
    regions.append({
        "left": j_left,
        "top": s_top,
        "width": s_width,
        "height": s_height,
        "name": "J"
    })
    
    # K区域：J右侧 + length距离
    k_left = j_left + s_width + length  # 2220 + 240 + 170 = 2630
    regions.append({
        "left": k_left,
        "top": s_top,
        "width": s_width,
        "height": s_height,
        "name": "K"
    })
    
    # L区域：K右侧 + length距离
    l_left = k_left + s_width + length  # 2630 + 240 + 170 = 3040
    regions.append({
        "left": l_left,
        "top": s_top,
        "width": s_width,
        "height": s_height,
        "name": "L"
    })
    
    return regions


def capture_single_region_loop(region: dict, region_index: int):
    """单个区域的捕捉和预览循环"""
    print(f"启动区域 {region['name']} (索引: {region_index})")
    
    with mss.mss() as sct:
        # 加载play和play_hold模板
        loaded_templates = _load_play_templates()
        if not loaded_templates:
            print(f"警告：区域 {region['name']} 未找到play.png、play_hold.png或play_holding.png模板文件")
        else:
            _build_play_template_cache(loaded_templates)
        
        # 计算FPS间隔
        interval = 1.0 / max(1, new_feature_state.fps)
        last_match_time = 0.0
        latest_scaled: dict = {}
        per_template: dict = {}
        
        while new_feature_state.running:
            try:
                # 捕捉屏幕区域
                frame_bgra = sct.grab(region)
                frame = np.array(frame_bgra)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                frame = apply_brightness_correction(frame)
                
                # 模板匹配
                loop_start = time.time()
                now = loop_start
                if loaded_templates and (now - last_match_time) >= interval:
                    last_match_time = now
                    latest_scaled, best_overlay, per_template = _match_play_template_on_frame(frame, loaded_templates, region['name'])
                    
                    # 处理按键操作
                    _handle_key_actions(region['name'], per_template)
                
                # 显示预览窗口
                if new_feature_state.preview_enabled:
                    # 添加匹配结果叠加
                    frame_with_overlay = _overlay_per_template(frame, per_template)
                    preview_img = _compose_unified_preview(frame_with_overlay, latest_scaled)
                    
                    # 添加信息文本
                    cv2.putText(preview_img, f"Region {region['name']}: {region['width']}x{region['height']}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(preview_img, f"FPS: {new_feature_state.fps}", 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(preview_img, "Press 'q' to quit", 
                               (10, preview_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
                    # 显示匹配分数
                    if per_template:
                        y_offset = 70
                        for name, data in per_template.items():
                            score = data.get("score", 0.0)
                            threshold = data.get("threshold", 0.8)
                            is_holding = data.get("is_holding", False)
                            template_type = data.get("template_type", "")
                            
                            # 确定显示状态
                            if score >= threshold:
                                if is_holding:
                                    status = "HOLDING"
                                    color = (0, 255, 255)  # 黄色表示holding
                                else:
                                    status = "MATCH"
                                    color = (0, 0, 255)  # 红色表示match
                            else:
                                status = "NO MATCH"
                                color = (0, 255, 0)  # 绿色表示no match
                            
                            # 显示模板名称和状态
                            if template_type == "play":
                                template_name = "Play"
                            elif template_type == "play_hold":
                                template_name = "PlayHold"
                            elif template_type == "play_holding":
                                template_name = "PlayHolding"
                            else:
                                template_name = "Unknown"
                            
                            cv2.putText(preview_img, f"{template_name}: {status} ({score:.3f})", 
                                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            y_offset += 20
                    
                    window_name = f"Region {region['name']} Preview"
                    cv2.imshow(window_name, preview_img)
                    
                    # 设置窗口置顶
                    try:
                        import win32gui
                        import win32con
                        hwnd = win32gui.FindWindow(None, window_name)
                        if hwnd:
                            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, 
                                                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
                    except ImportError:
                        pass  # 如果没有win32gui，忽略置顶功能
                    
                    # 检查退出键
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print(f"用户按下 'q' 键，退出区域 {region['name']}")
                        break
                
                # 控制帧率
                sleep_s = interval - (time.time() - loop_start)
                if sleep_s > 0:
                    time.sleep(sleep_s)
                
            except Exception as e:
                print(f"区域 {region['name']} 捕捉过程中发生错误: {e}")
                time.sleep(0.1)
    
    # 清理资源
    if new_feature_state.preview_enabled:
        cv2.destroyWindow(f"Region {region['name']} Preview")
    print(f"区域 {region['name']} 已停止")


def capture_and_preview_loop():
    """主循环：启动所有区域的捕捉"""
    print("新功能启动：监控所有六个区域 (S D F J K L)")
    print("模板匹配：play.png、play_hold.png 和 play_holding.png")
    
    # 获取六个区域
    regions = get_new_capture_regions()
    print(f"总共 {len(regions)} 个区域:")
    for i, region in enumerate(regions):
        print(f"  {i+1}. {region['name']}: ({region['left']}, {region['top']}) {region['width']}x{region['height']}")
    
    # 启动每个区域的线程
    region_threads = []
    for i, region in enumerate(regions):
        thread = threading.Thread(
            target=capture_single_region_loop, 
            args=(region, i), 
            daemon=True,
            name=f"Region-{region['name']}"
        )
        thread.start()
        region_threads.append(thread)
        print(f"已启动区域 {region['name']} 的线程")
    
    # 等待所有区域线程完成
    for thread in region_threads:
        thread.join()
    
    print("所有区域已停止")


def start_new_feature():
    """启动新功能"""
    if new_feature_state.running:
        print("新功能已在运行中")
        return
    
    new_feature_state.running = True
    new_feature_state.detection_threads.clear()  # 清空线程列表
    
    # 启动主线程
    main_thread = threading.Thread(target=capture_and_preview_loop, daemon=True)
    main_thread.start()
    new_feature_state.detection_threads.append(main_thread)
    print("新功能已启动（监控所有六个区域）")


def stop_new_feature():
    """停止新功能"""
    if not new_feature_state.running:
        print("新功能未在运行")
        return
    
    new_feature_state.running = False
    
    # 等待所有线程完成
    for thread in new_feature_state.detection_threads:
        if thread.is_alive():
            thread.join(timeout=2.0)
    
    new_feature_state.detection_threads.clear()
    print("新功能已停止")


def toggle_preview():
    """切换预览窗口"""
    new_feature_state.preview_enabled = not new_feature_state.preview_enabled
    print(f"预览窗口: {'开启' if new_feature_state.preview_enabled else '关闭'}")


def set_fps(fps: int):
    """设置FPS"""
    new_feature_state.fps = max(1, min(60, fps))
    print(f"FPS设置为: {new_feature_state.fps}")


def toggle_key_actions():
    """切换按键操作开关"""
    new_feature_state.key_actions_enabled = not new_feature_state.key_actions_enabled
    print(f"按键操作: {'开启' if new_feature_state.key_actions_enabled else '关闭'}")


def release_all_keys():
    """释放所有按键"""
    key_mapping = {
        'S': 's', 'D': 'd', 'F': 'f', 
        'J': 'j', 'K': 'k', 'L': 'l'
    }
    
    for region_name, key in key_mapping.items():
        try:
            pdi.keyUp(key, _pause=False)
        except Exception as e:
            print(f"释放按键失败: {e}")
    
    # 重置所有区域的按键状态
    for region_name in key_mapping.keys():
        new_feature_state.key_states[region_name] = {
            "is_pressed": False,
            "last_action": None,
            "action_time": 0
        }


def main():
    """主函数 - 独立运行测试"""
    # 解析命令行参数
    test_region = "S"  # 默认测试区域
    if len(sys.argv) > 1:
        test_region = sys.argv[1].upper()
        if test_region not in ['S', 'D', 'F', 'J', 'K', 'L']:
            print(f"错误：无效的区域名称 '{test_region}'")
            print("有效的区域名称：S, D, F, J, K, L")
            return
        new_feature_state.test_region = test_region
    
    print("=" * 50)
    print("新功能独立测试模式")
    print("=" * 50)
    print("功能说明：")
    print("- 捕捉区域：S(550,1440) 到 L(3280,1790)")
    print("- 区域大小：240x350 像素")
    print("- 在新线程中运行，不影响其他功能")
    print("- 默认关闭可视化（可通过命令开启）")
    print("- 监控所有六个区域：S D F J K L")
    print("- 自动按键操作：s d f j k l")
    print("- 按键状态实时显示")
    print("=" * 50)
    print("使用方法：")
    print("python new_feature.py")
    print("（现在会监控所有区域，不再需要指定区域名称）")
    print("=" * 50)
    
    try:
        # 启动新功能
        start_new_feature()
        
        # 等待用户输入
        print("\n控制命令：")
        print("- 输入 'p' 切换预览窗口")
        print("- 输入 'f' 设置FPS")
        print("- 输入 'k' 切换按键操作")
        print("- 输入 'r' 释放所有按键")
        print("- 输入 's' 停止功能")
        print("- 输入 'q' 退出程序")
        print("- 在预览窗口中按 'q' 也可以退出")
        
        while new_feature_state.running:
            try:
                command = input("\n请输入命令: ").strip().lower()
                
                if command == 'p':
                    toggle_preview()
                elif command == 'f':
                    try:
                        fps_input = input("请输入FPS (1-60): ").strip()
                        fps = int(fps_input)
                        set_fps(fps)
                    except ValueError:
                        print("无效的FPS值")
                elif command == 'k':
                    toggle_key_actions()
                elif command == 'r':
                    release_all_keys()
                elif command == 's':
                    stop_new_feature()
                    break
                elif command == 'q':
                    stop_new_feature()
                    break
                else:
                    print("未知命令，请重新输入")
                    
            except KeyboardInterrupt:
                print("\n检测到 Ctrl+C，正在退出...")
                break
            except EOFError:
                print("\n检测到输入结束，正在退出...")
                break
                
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        # 确保清理资源
        if new_feature_state.running:
            stop_new_feature()
        # 释放所有按键
        release_all_keys()
        print("程序已退出")


if __name__ == "__main__":
    main()
