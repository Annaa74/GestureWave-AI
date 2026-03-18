"""
GestureWave AI — Enhanced Gesture Engine v2.0
=============================================
Features:
  • Exponential Moving Average (EMA)  cursor smoothing
  • Velocity-adaptive dampening       (fast = smooth, slow = precise)
  • Dead zone                         (eliminates tremor / micro-jitter)
  • Full gesture state machine        (debounced, cooldown-aware)
  • Gestures:
      Index finger     → Move cursor
      Index+Thumb pinch→ Left click
      Middle+Thumb pinch→Right click
      Double pinch     → Double click
      Hold pinch+move  → Drag & drop
      Two-finger spread→ Zoom in/out  (Ctrl +/-)
      Peace sign up    → Scroll up
      Peace sign down  → Scroll down
      Open palm        → Pause / resume tracking
  • On-screen HUD showing gesture name, FPS, smoothing level
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque
from enum import Enum, auto
from gesture_utils import normalize_landmarks
from gesture_registry import GestureRegistry

# ── Safety ──────────────────────────────────────────────────────────
pyautogui.FAILSAFE = True   # move mouse to corner to emergency-stop
pyautogui.PAUSE   = 0.0

# ── Config ───────────────────────────────────────────────────────────
class Cfg:
    CAMERA_ID           = 0
    FLIP_H              = True

    # Smoothing — EMA alpha (0 = max smooth/laggy, 1 = no smoothing)
    SMOOTH_ALPHA        = 0.25
    # Velocity boost: if cursor jumps far, temporarily raise alpha
    VEL_BOOST_THRESHOLD = 60    # pixels/frame before boost kicks in
    VEL_BOOST_ALPHA     = 0.55

    # Dead zone: ignore movement smaller than N screen-pixels
    DEAD_ZONE           = 4

    # Gesture distance thresholds (in webcam pixels)
    CLICK_THRESH        = 32
    PINCH_RELEASE       = 55    # must open wider than this to reset

    # Drag: hold a pinch for this many frames to enter drag mode
    DRAG_HOLD_FRAMES    = 12

    # Double click: two clicks within this many seconds
    DBL_CLICK_WINDOW    = 0.38

    # Cooldowns (seconds between repeated triggers)
    CLICK_COOLDOWN      = 0.28
    SCROLL_COOLDOWN     = 0.08
    ZOOM_COOLDOWN       = 0.12
    PAUSE_COOLDOWN      = 1.20

    # Open-palm pause: all 5 finger tips must be above their MCP joints
    PALM_OPEN_FINGERS   = 4     # out of 5

    # Scroll sensitivity
    SCROLL_AMOUNT       = 15

    # Zoom (Ctrl +/-) steps
    ZOOM_STEP           = 1


# ── Gesture states ───────────────────────────────────────────────────
class GState(Enum):
    IDLE        = auto()
    MOVING      = auto()
    PINCHING    = auto()   # left-click in progress
    DRAGGING    = auto()
    SCROLLING   = auto()
    PAUSED      = auto()
    RECORDING   = auto()


# ── Colour palette (BGR) ─────────────────────────────────────────────
C = {
    "blue":   (255, 120,  40),
    "green":  ( 50, 220,  80),
    "yellow": ( 40, 230, 230),
    "red":    ( 60,  50, 230),
    "orange": ( 30, 160, 255),
    "white":  (255, 255, 255),
    "black":  (  0,   0,   0),
    "grey":   (140, 140, 140),
    "hud_bg": ( 20,  20,  20),
}


# ── Helper: Euclidean distance ────────────────────────────────────────
def dist(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ── EMA smoother ─────────────────────────────────────────────────────
class EMASmoother:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self._x: float | None = None
        self._y: float | None = None

    def update(self, x: float, y: float, vel: float = 0) -> tuple[float, float]:
        # Velocity-adaptive alpha
        a = Cfg.VEL_BOOST_ALPHA if vel > Cfg.VEL_BOOST_THRESHOLD else self.alpha
        if self._x is None:
            self._x, self._y = x, y
        else:
            self._x = a * x + (1 - a) * self._x
            self._y = a * y + (1 - a) * self._y
        return self._x, self._y

    def reset(self):
        self._x = self._y = None


# ── FPS tracker ──────────────────────────────────────────────────────
class FPSCounter:
    def __init__(self, window=30):
        self._times: deque = deque(maxlen=window)

    def tick(self):
        self._times.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


# ── Open-palm detector ───────────────────────────────────────────────
def is_open_palm(lm, fw, fh) -> bool:
    """Return True when 4+ fingers are extended (tips above their PIP)."""
    tips   = [8, 12, 16, 20]
    pips   = [6, 10, 14, 18]
    count  = sum(1 for t, p in zip(tips, pips)
                 if lm[t].y < lm[p].y)   # y is inverted in image coords
    return count >= Cfg.PALM_OPEN_FINGERS


# ── Peace-sign detector ──────────────────────────────────────────────
def is_peace_sign(lm) -> bool:
    """Index & middle extended, ring & pinky curled, thumb relaxed."""
    index_up  = lm[8].y  < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_down = lm[16].y > lm[14].y
    pinky_down= lm[20].y > lm[18].y
    return index_up and middle_up and ring_down and pinky_down


# ── Two-finger spread (zoom) ─────────────────────────────────────────
class ZoomTracker:
    def __init__(self):
        self._prev_dist: float | None = None

    def update(self, d: float) -> int:
        """Returns +1 (zoom in), -1 (zoom out), or 0 (no action)."""
        if self._prev_dist is None:
            self._prev_dist = d
            return 0
        delta = d - self._prev_dist
        self._prev_dist = d
        if delta > 6:
            return 1
        if delta < -6:
            return -1
        return 0

    def reset(self):
        self._prev_dist = None


# ── On-screen HUD ────────────────────────────────────────────────────
def draw_hud(frame, state: GState, gesture_name: str, fps: float, smoothed_xy):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Top-left info bar
    cv2.rectangle(overlay, (0, 0), (w, 40), C["hud_bg"], -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    state_color = {
        GState.IDLE:      C["grey"],
        GState.MOVING:    C["blue"],
        GState.PINCHING:  C["yellow"],
        GState.DRAGGING:  C["orange"],
        GState.SCROLLING: C["green"],
        GState.PAUSED:    C["red"],
    }.get(state, C["white"])

    cv2.putText(frame, f"GestureWave AI v2", (10, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, C["white"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:4.1f}", (220, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C["grey"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"{gesture_name}", (330, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2, cv2.LINE_AA)

    # State dot
    dot_color = state_color
    cv2.circle(frame, (w - 20, 20), 7, dot_color, -1)

    # Smoothed cursor crosshair overlay (tiny)
    if smoothed_xy and state != GState.PAUSED:
        sx, sy = int(smoothed_xy[0] * w), int(smoothed_xy[1] * h)
        cv2.drawMarker(frame, (sx, sy), C["blue"], cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)

    # If recording, show countdown
    if state == GState.RECORDING:
        cv2.putText(frame, "RECORDING IN 3s...", (w//2 - 100, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, C["red"], 3, cv2.LINE_AA)


# ── Main loop ────────────────────────────────────────────────────────
def run():
    cap = cv2.VideoCapture(Cfg.CAMERA_ID)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check your webcam connection.")
        return

    # Prefer higher resolution if the cam supports it
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          30)

    hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.72,
        min_tracking_confidence=0.60,
    )
    draw_utils = mp.solutions.drawing_utils
    draw_style  = mp.solutions.drawing_styles

    screen_w, screen_h = pyautogui.size()

    smoother     = EMASmoother(Cfg.SMOOTH_ALPHA)
    fps_counter  = FPSCounter()
    zoom_tracker = ZoomTracker()

    # State
    state: GState    = GState.IDLE
    gesture_name     = "Idle"
    prev_sx: float   = 0.0
    prev_sy: float   = 0.0

    # Timers & counters
    last_click_time    = 0.0
    last_scroll_time   = 0.0
    last_zoom_time     = 0.0
    last_pause_time    = 0.0
    last_click_ts      = 0.0   # for double-click detection
    pinch_hold_frames  = 0
    drag_active        = False
    
    registry = GestureRegistry()
    recording_start_time = 0.0

    print("[GestureWave AI v2.0] — Starting. Press ESC in the camera window to exit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if Cfg.FLIP_H:
            frame = cv2.flip(frame, 1)

        fh, fw = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        fps_counter.tick()

        smoothed_norm = None   # normalised (0-1) cursor position for HUD

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            # Draw landmarks with styled connections
            draw_utils.draw_landmarks(
                frame, hand,
                mp.solutions.hands.HAND_CONNECTIONS,
                draw_style.get_default_hand_landmarks_style(),
                draw_style.get_default_hand_connections_style(),
            )

            lm = hand.landmark

            # ── Raw landmark positions (px) ───────────────────────
            def px(idx):
                return int(lm[idx].x * fw), int(lm[idx].y * fh)

            thumb_pt  = px(4)
            index_pt  = px(8)
            middle_pt = px(12)
            ring_pt   = px(16)
            pinky_pt  = px(20)

            # ── Distances ─────────────────────────────────────────
            d_left   = dist(thumb_pt, index_pt)
            d_right  = dist(thumb_pt, middle_pt)
            d_scroll = dist(index_pt, middle_pt)
            d_zoom   = dist(index_pt, middle_pt)  # also used for 2-finger zoom

            # ── Raw → smoothed screen coords ──────────────────────
            raw_sx = (lm[8].x) * screen_w
            raw_sy = (lm[8].y) * screen_h

            vel = dist((raw_sx, raw_sy), (prev_sx, prev_sy))
            sm_sx, sm_sy = smoother.update(raw_sx, raw_sy, vel)

            # Dead zone suppression
            if dist((sm_sx, sm_sy), (prev_sx, prev_sy)) > Cfg.DEAD_ZONE:
                prev_sx, prev_sy = sm_sx, sm_sy

            smoothed_norm = (lm[8].x, lm[8].y)   # for HUD overlay

            now = time.perf_counter()

            # ── Open palm → pause / resume ────────────────────────
            if is_open_palm(lm, fw, fh):
                if now - last_pause_time > Cfg.PAUSE_COOLDOWN:
                    last_pause_time = now
                    if state == GState.PAUSED:
                        state        = GState.IDLE
                        gesture_name = "Resumed"
                    else:
                        if drag_active:
                            pyautogui.mouseUp()
                            drag_active = False
                        state        = GState.PAUSED
                        gesture_name = "Paused ✋"

            if state == GState.PAUSED:
                draw_hud(frame, state, gesture_name, fps_counter.fps, smoothed_norm)
                cv2.imshow("GestureWave AI", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue   # skip all gesture processing while paused

            # ── Peace sign → scroll ───────────────────────────────
            if is_peace_sign(lm) and d_left > Cfg.PINCH_RELEASE:
                if now - last_scroll_time > Cfg.SCROLL_COOLDOWN:
                    last_scroll_time = now
                    state = GState.SCROLLING
                    if lm[8].y < 0.45:
                        pyautogui.scroll(Cfg.SCROLL_AMOUNT)
                        gesture_name = "Scroll Up ✌"
                    else:
                        pyautogui.scroll(-Cfg.SCROLL_AMOUNT)
                        gesture_name = "Scroll Down ✌"
                    cv2.circle(frame, index_pt,  16, C["green"], cv2.FILLED)
                    cv2.circle(frame, middle_pt, 16, C["green"], cv2.FILLED)

            # ── Two-finger spread → zoom ──────────────────────────
            elif (lm[8].y < lm[6].y and lm[12].y < lm[10].y
                    and d_zoom > 35 and d_left > Cfg.CLICK_THRESH * 1.5):
                zoom_dir = zoom_tracker.update(d_zoom)
                if zoom_dir != 0 and now - last_zoom_time > Cfg.ZOOM_COOLDOWN:
                    last_zoom_time = now
                    if zoom_dir > 0:
                        pyautogui.hotkey("ctrl", "+")
                        gesture_name = "Zoom In 🔍"
                    else:
                        pyautogui.hotkey("ctrl", "-")
                        gesture_name = "Zoom Out 🔍"
                cv2.line(frame, index_pt, middle_pt, C["orange"], 2)

            # ── Right click ───────────────────────────────────────
            elif d_right < Cfg.CLICK_THRESH and d_left > Cfg.CLICK_THRESH:
                if now - last_click_time > Cfg.CLICK_COOLDOWN:
                    last_click_time = now
                    pyautogui.rightClick(prev_sx, prev_sy)
                    gesture_name = "Right Click 🤌"
                    state = GState.PINCHING
                cv2.circle(frame, middle_pt, 18, C["red"], cv2.FILLED)
                cv2.circle(frame, thumb_pt,  12, C["red"], cv2.FILLED)

            # ── Left click / drag ─────────────────────────────────
            elif d_left < Cfg.CLICK_THRESH:
                pinch_hold_frames += 1

                # Drag mode
                if pinch_hold_frames >= Cfg.DRAG_HOLD_FRAMES and not drag_active:
                    drag_active  = True
                    state        = GState.DRAGGING
                    gesture_name = "Dragging 🤏→"
                    pyautogui.mouseDown()

                if drag_active:
                    pyautogui.moveTo(prev_sx, prev_sy)
                    cv2.circle(frame, index_pt, 18, C["orange"], cv2.FILLED)
                    cv2.circle(frame, thumb_pt, 14, C["orange"], cv2.FILLED)
                else:
                    # Single / double click on release
                    state = GState.PINCHING
                    cv2.circle(frame, index_pt, 18, C["yellow"], cv2.FILLED)
                    cv2.circle(frame, thumb_pt, 14, C["yellow"], cv2.FILLED)
                    gesture_name = "Pinching…"

            # ── Pinch released ────────────────────────────────────
            else:
                if drag_active:
                    pyautogui.mouseUp()
                    drag_active   = False
                    gesture_name  = "Drop"
                    state         = GState.IDLE

                if state == GState.PINCHING and pinch_hold_frames < Cfg.DRAG_HOLD_FRAMES:
                    if now - last_click_time > Cfg.CLICK_COOLDOWN:
                        last_click_time = now
                        # Double-click detection
                        if now - last_click_ts < Cfg.DBL_CLICK_WINDOW:
                            pyautogui.doubleClick(prev_sx, prev_sy)
                            gesture_name = "Double Click ⚡"
                            last_click_ts = 0  # reset
                        else:
                            pyautogui.click(prev_sx, prev_sy)
                            gesture_name  = "Click 👆"
                            last_click_ts = now

                pinch_hold_frames = 0
                zoom_tracker.reset()

                # ── Normal cursor movement ─────────────────────────
                if d_left > Cfg.PINCH_RELEASE and d_right > Cfg.PINCH_RELEASE:
                    state = GState.MOVING
                    gesture_name = "Moving ☝"
                    pyautogui.moveTo(prev_sx, prev_sy, _pause=False)
                    cv2.circle(frame, index_pt, 10, C["blue"], cv2.FILLED)

                # ── Custom Gestures ────────────────────────────────
                normalized = normalize_landmarks(lm)
                custom_match, score = registry.recognize(normalized)
                if custom_match and state == GState.MOVING:
                    gesture_name = f"Custom: {custom_match}"
                    # You could map specific names to specific actions here
                    if custom_match == "Wave":
                        pass # Trigger wave action

        else:
            # No hand detected — reset
            smoother.reset()
            zoom_tracker.reset()
            if drag_active:
                pyautogui.mouseUp()
                drag_active = False
            state        = GState.IDLE
            gesture_name = "No Hand"
            pinch_hold_frames = 0

        # ── Draw HUD ──────────────────────────────────────────────
        draw_hud(frame, state, gesture_name, fps_counter.fps, smoothed_norm)

        cv2.imshow("GestureWave AI", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            break
        elif key == ord('r'): # Start recording
            state = GState.RECORDING
            recording_start_time = time.perf_counter()
            print("[Gesture] Recording started...")

        if state == GState.RECORDING:
            elapsed = time.perf_counter() - recording_start_time
            if elapsed > 3.0:
                if result.multi_hand_landmarks:
                    lm = result.multi_hand_landmarks[0].landmark
                    normalized = normalize_landmarks(lm)
                    g_name = f"Gesture_{len(registry.gestures) + 1}"
                    registry.add_gesture(g_name, normalized)
                    print(f"[Gesture] Saved as {g_name}")
                    gesture_name = f"Saved {g_name}!"
                state = GState.IDLE

    if drag_active:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()
    print("[GestureWave AI] Exited cleanly.")


if __name__ == "__main__":
    run()