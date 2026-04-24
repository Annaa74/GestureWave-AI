import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import threading
from collections import deque
from enum import Enum, auto

from core.config import Cfg
from core.gestures import classify_gesture, GestureType, GestureConfig
from core.actions import ActionExecutor

stop_flag = False


class GState(Enum):
    IDLE = auto()
    MOVING = auto()
    PINCHING = auto()
    SCROLLING = auto()
    ZOOMING = auto()
    PAUSED = auto()


C = {
    "blue":   (255, 120, 40),
    "green":  (50, 220, 80),
    "yellow": (40, 230, 230),
    "red":    (60, 50, 230),
    "orange": (30, 160, 255),
    "white":  (255, 255, 255),
    "grey":   (140, 140, 140),
    "hud_bg": (20, 20, 20),
}


def dist(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


class EMASmoother:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self._x = None
        self._y = None

    def update(self, x: float, y: float, vel: float = 0.0):
        a = Cfg.VEL_BOOST_ALPHA if vel > Cfg.VEL_BOOST_THRESHOLD else self.alpha
        if self._x is None:
            self._x, self._y = x, y
        else:
            self._x = a * x + (1 - a) * self._x
            self._y = a * y + (1 - a) * self._y
        return self._x, self._y

    def reset(self):
        self._x = None
        self._y = None


class FPSCounter:
    def __init__(self, window=30):
        self._times = deque(maxlen=window)

    def tick(self):
        self._times.append(time.perf_counter())

    @property
    def fps(self):
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._times) - 1) / elapsed


class TrainingPlayground:
    def __init__(self):
        self.step = 0
        self.success_time = 0
        
        # Interactive Elements
        self.btn = [465, 275, 120, 50]
        self.scroll_y = 0
        self.zoom_r = 30

    def in_rect(self, x, y, rect):
        rx, ry, rw, rh = rect
        return rx <= x <= rx+rw and ry <= y <= ry+rh

    def update(self, cx, cy, state, action_name):
        now = time.perf_counter()
        if self.success_time > 0:
            if now - self.success_time > 1.5:
                self.success_time = 0
                self.step += 1
            return

        if cx is None or cy is None: return

        if self.step == 0:
            if dist((cx, cy), (525, 300)) < 40:
                self.success_time = now
        elif self.step == 1:
            if action_name == "Click 👆" and self.in_rect(cx, cy, self.btn):
                self.success_time = now
        elif self.step == 2:
            if action_name == "Right Click 🤟" and self.in_rect(cx, cy, self.btn):
                self.success_time = now
        elif self.step == 3:
            if action_name == "Double Click ⚡" and self.in_rect(cx, cy, self.btn):
                self.success_time = now
        elif self.step == 4:
            if action_name == "Scroll Down ✌" and self.in_rect(cx, cy, [485, 180, 80, 200]):
                self.scroll_y = min(150, self.scroll_y + 15)
                if self.scroll_y >= 140:
                    self.success_time = now
        elif self.step == 5:
            if action_name == "Zoom In 👍" and dist((cx, cy), (525, 300)) < self.zoom_r * 2.5:
                self.zoom_r = min(80, self.zoom_r + 5)
                if self.zoom_r >= 75:
                    self.success_time = now

    def draw(self, frame):
        steps_info = [
            ("Move Practice", "Index Up", "Move cursor to the blue circle", "Move Cursor"),
            ("Left Click", "Index Pinch", "Left click the green button", "Left Click"),
            ("Right Click", "Three Fingers Up", "Right click the red button", "Right Click"),
            ("Double Click", "2x Index Pinch", "Double click the yellow button", "Double Click"),
            ("Scroll Up/Down", "Peace Sign + Move", "Scroll down inside the grey panel", "Scroll"),
            ("Zoom In/Out", "Thumbs Up / Down", "Zoom in (Thumbs Up) to grow the circle", "Zoom"),
            ("Training Complete!", "Open Palm", "You've mastered GestureWave AI!", "Pause / Exit"),
        ]
        
        idx = min(self.step, len(steps_info)-1)
        title, gest, expect, action = steps_info[idx]
        
        cv2.rectangle(frame, (350, 40), (710, 170), (20,20,20), -1)
        if idx == len(steps_info) - 1:
            cv2.putText(frame, f"{title}", (360, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C["green"], 2)
            cv2.putText(frame, "You can now exit training mode.", (360, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C["white"], 1)
        else:
            cv2.putText(frame, f"Step {idx+1}/{len(steps_info)-1}: {title}", (360, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C["white"], 2)
            cv2.putText(frame, f"Gesture: {gest}", (360, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C["yellow"], 1)
            cv2.putText(frame, f"Goal:    {expect}", (360, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C["grey"], 1)
            cv2.putText(frame, f"Action:  {action}", (360, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C["blue"], 1)
        
        if self.success_time > 0:
            cv2.putText(frame, "SUCCESS!", (450, 300), cv2.FONT_HERSHEY_DUPLEX, 1.2, C["green"], 2)
            return
            
        if self.step == 0:
            cv2.circle(frame, (525, 300), 40, C["blue"], 2)
            cv2.circle(frame, (525, 300), 4, C["blue"], -1)
        elif self.step == 1:
            cv2.rectangle(frame, (self.btn[0], self.btn[1]), (self.btn[0]+self.btn[2], self.btn[1]+self.btn[3]), C["green"], -1)
            cv2.putText(frame, "Click", (self.btn[0]+38, self.btn[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C["white"], 1)
        elif self.step == 2:
            cv2.rectangle(frame, (self.btn[0], self.btn[1]), (self.btn[0]+self.btn[2], self.btn[1]+self.btn[3]), C["red"], -1)
            cv2.putText(frame, "R-Click", (self.btn[0]+28, self.btn[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C["white"], 1)
        elif self.step == 3:
            cv2.rectangle(frame, (self.btn[0], self.btn[1]), (self.btn[0]+self.btn[2], self.btn[1]+self.btn[3]), C["yellow"], -1)
            cv2.putText(frame, "D-Click", (self.btn[0]+28, self.btn[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        elif self.step == 4:
            panel = [485, 180, 80, 200]
            cv2.rectangle(frame, (panel[0], panel[1]), (panel[0]+panel[2], panel[1]+panel[3]), (50,50,50), -1)
            sy = panel[1] + 20 + self.scroll_y
            cv2.circle(frame, (panel[0]+40, sy), 15, C["grey"], -1)
            cv2.putText(frame, "Scroll", (panel[0]+20, panel[1]+190), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["white"], 1)
        elif self.step == 5:
            cv2.circle(frame, (525, 300), self.zoom_r, C["orange"], -1)
            cv2.putText(frame, "Zoom", (505, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C["white"], 1)


def draw_hud(frame, state: GState, gesture_name: str, fps: float, smoothed_xy, training_mode: bool = False, gesture_result=None, executor=None, last_left_click_release=0.0, playground=None):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), C["hud_bg"], -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    state_color = {
        GState.IDLE: C["grey"],
        GState.MOVING: C["blue"],
        GState.PINCHING: C["yellow"],
        GState.SCROLLING: C["green"],
        GState.ZOOMING: C["orange"],
        GState.PAUSED: C["red"],
    }.get(state, C["white"])

    cv2.putText(
        frame,
        "GestureWave AI Expanded Core",
        (10, 26),
        cv2.FONT_HERSHEY_DUPLEX,
        0.58,
        C["white"],
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:4.1f}",
        (320, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        C["grey"],
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        gesture_name,
        (430, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        state_color,
        2,
        cv2.LINE_AA,
    )

    cv2.circle(frame, (w - 20, 20), 7, state_color, -1)

    if smoothed_xy and state != GState.PAUSED:
        sx, sy = int(smoothed_xy[0] * w), int(smoothed_xy[1] * h)
        cv2.drawMarker(frame, (sx, sy), C["blue"], cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)



def run():
    global stop_flag
    stop_flag = False

    cap = cv2.VideoCapture(Cfg.CAMERA_ID)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Cfg.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Cfg.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, Cfg.CAMERA_FPS)

    hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.70,
        min_tracking_confidence=0.60,
    )
    draw_utils = mp.solutions.drawing_utils
    draw_style = mp.solutions.drawing_styles

    screen_w, screen_h = pyautogui.size()
    executor = ActionExecutor(
        safe_top_bar=Cfg.SAFE_TOP_BAR,
        enable_click_actions=Cfg.ENABLE_CLICK_ACTIONS
    )

    smoother = EMASmoother(Cfg.SMOOTH_ALPHA)
    fps_counter = FPSCounter()

    state = GState.IDLE
    gesture_name = "Idle"

    prev_sx = 0.0
    prev_sy = 0.0

    last_click_time = 0.0
    last_pause_time = 0.0
    last_scroll_time = 0.0
    last_zoom_time = 0.0
    last_right_click_time = 0.0
    last_left_click_release = 0.0

    freeze_start = 0.0
    freeze_duration = 0.0

    # Stability tracking for advanced gestures
    last_gesture_type = None
    gesture_stable_frames = 0

    g_config = GestureConfig(
        click_thresh=Cfg.CLICK_THRESH,
        right_click_thresh=Cfg.RIGHT_CLICK_THRESH,
        pinch_release=Cfg.PINCH_RELEASE,
        palm_open_fingers=Cfg.PALM_OPEN_FINGERS,
        scroll_gap_max=Cfg.SCROLL_GAP_MAX,
    )

    print("[GestureWave AI Expanded Core] Starting. Press ESC to exit.")

    def handle_advanced(last_time, cooldown, action_func, action_args, new_state, active_name, ready_name):
        if gesture_stable_frames >= Cfg.ADVANCED_GESTURE_STABLE_FRAMES and (now - last_time) > cooldown:
            threading.Thread(target=action_func, args=action_args, daemon=True).start()
            return now, new_state, active_name
        return last_time, GState.IDLE, ready_name
        
    while not stop_flag:
        ok, frame = cap.read()
        if not ok:
            break

        if Cfg.FLIP_H:
            frame = cv2.flip(frame, 1)

        fh, fw = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        fps_counter.tick()

        smoothed_norm = None

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            draw_utils.draw_landmarks(
                frame,
                hand,
                mp.solutions.hands.HAND_CONNECTIONS,
                draw_style.get_default_hand_landmarks_style(),
                draw_style.get_default_hand_connections_style(),
            )

            lm = hand.landmark

            raw_sx = lm[8].x * screen_w
            raw_sy = max(lm[8].y * screen_h, Cfg.SAFE_TOP_BAR)

            vel = dist((raw_sx, raw_sy), (prev_sx, prev_sy))
            sm_sx, sm_sy = smoother.update(raw_sx, raw_sy, vel)

            if dist((sm_sx, sm_sy), (prev_sx, prev_sy)) > Cfg.DEAD_ZONE:
                prev_sx, prev_sy = sm_sx, sm_sy

            smoothed_norm = (lm[8].x, lm[8].y)
            now = time.perf_counter()

            if now - freeze_start < freeze_duration:
                remaining = freeze_duration - (now - freeze_start)
                gesture_name = f"Frozen ({remaining:.1f}s)"
                state = GState.PAUSED
                draw_hud(frame, state, gesture_name, fps_counter.fps, smoothed_norm)
                
                # Draw cooldown bar at HUD (y=40 to 44)
                bar_w = int((remaining / freeze_duration) * fw)
                cv2.rectangle(frame, (0, 40), (bar_w, 44), C["orange"], -1)
                
                cv2.imshow("GestureWave AI", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            gesture = classify_gesture(
                lm=lm,
                frame_width=fw,
                frame_height=fh,
                config=g_config,
            )

            if gesture.gesture == last_gesture_type:
                gesture_stable_frames += 1
            else:
                last_gesture_type = gesture.gesture
                gesture_stable_frames = 1

            # Pause / resume
            if gesture.gesture == GestureType.PAUSE:
                if "PAUSE" in Cfg.ALLOWED_GESTURES:
                    if now - last_pause_time > Cfg.PAUSE_COOLDOWN:
                        last_pause_time = now
                        if state == GState.PAUSED:
                            state = GState.IDLE
                            gesture_name = "Resumed"
                        else:
                            state = GState.PAUSED
                            gesture_name = "Paused ✋"
                else:
                    state = GState.IDLE
                    gesture_name = "Pause (Restricted)"

            if state == GState.PAUSED:
                draw_hud(frame, state, gesture_name, fps_counter.fps, smoothed_norm)
                cv2.imshow("GestureWave AI", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # Right click
            if gesture.gesture == GestureType.RIGHT_CLICK:
                if "RIGHT_CLICK" in Cfg.ALLOWED_GESTURES:
                    if (
                        gesture_stable_frames >= Cfg.ADVANCED_GESTURE_STABLE_FRAMES
                        and now - last_right_click_time > Cfg.CLICK_COOLDOWN
                    ):
                        last_right_click_time = now
                        executor.right_click(prev_sx, prev_sy)
                        state = GState.PINCHING
                        gesture_name = "Right Click 🤟"
                        freeze_start = now
                        freeze_duration = Cfg.RIGHT_CLICK_FREEZE
                    else:
                        state = GState.IDLE
                        gesture_name = "Right Click Ready"
                else:
                    state = GState.IDLE
                    gesture_name = "R-Click (Restricted)"

            # Left pinch
            elif gesture.gesture == GestureType.LEFT_PINCH:
                if "LEFT_PINCH" in Cfg.ALLOWED_GESTURES:
                    state = GState.PINCHING
                    gesture_name = "Pinching 🤏"
                else:
                    state = GState.IDLE
                    gesture_name = "Pinch (Restricted)"

            # Scroll
            elif gesture.gesture == GestureType.SCROLL_UP:
                if "SCROLL_UP" in Cfg.ALLOWED_GESTURES:
                    last_scroll_time, state, gesture_name = handle_advanced(
                        last_scroll_time, Cfg.SCROLL_COOLDOWN, executor.scroll_up, (Cfg.SCROLL_AMOUNT,),
                        GState.SCROLLING, "Scroll Up ✌", "Scroll Ready"
                    )

            elif gesture.gesture == GestureType.SCROLL_DOWN:
                if "SCROLL_DOWN" in Cfg.ALLOWED_GESTURES:
                    last_scroll_time, state, gesture_name = handle_advanced(
                        last_scroll_time, Cfg.SCROLL_COOLDOWN, executor.scroll_down, (Cfg.SCROLL_AMOUNT,),
                        GState.SCROLLING, "Scroll Down ✌", "Scroll Ready"
                    )

            # Zoom
            elif gesture.gesture == GestureType.ZOOM_IN:
                if "ZOOM_IN" in Cfg.ALLOWED_GESTURES:
                    last_zoom_time, state, gesture_name = handle_advanced(
                        last_zoom_time, Cfg.ZOOM_COOLDOWN, executor.zoom_in, (),
                        GState.ZOOMING, "Zoom In 👍", "Zoom Ready"
                    )
                    if state == GState.ZOOMING:
                        freeze_start = now
                        freeze_duration = Cfg.ZOOM_FREEZE

            elif gesture.gesture == GestureType.ZOOM_OUT:
                if "ZOOM_OUT" in Cfg.ALLOWED_GESTURES:
                    last_zoom_time, state, gesture_name = handle_advanced(
                        last_zoom_time, Cfg.ZOOM_COOLDOWN, executor.zoom_out, (),
                        GState.ZOOMING, "Zoom Out 👎", "Zoom Ready"
                    )
                    if state == GState.ZOOMING:
                        freeze_start = now
                        freeze_duration = Cfg.ZOOM_FREEZE
                else:
                    state = GState.IDLE
                    gesture_name = "Zoom Out (Restricted)"

            # Movement / release
            else:
                if state == GState.PINCHING:
                    if now - last_click_time > 0.1:
                        if now - last_left_click_release < Cfg.DOUBLE_CLICK_WINDOW:
                            if "DOUBLE_CLICK" in Cfg.ALLOWED_GESTURES:
                                executor.double_click(prev_sx, prev_sy)
                                gesture_name = "Double Click ⚡"
                                freeze_start = now
                                freeze_duration = Cfg.LEFT_CLICK_FREEZE
                            last_left_click_release = 0.0
                        else:
                            if "LEFT_PINCH" in Cfg.ALLOWED_GESTURES:
                                executor.left_click(prev_sx, prev_sy)
                                gesture_name = "Click 👆"
                                freeze_start = now
                                freeze_duration = Cfg.LEFT_CLICK_FREEZE
                            last_left_click_release = now
                        last_click_time = now
                    state = GState.IDLE

                if gesture.gesture == GestureType.MOVE:
                    if "MOVE" in Cfg.ALLOWED_GESTURES:
                        executor.move_cursor(prev_sx, prev_sy)
                        state = GState.MOVING
                        gesture_name = "Moving ☝"
                    else:
                        state = GState.IDLE
                        gesture_name = "Move (Restricted)"
                elif state not in {GState.PINCHING}:
                    state = GState.IDLE
                    if gesture_name not in {"Click 👆", "Double Click ⚡", "Resumed"}:
                        gesture_name = "Idle"

        else:
            smoother.reset()
            state = GState.IDLE
            gesture_name = "No Hand"
            last_gesture_type = None
            gesture_stable_frames = 0

        draw_hud(frame, state, gesture_name, fps_counter.fps, smoothed_norm)

        cv2.imshow("GestureWave AI", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[GestureWave AI] Exited cleanly.")


if __name__ == "__main__":
    run()