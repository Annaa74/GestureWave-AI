import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque
from enum import Enum, auto

from core.config import Cfg

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0


class GState(Enum):
    IDLE = auto()
    MOVING = auto()
    PINCHING = auto()
    DRAGGING = auto()
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


def draw_hud(frame, state: GState, gesture_name: str, fps: float, smoothed_xy):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), C["hud_bg"], -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    state_color = {
        GState.IDLE: C["grey"],
        GState.MOVING: C["blue"],
        GState.PINCHING: C["yellow"],
        GState.DRAGGING: C["orange"],
        GState.PAUSED: C["red"],
    }.get(state, C["white"])

    cv2.putText(frame, "GestureWave AI Stable Mode", (10, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.58, C["white"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:4.1f}", (290, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C["grey"], 1, cv2.LINE_AA)
    cv2.putText(frame, gesture_name, (410, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2, cv2.LINE_AA)

    cv2.circle(frame, (w - 20, 20), 7, state_color, -1)

    if smoothed_xy and state != GState.PAUSED:
        sx, sy = int(smoothed_xy[0] * w), int(smoothed_xy[1] * h)
        cv2.drawMarker(frame, (sx, sy), C["blue"], cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)


def is_finger_up(lm, tip_idx, pip_idx):
    return lm[tip_idx].y < lm[pip_idx].y


def finger_states(lm):
    index_up = is_finger_up(lm, 8, 6)
    middle_up = is_finger_up(lm, 12, 10)
    ring_up = is_finger_up(lm, 16, 14)
    pinky_up = is_finger_up(lm, 20, 18)
    return index_up, middle_up, ring_up, pinky_up


def is_open_palm(lm):
    index_up, middle_up, ring_up, pinky_up = finger_states(lm)
    return sum([index_up, middle_up, ring_up, pinky_up]) >= Cfg.PALM_OPEN_FINGERS


def is_index_only(lm):
    index_up, middle_up, ring_up, pinky_up = finger_states(lm)
    return index_up and (not middle_up) and (not ring_up) and (not pinky_up)


def run():
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
    safe_top_bar = Cfg.SAFE_TOP_BAR

    smoother = EMASmoother(Cfg.SMOOTH_ALPHA)
    fps_counter = FPSCounter()

    state = GState.IDLE
    gesture_name = "Idle"

    prev_sx = 0.0
    prev_sy = 0.0

    pinch_hold_frames = 0
    drag_active = False

    last_click_time = 0.0
    last_pause_time = 0.0

    print("[GestureWave AI Stable Mode] Starting. Press ESC to exit.")

    while True:
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

            def px(idx):
                return int(lm[idx].x * fw), int(lm[idx].y * fh)

            thumb_pt = px(4)
            index_pt = px(8)

            d_left = dist(thumb_pt, index_pt)

            raw_sx = lm[8].x * screen_w
            raw_sy = max(lm[8].y * screen_h, safe_top_bar)

            vel = dist((raw_sx, raw_sy), (prev_sx, prev_sy))
            sm_sx, sm_sy = smoother.update(raw_sx, raw_sy, vel)

            if dist((sm_sx, sm_sy), (prev_sx, prev_sy)) > Cfg.DEAD_ZONE:
                prev_sx, prev_sy = sm_sx, sm_sy

            smoothed_norm = (lm[8].x, lm[8].y)
            now = time.perf_counter()

            # Pause / resume
            if is_open_palm(lm):
                if now - last_pause_time > Cfg.PAUSE_COOLDOWN:
                    last_pause_time = now
                    if state == GState.PAUSED:
                        state = GState.IDLE
                        gesture_name = "Resumed"
                    else:
                        if drag_active:
                            pyautogui.mouseUp()
                            drag_active = False
                        state = GState.PAUSED
                        gesture_name = "Paused ✋"

            if state == GState.PAUSED:
                draw_hud(frame, state, gesture_name, fps_counter.fps, smoothed_norm)
                cv2.imshow("GestureWave AI", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # Pinch / click / drag
            if d_left < Cfg.CLICK_THRESH:
                pinch_hold_frames += 1

                if pinch_hold_frames >= Cfg.DRAG_HOLD_FRAMES and not drag_active:
                    if prev_sy > safe_top_bar and Cfg.ENABLE_CLICK_ACTIONS:
                        pyautogui.mouseDown()
                        drag_active = True
                    state = GState.DRAGGING
                    gesture_name = "Dragging 🤏"

                if drag_active:
                    pyautogui.moveTo(prev_sx, prev_sy, _pause=False)
                    state = GState.DRAGGING
                    gesture_name = "Dragging 🤏"
                    cv2.circle(frame, index_pt, 18, C["orange"], cv2.FILLED)
                    cv2.circle(frame, thumb_pt, 14, C["orange"], cv2.FILLED)
                else:
                    state = GState.PINCHING
                    gesture_name = "Pinching…"
                    cv2.circle(frame, index_pt, 18, C["yellow"], cv2.FILLED)
                    cv2.circle(frame, thumb_pt, 14, C["yellow"], cv2.FILLED)

            else:
                if drag_active:
                    pyautogui.mouseUp()
                    drag_active = False
                    state = GState.IDLE
                    gesture_name = "Drop"

                elif state == GState.PINCHING and pinch_hold_frames < Cfg.DRAG_HOLD_FRAMES:
                    if now - last_click_time > Cfg.CLICK_COOLDOWN:
                        last_click_time = now
                        if prev_sy > safe_top_bar and Cfg.ENABLE_CLICK_ACTIONS:
                            pyautogui.click(prev_sx, prev_sy)
                            gesture_name = "Click 👆"
                        else:
                            gesture_name = "Click Blocked"
                    state = GState.IDLE

                pinch_hold_frames = 0

                if is_index_only(lm) and d_left > Cfg.PINCH_RELEASE:
                    state = GState.MOVING
                    gesture_name = "Moving ☝"
                    pyautogui.moveTo(prev_sx, prev_sy, _pause=False)
                    cv2.circle(frame, index_pt, 10, C["blue"], cv2.FILLED)
                elif state not in {GState.PINCHING, GState.DRAGGING}:
                    state = GState.IDLE
                    if gesture_name not in {"Click 👆", "Drop", "Resumed"}:
                        gesture_name = "Idle"

        else:
            smoother.reset()
            if drag_active:
                pyautogui.mouseUp()
                drag_active = False
            state = GState.IDLE
            gesture_name = "No Hand"
            pinch_hold_frames = 0

        draw_hud(frame, state, gesture_name, fps_counter.fps, smoothed_norm)

        cv2.imshow("GestureWave AI", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    if drag_active:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()
    print("[GestureWave AI] Exited cleanly.")


if __name__ == "__main__":
    run()