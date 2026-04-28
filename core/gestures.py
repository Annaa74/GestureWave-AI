from dataclasses import dataclass
from enum import Enum, auto
import math
import mediapipe as mp

HL = mp.solutions.hands.HandLandmark


class GestureType(Enum):
    IDLE = auto()
    MOVE = auto()
    LEFT_PINCH = auto()
    RIGHT_CLICK = auto()
    DOUBLE_CLICK = auto()
    SCROLL_UP = auto()
    SCROLL_DOWN = auto()
    ZOOM_IN = auto()
    ZOOM_OUT = auto()
    PAUSE = auto()
    BANNED = auto()       # Offensive gestures (middle finger, etc.)


@dataclass
class GestureResult:
    gesture: GestureType
    left_pinch_distance: float
    right_pinch_distance: float
    two_finger_gap: float
    index_up: bool
    middle_up: bool
    ring_up: bool
    pinky_up: bool


@dataclass
class GestureConfig:
    click_thresh: float
    right_click_thresh: float
    pinch_release: float
    palm_open_fingers: int = 4
    scroll_gap_max: float = 75.0

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def is_finger_up(lm, tip_idx, pip_idx):
    return lm[tip_idx].y < lm[pip_idx].y


def finger_states(lm):
    index_up = is_finger_up(lm, HL.INDEX_FINGER_TIP, HL.INDEX_FINGER_PIP)
    middle_up = is_finger_up(lm, HL.MIDDLE_FINGER_TIP, HL.MIDDLE_FINGER_PIP)
    ring_up = is_finger_up(lm, HL.RING_FINGER_TIP, HL.RING_FINGER_PIP)
    pinky_up = is_finger_up(lm, HL.PINKY_TIP, HL.PINKY_PIP)
    return index_up, middle_up, ring_up, pinky_up


def is_open_palm(lm, required_fingers=4):
    index_up, middle_up, ring_up, pinky_up = finger_states(lm)
    count = sum([index_up, middle_up, ring_up, pinky_up])
    return count >= required_fingers


def is_index_only(lm):
    index_up, middle_up, ring_up, pinky_up = finger_states(lm)
    return index_up and (not middle_up) and (not ring_up) and (not pinky_up)


def is_middle_finger_only(lm):
    """Detect middle finger gesture (offensive — only middle finger extended)."""
    index_up, middle_up, ring_up, pinky_up = finger_states(lm)
    # Middle finger is up, all other fingers are down
    return (not index_up) and middle_up and (not ring_up) and (not pinky_up)


def classify_gesture(
    lm,
    frame_width,
    frame_height,
    config: GestureConfig,
):
    def px(idx):
        return int(lm[idx].x * frame_width), int(lm[idx].y * frame_height)

    thumb_pt = px(HL.THUMB_TIP)
    index_pt = px(HL.INDEX_FINGER_TIP)
    middle_pt = px(HL.MIDDLE_FINGER_TIP)

    left_pinch = distance(thumb_pt, index_pt)
    right_pinch = distance(thumb_pt, middle_pt)
    two_finger_gap = distance(index_pt, middle_pt)

    index_up, middle_up, ring_up, pinky_up = finger_states(lm)

    # ── BANNED GESTURES (checked FIRST — highest priority) ──────────────────
    # Middle finger only = offensive gesture → block immediately
    if is_middle_finger_only(lm):
        return GestureResult(GestureType.BANNED, left_pinch, right_pinch, two_finger_gap, index_up, middle_up, ring_up, pinky_up)

    # Right click: three fingers up (index, middle, ring)
    if index_up and middle_up and ring_up and (not pinky_up):
        return GestureResult(GestureType.RIGHT_CLICK, left_pinch, right_pinch, two_finger_gap, index_up, middle_up, ring_up, pinky_up)

    # Left pinch
    if index_up and left_pinch < config.click_thresh:
        return GestureResult(GestureType.LEFT_PINCH, left_pinch, right_pinch, two_finger_gap, index_up, middle_up, ring_up, pinky_up)

    # Two-finger gestures
    if index_up and middle_up and (not ring_up) and (not pinky_up) and left_pinch > config.pinch_release:
        # Scroll: fingers closer together
        if two_finger_gap <= config.scroll_gap_max:
            if lm[HL.INDEX_FINGER_TIP].y < 0.45:
                return GestureResult(GestureType.SCROLL_UP, left_pinch, right_pinch, two_finger_gap, index_up, middle_up, ring_up, pinky_up)
            return GestureResult(GestureType.SCROLL_DOWN, left_pinch, right_pinch, two_finger_gap, index_up, middle_up, ring_up, pinky_up)

    # Zoom In: thumbs up (upright fist + thumb extended up)
    if (not index_up) and (not middle_up) and (not ring_up) and (not pinky_up):
        if lm[HL.THUMB_TIP].y < lm[HL.THUMB_MCP].y - 0.02:
            return GestureResult(GestureType.ZOOM_IN, left_pinch, right_pinch, two_finger_gap, index_up, middle_up, ring_up, pinky_up)

    # Zoom Out: thumbs down (inverted fist + thumb extended down)
    if (not index_up) and (not middle_up) and (not ring_up) and (not pinky_up):
        if lm[HL.THUMB_TIP].y > lm[HL.THUMB_MCP].y + 0.02:
            return GestureResult(GestureType.ZOOM_OUT, left_pinch, right_pinch, two_finger_gap, index_up, middle_up, ring_up, pinky_up)

    if is_index_only(lm) and left_pinch > config.pinch_release and right_pinch > config.pinch_release:
        return GestureResult(GestureType.MOVE, left_pinch, right_pinch, two_finger_gap, index_up, middle_up, ring_up, pinky_up)

    # Pause (Open Palm): Catch this last so it doesn't swallow strict inverted gestures
    if is_open_palm(lm, config.palm_open_fingers):
        return GestureResult(GestureType.PAUSE, left_pinch, right_pinch, two_finger_gap, index_up, middle_up, ring_up, pinky_up)

    return GestureResult(GestureType.IDLE, left_pinch, right_pinch, two_finger_gap, index_up, middle_up, ring_up, pinky_up)
# Gesture tracking module
