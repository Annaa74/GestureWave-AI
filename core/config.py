class Cfg:
    CAMERA_ID = 0
    FLIP_H = True

    FRAME_WIDTH = 960
    FRAME_HEIGHT = 540
    CAMERA_FPS = 24

    # Cursor smoothing
    SMOOTH_ALPHA = 0.40
    VEL_BOOST_THRESHOLD = 65
    VEL_BOOST_ALPHA = 0.70
    DEAD_ZONE = 3

    # Gesture thresholds
    CLICK_THRESH = 30
    RIGHT_CLICK_THRESH = 32
    PINCH_RELEASE = 50

    # Timing / hold behavior
    CLICK_COOLDOWN = 0.22
    DOUBLE_CLICK_WINDOW = 0.36
    SCROLL_COOLDOWN = 0.08
    ZOOM_COOLDOWN = 0.10
    PAUSE_COOLDOWN = 1.00

    # Freeze times
    LEFT_CLICK_FREEZE = 2.0
    RIGHT_CLICK_FREEZE = 3.0
    ZOOM_FREEZE = 6.0

    PALM_OPEN_FINGERS = 4

    SAFE_TOP_BAR = 90
    ENABLE_CLICK_ACTIONS = True

    SCROLL_AMOUNT = 100
    SCROLL_GAP_MAX = 74

    # Stability / debounce
    ADVANCED_GESTURE_STABLE_FRAMES = 4

    # ── Security & Auth ────────────────────────────────────────────────────────
    USER_ROLE = "guest"  # 'guest', 'standard', 'admin'
    
    # Safe Mode defaults: Zoom Out is disabled by default due to reliability issues.
    ALLOWED_GESTURES = {
        "MOVE",
        "LEFT_PINCH",
        "RIGHT_CLICK",
        "DOUBLE_CLICK",
        "SCROLL_UP",
        "SCROLL_DOWN",
        "ZOOM_IN",
        "PAUSE"
    }