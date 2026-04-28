class Cfg:
    CAMERA_ID = 0
    FLIP_H = True

    FRAME_WIDTH = 960
    FRAME_HEIGHT = 540
    CAMERA_FPS = 24

    # Cursor smoothing
    SMOOTH_ALPHA = 0.30
    VEL_BOOST_THRESHOLD = 35
    VEL_BOOST_ALPHA = 0.85
    DEAD_ZONE = 2

    # Gesture thresholds
    CLICK_THRESH = 30
    RIGHT_CLICK_THRESH = 32
    PINCH_RELEASE = 50

    # Timing / hold behavior
    CLICK_COOLDOWN = 0.30        # ← Increased: prevents rapid re-clicks
    DOUBLE_CLICK_WINDOW = 0.40
    SCROLL_COOLDOWN = 0.08       # ← Slightly increased for smoother scrolling
    ZOOM_COOLDOWN = 0.10         # ← Slightly increased
    PAUSE_COOLDOWN = 0.50

    # Freeze times — REDUCED for responsiveness
    # These only lock the cursor BRIEFLY after an action fires
    LEFT_CLICK_FREEZE = 0.05     # ← Was 0.15, now barely noticeable
    RIGHT_CLICK_FREEZE = 0.12    # ← Was 0.30, feels snappy now
    ZOOM_FREEZE = 0.15           # ← Was 0.40, much faster

    PALM_OPEN_FINGERS = 4

    SAFE_TOP_BAR = 90
    ENABLE_CLICK_ACTIONS = True

    SCROLL_AMOUNT = 100
    SCROLL_GAP_MAX = 74

    # Stability / debounce
    ADVANCED_GESTURE_STABLE_FRAMES = 4
    PINCH_STABLE_FRAMES = 3      # ← NEW: require 3 stable frames before pinch registers

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
# Core configuration initialized
