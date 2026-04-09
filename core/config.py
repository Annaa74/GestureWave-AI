class Cfg:
    CAMERA_ID = 0
    FLIP_H = True

    FRAME_WIDTH = 960
    FRAME_HEIGHT = 540
    CAMERA_FPS = 24

    # Cursor smoothing
    SMOOTH_ALPHA = 0.26
    VEL_BOOST_THRESHOLD = 65
    VEL_BOOST_ALPHA = 0.62
    DEAD_ZONE = 5

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

    PALM_OPEN_FINGERS = 4

    SAFE_TOP_BAR = 90
    ENABLE_CLICK_ACTIONS = True

    SCROLL_AMOUNT = 100
    SCROLL_GAP_MAX = 74

    # Stability / debounce
    ADVANCED_GESTURE_STABLE_FRAMES = 2

  