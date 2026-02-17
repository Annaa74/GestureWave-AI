import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    normalized = landmarks - wrist
    max_val = np.max(np.linalg.norm(normalized, axis=1))
    if max_val > 0:
        normalized = normalized / max_val
    return normalized

def finger_open(landmarks, tip, pip):
    return landmarks[tip][1] < landmarks[pip][1]
def is_thumb_open(landmarks, handedness):
    """
    Uses angle + handedness to determine thumb openness
    """
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    wrist = landmarks[0]

    # Vector from MCP to tip
    v1 = thumb_tip - thumb_mcp
    # Vector from MCP to wrist
    v2 = wrist - thumb_mcp

    # Angle between vectors
    angle = np.degrees(
        np.arccos(
            np.dot(v1, v2) /
            (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        )
    )

    # Open thumb has a large angle
    return angle > 40

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
            
        handedness = results.multi_handedness[0].classification[0].label

        landmarks = normalize_landmarks(np.array(landmarks))
        finger_states = {
        "thumb": is_thumb_open(landmarks, handedness),
        "index": finger_open(landmarks, 8, 6),
        "middle": finger_open(landmarks, 12, 10),
        "ring": finger_open(landmarks, 16, 14),
        "pinky": finger_open(landmarks, 20, 18),
    }


        print(finger_states)


        print("Raw landmarks shape:", landmarks.shape)

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

    cv2.imshow("Feature Extraction - Raw Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
