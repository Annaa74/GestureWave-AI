import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

# Optional: PyAutoGUI fail-safe
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS) 
            landmarks = hand.landmark
            
            # Tip ids
            THUMB_TIP, INDEX_TIP, MIDDLE_TIP = 4, 8, 12
            
            thumb_x = int(landmarks[THUMB_TIP].x * frame_width)
            thumb_y = int(landmarks[THUMB_TIP].y * frame_height)
            index_x = int(landmarks[INDEX_TIP].x * frame_width)
            index_y = int(landmarks[INDEX_TIP].y * frame_height)
            middle_x = int(landmarks[MIDDLE_TIP].x * frame_width)
            middle_y = int(landmarks[MIDDLE_TIP].y * frame_height)
            
            mouse_x = screen_width / frame_width * index_x
            mouse_y = screen_height / frame_height * index_y
            
            # Distances
            dist_left_click = ((thumb_x - index_x)**2 + (thumb_y - index_y)**2)**0.5
            dist_right_click = ((thumb_x - middle_x)**2 + (thumb_y - middle_y)**2)**0.5
            dist_fingers = ((index_x - middle_x)**2 + (index_y - middle_y)**2)**0.5
            
            threshold = 30
            
            # If index and middle fingers are close, activate scrolling
            if dist_fingers < 40 and dist_left_click > 50:
                cv2.circle(frame, (index_x, index_y), 15, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, (middle_x, middle_y), 15, (0, 255, 0), cv2.FILLED)
                
                # Scroll direction based on middle finger position
                if middle_y < frame_height / 2:
                    pyautogui.scroll(20) # Scroll up
                else:
                    pyautogui.scroll(-20) # Scroll down
            else:
                # Move mouse
                pyautogui.moveTo(mouse_x, mouse_y)
                cv2.circle(frame, (index_x, index_y), 10, (255, 0, 0), cv2.FILLED)
                
                # Check Clicks
                if dist_left_click < threshold:
                    cv2.circle(frame, (thumb_x, thumb_y), 15, (0, 255, 255), cv2.FILLED)
                    pyautogui.click()
                    pyautogui.sleep(0.3)
                elif dist_right_click < threshold:
                    cv2.circle(frame, (middle_x, middle_y), 15, (0, 0, 255), cv2.FILLED)
                    pyautogui.rightClick()
                    pyautogui.sleep(0.3)

    cv2.imshow('GestureWave AI', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()