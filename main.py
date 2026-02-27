import cv2
import mediapipe as mp
import pyautogui
# import numpy as np
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# # Disable pyautogui failsafe for smooth control (optional, re-enable if needed)
# pyautogui.FAILSAFE = False
# pyautogui.PAUSE = 0.01

# # Screen size
# screen_width, screen_height = pyautogui.size()
# print(f"Screen: {screen_width}x{screen_height}")

# # Model path - download from https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models
# MODEL_PATH = 'hand_landmarker.task'  # Lite model recommended

# # Global vars for mouse positions
# index_x, index_y = 0, 0
# thumb_x, thumb_y = 0, 0

# # Callback to process detection results
# def mouse_callback(result, output_image: mp.Image, timestamp_ms: int):
#     global index_x, index_y, thumb_x, thumb_y
    
#     frame = output_image.numpy_view()  # BGR frame from callback
#     h, w, _ = frame.shape
    
#     if result.hand_landmarks:
#         for hand_landmarks in result.hand_landmarks:
#             # Draw landmarks connections (optional)
#             mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks)
            
#             # Index finger tip (landmark 8)
#             lm8 = hand_landmarks[8]
#             cx8, cy8 = int(lm8.x * w), int(lm8.y * h)
#             cv2.circle(frame, (cx8, cy8), 15, (0, 255, 255), cv2.FILLED)
#             cv2.circle(frame, (cx8, cy8), 15, (255, 255, 255), 2)
            
#             index_x = np.interp(lm8.x, (0, 1), (0, screen_width))
#             index_y = np.interp(lm8.y, (0, 1), (0, screen_height))
#             pyautogui.moveTo(index_x, index_y)
            
#             # Thumb tip (landmark 4)
#             lm4 = hand_landmarks[4]
#             cx4, cy4 = int(lm4.x * w), int(lm4.y * h)
#             cv2.circle(frame, (cx4, cy4), 15, (0, 255, 0), cv2.FILLED)
#             cv2.circle(frame, (cx4, cy4), 15, (255, 255, 255), 2)
            
#             thumb_x = np.interp(lm4.x, (0, 1), (0, screen_width))
#             thumb_y = np.interp(lm4.y, (0, 1), (0, screen_height))
            
#             # Pinch detection (distance threshold)
#             dist = np.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)
#             if dist < 50:  # Adjust threshold as needed
#                 pyautogui.click()
#                 print("Click!")
#                 pyautogui.sleep(0.5)  # Debounce

# # Setup HandLandmarker options
# BaseOptions = mp.tasks.BaseOptions
# HandLandmarker = mp.tasks.vision.HandLandmarker
# HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
# VisionRunningMode = mp.tasks.vision.RunningMode

# options = HandLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=MODEL_PATH),
#     running_mode=VisionRunningMode.LIVE_STREAM,
#     num_hands=1,  # Single hand for simplicity
#     min_hand_detection_confidence=0.7,
#     min_hand_presence_confidence=0.7,
#     min_tracking_confidence=0.7,
#     result_callback=mouse_callback
# )

# # Webcam setup
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# timestamp = 0

# print("Starting virtual mouse... Press 'q' to quit.")

# with HandLandmarker.create_from_options(options) as landmarker:
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             continue
        
#         frame = cv2.flip(frame, 1)  # Mirror
#         frame_height, frame_width, _ = frame.shape
        
#         # Convert to mp.Image
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
#         # Send to landmarker (async)
#         landmarker.detect_async(mp_image, timestamp)
#         timestamp += 33  # ~30 FPS
        
#         # Show frame (updated in callback)
#         cv2.imshow('Virtual Mouse', frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()
# pyautogui.FAILSAFE = True  # Re-enable failsafe
# print("Stopped.")









cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_x, index_y = 0, 0
thumb_x, thumb_y = 0, 0
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    if hands:
        for hand in hands:
           drawing_utils.draw_landmarks(frame, hand) 
           landmarks = hand.landmark
           for id, landmark in enumerate(landmarks):
               h, w, c = frame.shape
               cx, cy = int(landmark.x * w), int(landmark.y * h)
               print(id, cx, cy)
               if id == 8:
                   cv2.circle(frame, (cx, cy), 10, (0, 255, 255), cv2.FILLED) 
                   index_x = screen_width / frame_width * cx
                   index_y = screen_height / frame_height * cy    
                   pyautogui.moveTo(index_x, index_y)
               if id == 4:
                   cv2.circle(frame, (cx, cy), 10, (0, 255, 255), cv2.FILLED) 
                   thumb_x = screen_width / frame_width * cx
                   thumb_y = screen_height / frame_height * cy
                   print(abs(index_x - thumb_x)) 
                   if abs(index_x - thumb_x) < 20 and abs(index_y - thumb_y) < 20:
                       pyautogui.click()
                       pyautogui.sleep(1) 
    cv2.imshow('Virtual Mouse', frame)
    cv2.waitKey(1)