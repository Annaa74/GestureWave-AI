import cv2
import mediapipe as mp
import numpy as np
from .engine_base import InferenceEngine

class MediaPipeEngine(InferenceEngine):
    """
    Standard MediaPipe implementation wrapped in the InferenceEngine interface.
    """
    
    def __init__(self, min_detection_confidence=0.72, min_tracking_confidence=0.60):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.hands = None
        
    def initialize(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        print("[Engine] MediaPipe Inference Engine initialized.")

    def process_frame(self, frame: np.ndarray):
        if self.hands is None:
            self.initialize()
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        
        landmarks = None
        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            
        return landmarks, rgb

    def release(self):
        if self.hands:
            self.hands.close()
            self.hands = None
