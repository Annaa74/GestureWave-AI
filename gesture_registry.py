import json
import os
import numpy as np
from gesture_utils import calculate_distance

class GestureRegistry:
    def __init__(self, filepath="custom_gestures.json"):
        self.filepath = filepath
        self.gestures = {}
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    data = json.load(f)
                    # Convert lists back to numpy arrays
                    self.gestures = {name: np.array(lms) for name, lms in data.items()}
            except Exception as e:
                print(f"[Registry] Error loading gestures: {e}")
                self.gestures = {}

    def save(self):
        try:
            # Convert numpy arrays to lists for JSON serialization
            data = {name: lms.tolist() for name, lms in self.gestures.items()}
            with open(self.filepath, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"[Registry] Error saving gestures: {e}")

    def add_gesture(self, name, normalized_landmarks):
        self.gestures[name] = normalized_landmarks
        self.save()

    def recognize(self, current_landmarks, threshold=0.15):
        best_match = None
        min_dist = float('inf')

        for name, saved_lms in self.gestures.items():
            dist = calculate_distance(current_landmarks, saved_lms)
            if dist < min_dist:
                min_dist = dist
                best_match = name

        if min_dist < threshold:
            return best_match, min_dist
        return None, None
