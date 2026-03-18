import numpy as np

def normalize_landmarks(landmarks):
    """
    Normalizes hand landmarks:
    1. Translates wrist (index 0) to origin (0,0,0).
    2. Scales landmarks so the maximum distance from the wrist is 1.0.
    """
    # Convert to numpy array if not already
    lms = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    wrist = lms[0]
    normalized = lms - wrist
    
    # Scale by max distance to keep it size-invariant
    max_val = np.max(np.linalg.norm(normalized, axis=1))
    if max_val > 0:
        normalized = normalized / max_val
        
    return normalized

def calculate_distance(lms1, lms2):
    """
    Calculates Euclidean distance between two sets of normalized landmarks.
    """
    return np.linalg.norm(lms1 - lms2)
