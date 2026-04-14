from abc import ABC, abstractmethod
import numpy as np

class InferenceEngine(ABC):
    """
    Abstract base class for hand landmark inference engines.
    Allows for hot-swapping between MediaPipe and ONNX/TensorRT backends.
    """
    
    @abstractmethod
    def initialize(self):
        """Setup model weights and hardware delegates."""
        pass
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray):
        """
        Process a single BGR frame and return hand landmarks.
        Returns:
            landmarks: MediaPipe-style landmark list or None if no hand detected.
            raw_rgb: The processed RGB frame for visualization.
        """
        pass
    
    @abstractmethod
    def release(self):
        """Clean up resources and release hardware."""
        pass
