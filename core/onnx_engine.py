import cv2
import numpy as np
import time

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from .engine_base import InferenceEngine

class ONNXInferenceEngine(InferenceEngine):
    """
    Advanced ONNX Inference Engine with DirectML (GPU) acceleration.
    Requires a converted MediaPipe hand_landmark.onnx model.
    """
    
    def __init__(self, model_path="assets/hand_landmark.onnx"):
        self.model_path = model_path
        self.session = None
        self.providers = ['DMLExecutionProvider', 'CPUExecutionProvider']
        
    def initialize(self):
        if ort is None:
            raise ImportError("onnxruntime-directml is not installed. Use 'pip install onnxruntime-directml'.")
        
        print(f"[Advanced] Loading ONNX model from {self.model_path}...")
        try:
            # DirectML is the standard for Windows GPU acceleration across all vendors (AMD/Intel/NVIDIA)
            self.session = ort.InferenceSession(self.model_path, providers=self.providers)
            print("[Advanced] ONNX Runtime initialized with DirectML.")
        except Exception as e:
            print(f"[Warning] Failed to load ONNX model: {e}")
            print("[Advanced] Falling back to CPU/Reference engine.")
            self.session = None

    def process_frame(self, frame: np.ndarray):
        if self.session is None:
            return None, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        # ── Pre-processing ───────────────────────────────────────────
        # This is a simplified prototype. Real implementation requires 
        # cropping based on palm detection, normalization to [0,1], 
        # and resizing to the model's expected input (usually 224x224 or 256x256).
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_data = cv2.resize(rgb, (224, 224))
        input_data = input_data.astype(np.float32) / 255.0
        input_data = np.transpose(input_data, (2, 0, 1)) # HWC to CHW
        input_data = np.expand_dims(input_data, axis=0) # Add batch dim

        # ── Inference ────────────────────────────────────────────────
        input_name = self.session.get_inputs()[0].name
        # result = self.session.run(None, {input_name: input_data})
        
        # ── Post-processing ──────────────────────────────────────────
        # Manual decoding of the 63 output values (21 landmarks * 3 coords)
        # return landmarks, rgb
        return None, rgb

    def release(self):
        self.session = None
