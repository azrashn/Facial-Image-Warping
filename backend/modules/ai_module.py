import cv2
import numpy as np
import base64
from deepface import DeepFace

class AIProcessor:
    def __init__(self):
        self.model_name = "age"
        self.target_size = (512, 512)

    def _validate_image(self, image: np.ndarray):
        if image is None:
            return {"error": "Input image is None."}
        if not isinstance(image, np.ndarray):
            return {"error": "Image must be a numpy array."}
        if image.ndim != 3 or image.shape[2] != 3:
            return {"error": "Image must be a 3-channel (BGR/RGB) image."}
        return None

    def _to_standard_format(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, self.target_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.uint8)

    def _encode_to_base64(self, image: np.ndarray) -> str:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".png", bgr)
        return base64.b64encode(buffer).decode("utf-8")

    def analyze_age(self, image: np.ndarray) -> dict:
        error = self._validate_image(image)
        if error:
            return error
        try:
            standardized = self._to_standard_format(image)
            results = DeepFace.analyze(
                standardized,
                actions=["age"],
                enforce_detection=False
            )
            if isinstance(results, list):
                estimated_age = results[0]["age"]
            else:
                estimated_age = results["age"]
            image_b64 = self._encode_to_base64(standardized)
            return {
                "estimated_age": int(estimated_age),
                "status": "success",
                "image_b64": image_b64,
                "image_size": standardized.shape[:2],
            }
        except Exception as e:
            return {"error": f"AI Analysis failed: {str(e)}", "status": "failed"}

    def apply_ai_transformation(self, image: np.ndarray) -> dict:
        error = self._validate_image(image)
        if error:
            return error
        try:
            standardized = self._to_standard_format(image)
            image_b64 = self._encode_to_base64(standardized)
            return {
                "status": "success",
                "note": "GAN pipeline not implemented yet.",
                "image_b64": image_b64,
                "image_size": standardized.shape[:2],
            }
        except Exception as e:
            return {"error": f"Transformation failed: {str(e)}", "status": "failed"}
