import cv2
import numpy as np
from deepface import DeepFace
import logging
from typing import Dict, Optional, Tuple

class FaceVerificationSystem:
    def __init__(self, model_name: str = 'ArcFace', threshold: float = 0.4):
        """
        Initialize face verification system
        
        Args:
            model_name: 'VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib'
            threshold: Similarity threshold (model-specific)
        """
        self.model_name = model_name
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        
    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess image"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                self.logger.error(f"Could not load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Optional: resize if too large
            height, width = img_rgb.shape[:2]
            if height > 1000 or width > 1000:
                scale = 1000 / max(height, width)
                new_size = (int(width * scale), int(height * scale))
                img_rgb = cv2.resize(img_rgb, new_size)
            
            return img_rgb
        except Exception as e:
            self.logger.error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def verify_faces(
        self, 
        selfie_path: str, 
        id_path: str,
        enforce_detection: bool = True
    ) -> Dict:
        """
        Verify if two faces belong to the same person
        
        Returns:
            Dictionary with verification results
        """
        try:
            # Verify faces using DeepFace
            result = DeepFace.verify(
                img1_path=selfie_path,
                img2_path=id_path,
                model_name=self.model_name,
                detector_backend='mtcnn',  # Good balance of accuracy/speed
                distance_metric='cosine',
                enforce_detection=enforce_detection,
                align=True  # Align faces for better accuracy
            )
            
            # Extract face regions if available
            face1_region = result.get('facial_areas', {}).get('img1', {})
            face2_region = result.get('facial_areas', {}).get('img2', {})
            
            # Calculate similarity score (0-100%)
            distance = result['distance']
            similarity_score = self._distance_to_score(distance)
            
            return {
                "success": True,
                "verified": result["verified"],
                "similarity_score": similarity_score,
                "distance": float(distance),
                "threshold": float(result["threshold"]),
                "model": self.model_name,
                "face_detected": True,
                "face_regions": {
                    "selfie": face1_region,
                    "id": face2_region
                },
                "message": "Verification successful"
            }
            
        except ValueError as e:
            if "Face could not be detected" in str(e):
                return {
                    "success": False,
                    "verified": False,
                    "error": "No face detected in one or both images",
                    "face_detected": False,
                    "message": str(e)
                }
            else:
                return {
                    "success": False,
                    "verified": False,
                    "error": str(e),
                    "message": "Verification failed"
                }
        except Exception as e:
            self.logger.error(f"Error in face verification: {e}")
            return {
                "success": False,
                "verified": False,
                "error": str(e),
                "message": "Internal error"
            }
    
    def _distance_to_score(self, distance: float) -> float:
        """Convert distance metric to similarity score (0-100%)"""
        # This conversion depends on the model and distance metric
        # For cosine distance with FaceNet/ArcFace models:
        if self.model_name in ['Facenet', 'ArcFace']:
            # Cosine distance ranges from 0 to 2
            # Convert to similarity score (0-1)
            similarity = 1 - (distance / 2)
            return max(0, min(1, similarity)) * 100
        else:
            # Generic conversion
            similarity = 1 - min(distance, 1.0)
            return similarity * 100
    
    def analyze_single_face(self, image_path: str) -> Dict:
        """Analyze a single face (demographics, emotions, etc.)"""
        try:
            analysis = DeepFace.analyze(
                img_path=image_path,
                actions=['age', 'gender', 'emotion', 'race'],
                detector_backend='mtcnn',
                enforce_detection=False,
                silent=True
            )
            
            if analysis and len(analysis) > 0:
                return {
                    "success": True,
                    "analysis": analysis[0],
                    "face_detected": True
                }
            else:
                return {
                    "success": False,
                    "error": "No face detected",
                    "face_detected": False
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "face_detected": False
            }

# Usage example
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize verifier
    verifier = FaceVerificationSystem(model_name='ArcFace', threshold=0.4)
    
    # Verify faces
    result = verifier.verify_faces("selfie.jpg", "id_photo.jpg")
    
    print("Verification Results:")
    print(f"  Same person: {result.get('verified')}")
    print(f"  Similarity: {result.get('similarity_score', 0):.1f}%")
    print(f"  Distance: {result.get('distance', 0):.4f}")
    print(f"  Threshold: {result.get('threshold', 0):.4f}")
    print(f"  Message: {result.get('message')}")
    
    if result.get('verified'):
        print("✅ Faces match!")
    else:
        print("❌ Faces don't match or couldn't verify")