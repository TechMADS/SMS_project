import cv2
import numpy as np
import os
import pickle
from datetime import datetime

class FaceDetector:
    def __init__(self):
        # Load OpenCV's deep learning face detector
        self.prototxt = "utils/deploy.prototxt"
        self.model = "utils/res10_300x300_ssd_iter_140000.caffemodel"
        
        # Download the model files if they don't exist
        if not os.path.exists(self.prototxt) or not os.path.exists(self.model):
            print("Downloading face detection model...")
            self.download_models()
        
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)
        
        # For face recognition (simplified - we'll use basic matching)
        self.known_faces = {}
        self.load_known_faces()
    
    def download_models(self):
        """Download required model files"""
        import urllib.request
        
        # Model URLs
        prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        
        # Download files
        urllib.request.urlretrieve(prototxt_url, self.prototxt)
        urllib.request.urlretrieve(model_url, self.model)
        print("Models downloaded successfully")
    
    def detect(self, image, min_confidence=0.5):
        """Detect faces in an image using OpenCV DNN"""
        (h, w) = image.shape[:2]
        
        # Prepare image for DNN
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 
            1.0, 
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        
        # Pass through network
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        
        # Loop over detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > min_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure bounding box stays within image dimensions
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                width = endX - startX
                height = endY - startY
                
                faces.append((startX, startY, width, height))
        
        return faces
    
    def identify_student(self, face_image):
        """Simple face recognition using template matching"""
        if not self.known_faces:
            return "Unknown"
        
        # Resize face image to standard size
        face_resized = cv2.resize(face_image, (100, 100))
        
        best_match = None
        best_score = 0
        threshold = 0.6  # Similarity threshold
        
        for student_id, template in self.known_faces.items():
            # Resize template if needed
            template_resized = cv2.resize(template, (100, 100))
            
            # Convert to grayscale
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)
            
            # Use template matching
            result = cv2.matchTemplate(face_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            similarity = np.max(result)
            
            if similarity > best_score and similarity > threshold:
                best_score = similarity
                best_match = student_id
        
        return best_match if best_match else "Unknown"
    
    def add_student_face(self, student_id, face_image):
        """Add a student's face to known faces"""
        # Store the face template
        self.known_faces[student_id] = face_image.copy()
        
        # Save to file
        self.save_known_faces()
        
        return True
    
    def load_known_faces(self):
        """Load known faces from file"""
        try:
            if os.path.exists('utils/known_faces.pkl'):
                with open('utils/known_faces.pkl', 'rb') as f:
                    self.known_faces = pickle.load(f)
                print(f"Loaded {len(self.known_faces)} known faces")
        except Exception as e:
            print(f"Error loading known faces: {e}")
            self.known_faces = {}
    
    def save_known_faces(self):
        """Save known faces to file"""
        try:
            with open('utils/known_faces.pkl', 'wb') as f:
                pickle.dump(self.known_faces, f)
        except Exception as e:
            print(f"Error saving known faces: {e}")
    
    def extract_face_features(self, face_image):
        """Extract basic features from face for recognition"""
        # Simple feature extraction using HOG
        face_resized = cv2.resize(face_image, (64, 128))
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate HOG features
        from skimage.feature import hog
        features, _ = hog(
            gray, 
            orientations=9, 
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), 
            visualize=True
        )
        
        return features
    
    def draw_face_boxes(self, image, faces):
        """Draw bounding boxes around detected faces"""
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return image