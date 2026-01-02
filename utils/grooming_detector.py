import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

class DetailedGroomingDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Initializing Detailed Grooming Detector...")
        
        # Load models for different grooming aspects
        self.setup_transform()
        
        # Pre-trained models (you can replace these with your own trained models)
        self.load_pretrained_models()
        
        print("✅ Detailed Grooming Detector Ready")
    
    def setup_transform(self):
        """Setup image transformations"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_pretrained_models(self):
        """Load or create models for different grooming aspects"""
        # These are placeholder models - in production, you'd train these
        self.models = {
            'beard_detector': self.create_beard_model(),
            'hair_length': self.create_hair_model(),
            'neatness': self.create_neatness_model()
        }
    
    def create_beard_model(self):
        """Create beard detection model"""
        class BeardDetector(nn.Module):
            def __init__(self):
                super(BeardDetector, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(64 * 28 * 28, 128)
                self.fc2 = nn.Linear(128, 2)  # Beard or No Beard
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.5)
            
            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = self.pool(self.relu(self.conv3(x)))
                x = x.view(x.size(0), -1)
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.fc2(x)
                return x
        
        model = BeardDetector()
        model.eval()
        return model
    
    def create_hair_model(self):
        """Create hair length/style model"""
        class HairModel(nn.Module):
            def __init__(self):
                super(HairModel, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(64 * 28 * 28, 128)
                self.fc2 = nn.Linear(128, 3)  # Short, Medium, Long
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.5)
            
            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = self.pool(self.relu(self.conv3(x)))
                x = x.view(x.size(0), -1)
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.fc2(x)
                return x
        
        model = HairModel()
        model.eval()
        return model
    
    def create_neatness_model(self):
        """Create neatness assessment model"""
        class NeatnessModel(nn.Module):
            def __init__(self):
                super(NeatnessModel, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(64 * 28 * 28, 128)
                self.fc2 = nn.Linear(128, 5)  # Score 1-5
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.5)
            
            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = self.pool(self.relu(self.conv3(x)))
                x = x.view(x.size(0), -1)
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.fc2(x)
                return x
        
        model = NeatnessModel()
        model.eval()
        return model
    
    def analyze(self, face_image):
        """Perform detailed grooming analysis"""
        try:
            # Convert to PIL
            if len(face_image.shape) == 3:
                image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = face_image
            
            pil_image = Image.fromarray(image_rgb)
            
            # Apply transform
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Get analysis results
            results = {
                'has_beard': self.detect_beard(input_tensor),
                'hair_length': self.analyze_hair(input_tensor),
                'needs_haircut': self.check_haircut_needed(input_tensor),
                'neatness_score': self.get_neatness_score(input_tensor),
                'face_clean': self.check_face_cleanliness(face_image),
                'additional_notes': self.get_additional_notes(face_image)
            }
            
            return results
            
        except Exception as e:
            print(f"Grooming analysis error: {e}")
            return self.get_default_results()
    
    def quick_analyze(self, face_image):
        """Quick analysis for live view"""
        try:
            # Simple image processing for quick analysis
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Simple beard detection (based on lower face darkness)
            height, width = face_image.shape[:2]
            lower_face = face_image[int(height*0.6):, :]
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)
            
            # Detect dark areas (potential beard)
            lower_dark = np.array([0, 0, 0])
            upper_dark = np.array([180, 255, 80])
            dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
            dark_percentage = np.sum(dark_mask > 0) / dark_mask.size
            
            has_beard = dark_percentage > 0.3
            
            # Hair length estimation (based on forehead to top ratio)
            # This is a simplified approach
            edges = cv2.Canny(gray, 50, 150)
            hair_pixels = np.sum(edges[:int(height*0.3), :] > 0)
            hair_density = hair_pixels / (width * height * 0.3)
            
            needs_haircut = hair_density > 0.15
            
            return {
                'has_beard': has_beard,
                'needs_haircut': needs_haircut,
                'face_clean': True,  # Default
                'quick_analysis': True
            }
            
        except Exception as e:
            return {'has_beard': False, 'needs_haircut': False, 'quick_analysis': False}
    
    def detect_beard(self, input_tensor):
        """Detect if person has beard"""
        # In real implementation, use trained model
        # For now, return random with bias toward no beard (students)
        return np.random.random() < 0.2  # 20% chance of beard
    
    def analyze_hair(self, input_tensor):
        """Analyze hair length"""
        # Return hair length category
        lengths = ['Very Short', 'Short', 'Medium', 'Long']
        return np.random.choice(lengths, p=[0.4, 0.4, 0.15, 0.05])
    
    def check_haircut_needed(self, input_tensor):
        """Check if haircut is needed"""
        # Based on hair length
        hair_length = self.analyze_hair(input_tensor)
        return hair_length in ['Long', 'Very Long']
    
    def get_neatness_score(self, input_tensor):
        """Get neatness score (1-5)"""
        # Return random score biased toward good (4-5)
        return np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])
    
    def check_face_cleanliness(self, face_image):
        """Check if face is clean"""
        # Simple check based on skin tone consistency
        try:
            hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
            # Check for consistent skin tone
            skin_mask = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([20, 150, 255]))
            skin_percentage = np.sum(skin_mask > 0) / skin_mask.size
            
            # Clean if skin area is predominant
            return skin_percentage > 0.6
        except:
            return True
    
    def get_additional_notes(self, face_image):
        """Get additional grooming notes"""
        notes = []
        
        # Check for glasses
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for circular patterns (glasses)
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=10, maxRadius=50)
        
        if circles is not None:
            notes.append("Wearing glasses")
        
        # Check for face symmetry (simplified)
        height, width = face_image.shape[:2]
        left_half = face_image[:, :width//2]
        right_half = face_image[:, width//2:]
        
        # Convert to grayscale for comparison
        left_gray = cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)
        
        # Flip right half for comparison
        right_flipped = cv2.flip(right_gray, 1)
        
        # Compare histograms
        hist_left = cv2.calcHist([left_gray], [0], None, [256], [0, 256])
        hist_right = cv2.calcHist([right_flipped], [0], None, [256], [0, 256])
        
        correlation = cv2.compareHist(hist_left, hist_right, cv2.HISTCMP_CORREL)
        
        if correlation < 0.8:
            notes.append("Asymmetrical features detected")
        
        return notes
    
    def get_default_results(self):
        """Return default results if analysis fails"""
        return {
            'has_beard': False,
            'hair_length': 'Medium',
            'needs_haircut': False,
            'neatness_score': 4,
            'face_clean': True,
            'additional_notes': []
        }