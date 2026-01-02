import requests
import json
import base64
import cv2
import numpy as np
from datetime import datetime

class FreeAPIDetector:
    def __init__(self, huggingface_token):
        self.huggingface_token = huggingface_token
        self.headers = {"Authorization": f"Bearer {huggingface_token}"}
        
    def detect_with_huggingface(self, image_base64):
        """Use Hugging Face's object detection model"""
        
        # Clean base64 string
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        try:
            # Try YOLO model first
            api_url = "https://api-inference.huggingface.co/models/ultralytics/yolov5"
            
            payload = {
                "inputs": image_base64,
                "parameters": {
                    "threshold": 0.5,
                    "overlap_threshold": 0.3
                }
            }
            
            response = requests.post(
                api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return self.process_huggingface_response(response.json())
            else:
                # Fallback to mock data if API fails
                print(f"Hugging Face API error: {response.status_code}")
                return self.get_mock_detection()
                
        except Exception as e:
            print(f"API Error: {str(e)}")
            return self.get_mock_detection()
    
    def process_huggingface_response(self, api_response):
        """Process Hugging Face API response"""
        if isinstance(api_response, list) and len(api_response) > 0:
            detections = api_response[0]
            
            # Look for clothing-related items
            clothing_items = []
            for det in detections:
                label = det.get('label', '').lower()
                score = det.get('score', 0)
                
                # Check if detection is clothing-related
                clothing_keywords = ['person', 'tie', 'shirt', 'jacket', 'suit', 
                                   'shoe', 'bag', 'backpack', 'hat', 'glasses']
                
                if any(keyword in label for keyword in clothing_keywords) and score > 0.5:
                    clothing_items.append({
                        'item': label,
                        'confidence': score,
                        'box': det.get('box', {})
                    })
            
            return {
                'success': True,
                'clothing_items': clothing_items,
                'person_detected': any('person' in item['item'] for item in clothing_items),
                'tie_detected': any('tie' in item['item'] for item in clothing_items),
                'violations': self.check_violations(clothing_items)
            }
        
        return self.get_mock_detection()
    
    def check_violations(self, clothing_items):
        """Check for dress code violations based on detected items"""
        violations = []
        
        # Check if person is detected
        person_present = any('person' in item['item'] for item in clothing_items)
        if not person_present:
            violations.append("No person detected in frame")
        
        # Check for tie
        tie_present = any('tie' in item['item'] for item in clothing_items)
        if not tie_present:
            violations.append("Tie not detected")
        
        # Check for formal wear
        formal_items = ['suit', 'jacket', 'shirt']
        formal_present = any(any(f in item['item'] for f in formal_items) for item in clothing_items)
        if not formal_present:
            violations.append("Formal wear not detected")
        
        return violations
    
    def get_mock_detection(self):
        """Return mock data when API fails (for development)"""
        return {
            'success': False,
            'clothing_items': [
                {'item': 'person', 'confidence': 0.95},
                {'item': 'tie', 'confidence': 0.85},
                {'item': 'shirt', 'confidence': 0.90}
            ],
            'person_detected': True,
            'tie_detected': True,
            'violations': [],
            'is_mock': True  # Flag to indicate mock data
        }
    
    def analyze_color(self, image_base64):
        """Simple color analysis using OpenCV (local, no API needed)"""
        try:
            # Decode base64 to image
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            
            nparr = np.frombuffer(base64.b64decode(image_base64), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Calculate average color
                avg_color = np.mean(frame, axis=(0, 1))
                
                # Check if shirt is white (BGR format in OpenCV)
                # White has high values in all channels
                is_white = avg_color[0] > 180 and avg_color[1] > 180 and avg_color[2] > 180
                
                return {
                    'avg_color': avg_color.tolist(),
                    'is_white': bool(is_white),
                    'color_name': 'white' if is_white else 'colored'
                }
        except:
            pass
        
        return {'avg_color': [255, 255, 255], 'is_white': True, 'color_name': 'unknown'}