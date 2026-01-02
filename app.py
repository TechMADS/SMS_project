from email.mime import image
from flask import Flask, render_template, request, jsonify, Response

import os
import cv2
import numpy as np
import sqlite3
from datetime import datetime, date
import json
import time
import threading
import base64
import hashlib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = 'student-monitoring-system-2024'

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('database', exist_ok=True)
os.makedirs('utils', exist_ok=True)

# Global camera variables
camera = None
camera_active = False
camera_lock = threading.Lock()

# Initialize database with proper schema
def init_db():
    conn = sqlite3.connect('database/students.db')
    c = conn.cursor()
    
    # Students table - UPDATED SCHEMA
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE,
            name TEXT,
            class TEXT,
            roll_number INTEGER DEFAULT 0,
            gender TEXT DEFAULT 'Unknown',
            contact TEXT,
            department TEXT,
            year TEXT,
            registered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Check if roll_number column exists, if not add it
    try:
        c.execute("SELECT roll_number FROM students LIMIT 1")
    except sqlite3.OperationalError:
        print("Adding roll_number column to students table...")
        c.execute('ALTER TABLE students ADD COLUMN roll_number INTEGER DEFAULT 0')
    
    # Attendance table
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            date DATE,
            check_in TIME,
            grooming_status TEXT,
            uniform_status TEXT,
            violations TEXT
        )
    ''')
    
    # Add sample students if table is empty
    c.execute("SELECT COUNT(*) FROM students")
    if c.fetchone()[0] == 0:
        sample_students = [
            ('S001', 'John Doe', 'II - ECE', 1),
            ('S002', 'Jane Smith', 'I - ECE', 2),
            ('S003', 'Robert Johnson', 'IV - ECE', 1),
            ('S004', 'Emily Davis', 'III - ECE', 3),
            ('S005', 'Michael Wilson', 'II - ECE', 5),
        ]
        
        for student in sample_students:
            c.execute('''
                INSERT INTO students (student_id, name, class, roll_number) 
                VALUES (?, ?, ?, ?)
            ''', student)
        
        print("✅ Added 5 sample students")
    
    # Add sample students with complete information if table is empty
    
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully!")

init_db()

# Import your AI model
try:
    from utils.model_loader import GroomingModel
    ai_model = GroomingModel('grooming_model_with_gender.pth')
    print("✅ AI Model Loaded Successfully")
except Exception as e:
    print(f"⚠️  AI Model Loading Error: {e}")
    ai_model = None

# Simple face detector using OpenCV
class SimpleFaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.known_faces = {}
    
    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def identify_student(self, face_image):
        # Simple identification - returns "Unknown" for now
        return "Unknown"
    
    def detect_gender(self, face_image):
        """Simple gender detection based on facial features"""
        try:
            if face_image.size == 0:
                return "Unknown"
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Get face dimensions
            height, width = gray.shape
            
            # Calculate face aspect ratio
            aspect_ratio = width / height if height > 0 else 1
            
            # Simple rule: wider faces are often male, narrower faces often female
            if aspect_ratio > 0.85:
                return "Male"
            else:
                return "Female"
                
        except Exception as e:
            print(f"Gender detection error: {e}")
            return "Unknown"

# Uniform and grooming analyzer with gender awareness
class UniformGroomingAnalyzer:
    def __init__(self):
        self.ai_model = ai_model
    
    def analyze(self, image, face_region, gender="Unknown"):
        x, y, w, h = face_region
        image_height, image_width = image.shape[:2]
        
        # Initialize results with default values
        results = {
            'gender': gender,
            'beard': False,
            'ID': 'Not Visible',
            'shoes': 'Not Visible',
            'hair_neat': True,
            'uniform_proper': True
        }
        
        try:
            face_roi = image[y:y+h, x:x+w]
            
            # === GET AI MODEL PREDICTION ===
            ai_prediction = {}
            if self.ai_model and face_roi.size > 0:
                try:
                    ai_prediction = self.ai_model.predict(face_roi)
                    # Update gender from AI if available
                    if 'gender' in ai_prediction and ai_prediction['gender'] != 'unknown':
                        results['gender'] = ai_prediction['gender']
                        gender = ai_prediction['gender']  # Update gender for rule-based checks
                except Exception as e:
                    print(f"AI prediction error: {e}")
            
            # === BEARD DETECTION (Rule-based + AI) ===
            if gender == "Male":
                if face_roi.size > 0:
                    # Rule-based beard detection
                    lower_face = face_roi[int(h*0.6):, :]
                    if lower_face.size > 0:
                        gray_lower = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
                        dark_pixels = np.sum(gray_lower < 50) / gray_lower.size
                        rule_beard = bool(dark_pixels > 0.3)
                    
                    # AI beard detection if available
                    ai_beard = False
                    if 'facial_hair' in ai_prediction:
                        ai_beard = ai_prediction['facial_hair'] == 'bearded'
                    
                    # Combine both detections (AI has priority if available)
                    if 'facial_hair' in ai_prediction:
                        results['beard'] = ai_beard
                    else:
                        results['beard'] = rule_beard
            else:
                # For females or unknown, no beard check
                results['beard'] = False
            
            # === ID CARD DETECTION (Rule-based only) ===
            neck_y = min(y + h, image_height - 1)
            neck_h = min(50, image_height - neck_y)
            
            if neck_y + neck_h < image_height and neck_h > 10:
                neck_region = image[neck_y:neck_y+neck_h, x:x+w]
                if neck_region.size > 0:
                    hsv = cv2.cvtColor(neck_region, cv2.COLOR_BGR2HSV)
                    red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
                    blue_mask = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))
                    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
                    
                    ID_pixels = np.sum(red_mask > 0) + np.sum(blue_mask > 0) + np.sum(white_mask > 0)
                    total_pixels = neck_region.shape[0] * neck_region.shape[1]
                    
                    if ID_pixels / total_pixels < 0.1:
                        results['ID'] = 'Missing'
                    else:
                        results['ID'] = 'Present'
                else:
                    results['ID'] = 'Not Visible'
            else:
                results['ID'] = 'Not Visible'
                
            # === FOOTWEAR DETECTION (Improved with better heuristics) ===
            def detect_footwear(image, face_region, ai_prediction):
                """
                Improved footwear detection that:
                1. Checks if full body is likely in the image
                2. Uses AI model when available
                3. Only says 'Not Visible' when feet are definitely not in frame
                4. Has better fallback logic
                """
                x, y, w, h = face_region
                image_height, image_width = image.shape[:2]
        
                # Calculate if feet are likely in the image
                # Basic assumption: average human height is ~7.5 heads tall
                estimated_full_height = h * 7.5
                bottom_of_body = y + estimated_full_height
        
                # Check 1: Is the full body likely in the image?
                is_full_body_visible = bottom_of_body <= image_height * 1.1  # 10% margin
            
                if not is_full_body_visible:
                # If full body is definitely not visible
                    return 'Not Visible'
            
                # Check 2: Try AI detection first (most accurate)
                if 'footwear' in ai_prediction:
                    ai_footwear = ai_prediction['footwear']
                # Map AI footwear to our categories
                if ai_footwear == 'shoes':
                    return 'Shoes'
                elif ai_footwear in ['slippers', 'sandals']:
                    return 'Slippers'
                elif ai_footwear in ['boots', 'loafers', 'oxfords']:
                    return 'Shoes'  # Map various shoe types to 'Shoes'
            
                # Check 3: Try rule-based detection
                # Estimate foot region (below face, centered)
                foot_y = min(y + int(h * 1.5), image_height - 50)  # Start below face
                foot_h = min(150, image_height - foot_y)  # Increased height for better detection
            
                if foot_y >= image_height - 20 or foot_h < 30:
                    return 'Unknown'  # Feet might not be visible, but don't say "Not Visible"
            
                foot_region = image[foot_y:foot_y+foot_h, max(0, x-w//2):min(image_width, x+w*2)]
            
                if foot_region.size == 0:
                    return 'Unknown'
            
                # Improved rule-based detection
                try:
                    # Convert to different color spaces for better detection
                    hsv = cv2.cvtColor(foot_region, cv2.COLOR_BGR2HSV)
                    gray = cv2.cvtColor(foot_region, cv2.COLOR_BGR2GRAY)
                
                    # Detect skin (slippers/sandals show more skin)
                    skin_mask = cv2.inRange(hsv, (0, 20, 70), (20, 255, 255))
                    skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
                
                    # Detect dark colors (shoes are often dark)
                    dark_mask = cv2.inRange(gray, 0, 50)
                    dark_ratio = np.sum(dark_mask > 0) / dark_mask.size
                
                    # Detect bright/white colors (some shoes)
                    bright_mask = cv2.inRange(gray, 200, 255)
                    bright_ratio = np.sum(bright_mask > 0) / bright_mask.size
                
                    # Improved decision logic
                    if skin_ratio > 0.25:
                        return 'Slippers'
                    elif dark_ratio > 0.2 or bright_ratio > 0.2:
                    # Has significant dark or bright areas - likely shoes
                        return 'Shoes'
                    else:
                        return 'Unknown'

                except Exception as e:
                    print(f"Footwear detection error: {e}")
                    return 'Unknown'


                
    

            
            # === SHOES DETECTION (Rule-based + AI) ===
            # face_position_ratio = y / image_height if image_height > 0 else 0
            
            # if face_position_ratio < 0.6:
            #     feet_y = min(y + int(h * 1.8), image_height - 1)
            #     feet_h = min(100, image_height - feet_y)
                
            #     if feet_y < image_height and feet_h > 10:
            #         feet_region = image[feet_y:feet_y+feet_h, x:x+w]
            #         if feet_region.size > 0:
            #             # Rule-based shoes detection
            #             hsv_feet = cv2.cvtColor(feet_region, cv2.COLOR_BGR2HSV)
            #             skin_mask = cv2.inRange(hsv_feet, (0, 20, 70), (20, 255, 255))
            #             skin_ratio = np.sum(skin_mask > 0) / skin_mask.size if skin_mask.size > 0 else 0
                        
            #             rule_shoes = 'Unknown'
            #             if skin_ratio > 0.3:
            #                 rule_shoes = 'Slippers'
            #             else:
            #                 gray_feet = cv2.cvtColor(feet_region, cv2.COLOR_BGR2GRAY)
            #                 dark_ratio = np.sum(gray_feet < 50) / gray_feet.size if gray_feet.size > 0 else 0
            #                 if dark_ratio > 0.3:
            #                     rule_shoes = 'Shoes'
                        
            #             # AI footwear detection if available
            #             ai_footwear = 'Unknown'
            #             if 'footwear' in ai_prediction:
            #                 ai_footwear = ai_prediction['footwear']
            #                 # Map AI footwear to our categories
            #                 if ai_footwear == 'shoes':
            #                     ai_footwear = 'Shoes'
            #                 elif ai_footwear in ['slippers', 'sandals']:
            #                     ai_footwear = 'Slippers'
                        
            #             # Combine detections (AI has priority)
            #             if 'footwear' in ai_prediction:
            #                 results['shoes'] = ai_footwear
            #             else:
            #                 results['shoes'] = rule_shoes
            #         else:
            #             results['shoes'] = 'Not Visible'
            #     else:
            #         results['shoes'] = 'Not Visible'
            # else:
            #     results['shoes'] = 'Not Visible'
            
            # === SHOES DETECTION (Improved) ===
            results['shoes'] = detect_footwear(image, (x, y, w, h), ai_prediction)
            
            # === UNIFORM DETECTION (Rule-based + AI) ===
            # Rule-based uniform detection (simplified)
            if face_roi.size > 0:
                # Check for uniform colors (example: blue/white for school uniform)
                hsv_upper = cv2.cvtColor(face_roi[:int(h/2), :], cv2.COLOR_BGR2HSV)
                blue_mask = cv2.inRange(hsv_upper, (100, 50, 50), (130, 255, 255))
                white_mask = cv2.inRange(hsv_upper, (0, 0, 200), (180, 30, 255))
                
                uniform_pixels = np.sum(blue_mask > 0) + np.sum(white_mask > 0)
                total_pixels = face_roi.shape[0] * face_roi.shape[1]
                
                rule_uniform = uniform_pixels / total_pixels > 0.2
            
            # AI uniform detection if available
            ai_uniform = True
            if 'uniform' in ai_prediction:
                ai_uniform = ai_prediction['uniform'] == 'proper_uniform'
            
            # Combine detections (AI has priority)
            if 'uniform' in ai_prediction:
                results['uniform_proper'] = ai_uniform
            else:
                results['uniform_proper'] = rule_uniform
            
            # Store AI prediction for later use
            results['ai_prediction'] = ai_prediction
            
            return results
            
        except Exception as e:
            print(f"Analysis error: {e}")
            return results

# Initialize detectors
face_detector = SimpleFaceDetector()
analyzer = UniformGroomingAnalyzer()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Save file
        filename = f"upload_{int(time.time())}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        result = process_image(filepath, is_upload=True)
        
        return jsonify(json_serializable(result))
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def json_serializable(obj):
    """Convert non-serializable objects to serializable format"""
    if isinstance(obj, dict):
        return {k: json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serializable(item) for item in obj]
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def process_image(image_path, is_upload=False):
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'success': False, 'error': 'Could not load image'}
        
        # Resize for consistency
        image = cv2.resize(image, (640, 480))
        result_image = image.copy()
        
        # Detect faces
        faces = face_detector.detect(image)
        
        results = []
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = image[y:y+h, x:x+w]
            
            # Detect gender
            gender = face_detector.detect_gender(face_roi)
            
            # Analyze grooming and uniform with gender awareness
            analysis = analyzer.analyze(image, (x, y, w, h), gender)
            
            # Get AI model prediction if available
            ai_result = analysis.get('ai_prediction', {})
            
            # Identify student
            student_id = face_detector.identify_student(face_roi)
            
            # Detect violations with gender awareness - COMBINED ANALYSIS
            violations = []
            
            # 1. BEARD - Only for males (Rule + AI combined)
            if gender == "Male":
                if analysis['beard']:
                    violations.append('Beard detected')
            
            # 2. ID CARD - Check only if visible (Rule-based only)
            if analysis['ID'] == 'Missing':
                violations.append('No ID card')
            
            # 3. SHOES - Check only if visible and improper (Rule + AI combined)
            if analysis['shoes'] == 'Slippers':
                violations.append('Slippers/Improper footwear')
            elif analysis['shoes'] == 'Unknown':
                violations.append('Check footwear')
            
            # 4. UNIFORM - For all genders (Rule + AI combined)
            if not analysis['uniform_proper']:
                # violations.append('proper uniform')

                violations.append('Improper uniform')
            
            # 5. HAIR - For all genders (from AI if available)
            if 'facial_hair' in ai_result and ai_result['facial_hair'] != 'clean_shaven' and gender == "Male":
                violations.append('Facial hair not properly groomed')
            
            # Create result with AI integration
            result = {
                'student_id': str(student_id),
                'gender': str(gender),
                'face_location': {
                    'x': int(x),
                    'y': int(y), 
                    'w': int(w),
                    'h': int(h)
                },
                'analysis': {
                    'gender': str(analysis['gender']),
                    'beard': bool(analysis['beard']),
                    'ID': str(analysis['ID']),
                    'shoes': str(analysis['shoes']),
                    'hair_neat': bool(not analysis['beard'] if gender == "Male" else True),
                    'uniform_proper': bool(analysis['uniform_proper'])
                },
                'ai_analysis': ai_result,  # Full AI prediction results
                'combined_result': {
                    'detection_method': 'AI + Rule-based Hybrid',
                    'confidence': ai_result.get('compliance_score', 0.5) if ai_result else 0.5,
                    'ai_available': bool(ai_result)
                },
                'grooming_result': {  # For compatibility with frontend
                    'gender': str(analysis['gender']),
                    'beard': bool(analysis['beard']),
                    'ID': str(analysis['ID']),
                    'shoes': str(analysis['shoes']),
                    'uniform': 'Proper' if analysis['uniform_proper'] else 'Improper',
                    'compliance_score': ai_result.get('compliance_score', 0.8) if ai_result else 0.8,
                    'label': 'Good' if len(violations) == 0 else 'Needs Attention',
                    'confidence': 0.9 if ai_result else 0.6
                },
                'violations': [str(v) for v in violations],
                'has_issues': bool(len(violations) > 0)
            }
            results.append(result)
            
            # Draw on image with AI status
            color = (0, 255, 0) if not result['has_issues'] else (0, 0, 255)
            cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
            
            # Add text
            status = "GOOD" if not result['has_issues'] else "NEEDS ATTENTION"
            cv2.putText(result_image, f"ID: {student_id}", (x, y-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(result_image, f"Gender: {gender}", (x, y-60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(result_image, f"AI: {'✓' if ai_result else '✗'}", (x, y-80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(result_image, f"Status: {status}", (x, y-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Add violation details
            if violations:
                for i, violation in enumerate(violations[:3]):
                    cv2.putText(result_image, f"- {violation}", (x, y+h+20+i*20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            else:
                cv2.putText(result_image, "✓ All Good", (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Save result image
        result_filename = f"result_{int(time.time())}.jpg"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, result_image)
        
        return {
            'success': True,
            'faces_detected': int(len(faces)),
            'results': results,
            'result_image': f'static/uploads/{result_filename}',
            'original_image': f'static/uploads/{os.path.basename(image_path)}'
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global camera, camera_active
    
    with camera_lock:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while camera_active:
            success, frame = camera.read()
            if not success:
                break
            
            # Process frame
            processed_frame = process_live_frame(frame)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.03)
        
        camera.release()
        camera = None

def process_live_frame(frame):
    try:
        # Resize frame
        frame = cv2.resize(frame, (640, 480))
        
        # Detect faces
        faces = face_detector.detect(frame)
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Detect gender
            gender = face_detector.detect_gender(face_roi)
            
            # Quick analysis
            analysis = analyzer.analyze(frame, (x, y, w, h), gender)
            ai_result = analysis.get('ai_prediction', {})
            
            # Determine color
            color = (0, 255, 0)  # Green
            issues = []
            info = []
            
            # Add gender info
            info.append(f"Gender: {gender}")
            
            # Add AI status
            if ai_result:
                info.append(f"AI: Active")
            else:
                info.append(f"AI: Offline")
            
            # Gender-specific checks
            if gender == "Male":
                if analysis['beard']:
                    issues.append('BEARD')
                    color = (0, 165, 255)  # Orange
            
            # Common checks
            if analysis['ID'] == 'Missing':
                issues.append('NO ID')
                color = (0, 0, 255)  # Red
            elif analysis['ID'] == 'Not Visible':
                info.append('ID: Not Visible')
            
            if analysis['shoes'] == 'Slippers':
                issues.append('SLIPPERS')
                color = (0, 0, 255)  # Red
            elif analysis['shoes'] == 'Not Visible':
                info.append('Shoes: Not Visible')
            elif analysis['shoes'] == 'Unknown':
                issues.append('CHECK SHOES')
                color = (0, 165, 255)  # Orange
            
            if not analysis['uniform_proper']:
                issues.append('UNIFORM')
                color = (0, 0, 255)  # Red
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Display info
            y_offset = 80
            cv2.putText(frame, f"ID: Unknown", (x, y-y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Show gender with color
            gender_color = (255, 0, 0) if gender == "Male" else (255, 0, 255)
            cv2.putText(frame, f"Gender: {gender}", (x, y-y_offset+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, gender_color, 1)
            
            # Show AI status
            ai_color = (0, 255, 255) if ai_result else (200, 200, 200)
            ai_status = "✓ AI Active" if ai_result else "✗ AI Offline"
            cv2.putText(frame, ai_status, (x, y-y_offset+40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, ai_color, 1)
            
            # Show additional info
            for i, item in enumerate(info[:2]):
                cv2.putText(frame, item, (x, y-y_offset+60+i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            if issues:
                cv2.putText(frame, "Issues:", (x, y+h+10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                for i, issue in enumerate(issues[:2]):
                    cv2.putText(frame, f"- {issue}", (x, y+h+30+i*20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            else:
                cv2.putText(frame, "All Good ✓", (x, y+h+10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
        
    except Exception as e:
        print(f"Live frame error: {e}")
        return frame

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    global camera_active
    camera_active = True
    return jsonify({'success': True, 'message': 'Camera started'})

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    global camera_active
    camera_active = False
    time.sleep(0.5)
    return jsonify({'success': True, 'message': 'Camera stopped'})

@app.route('/api/camera/capture', methods=['POST'])
def capture_frame():
    global camera
    try:
        with camera_lock:
            if camera and camera_active:
                success, frame = camera.read()
                if success:
                    # Save frame
                    filename = f"capture_{int(time.time())}.jpg"
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    cv2.imwrite(filepath, frame)
                    
                    # Process it
                    result = process_image(filepath, is_upload=True)
                    return jsonify(json_serializable(result))
        
        return jsonify({'success': False, 'error': 'Camera not active'}), 400
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/students', methods=['GET'])
def get_students():
    try:
        conn = sqlite3.connect('database/students.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute('SELECT * FROM students ORDER BY class, roll_number')
        students = [dict(row) for row in c.fetchall()]
        
        conn.close()
        return jsonify({'success': True, 'students': students})
        
    except Exception as e:
        print(f"Error getting students: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/student/add', methods=['POST'])
def add_student():
    try:
        data = request.json
        required = ['student_id', 'name', 'class']
        
        for field in required:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing {field}'}), 400
        
        conn = sqlite3.connect('database/students.db')
        c = conn.cursor()
        
        try:
            c.execute('''
                INSERT INTO students (student_id, name, class, roll_number)
                VALUES (?, ?, ?, ?)
            ''', (
                data['student_id'],
                data['name'],
                data['class'],
                data.get('roll_number', 0)
            ))
            
            conn.commit()
            return jsonify({'success': True, 'message': 'Student added successfully'})
            
        except sqlite3.IntegrityError:
            return jsonify({'success': False, 'error': 'Student ID already exists'}), 400
        except Exception as e:
            print(f"Database error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
        finally:
            conn.close()
            
    except Exception as e:
        print(f"Add student error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/student/delete/<student_id>', methods=['DELETE'])
def delete_student(student_id):
    try:
        conn = sqlite3.connect('database/students.db')
        c = conn.cursor()
        
        c.execute('DELETE FROM students WHERE student_id = ?', (student_id,))
        conn.commit()
        
        if c.rowcount > 0:
            return jsonify({'success': True, 'message': 'Student deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Student not found'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/attendance/today', methods=['GET'])
def get_today_attendance():
    try:
        today = date.today()
        conn = sqlite3.connect('database/students.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute('''
            SELECT s.*, a.check_in, a.grooming_status, a.uniform_status
            FROM students s
            LEFT JOIN attendance a ON s.student_id = a.student_id AND a.date = ?
            ORDER BY s.class, s.roll_number
        ''', (today,))
        
        rows = c.fetchall()
        attendance = []
        
        for row in rows:
            record = dict(row)
            if record['check_in'] is None:
                record['status'] = 'Absent'
            else:
                record['status'] = 'Present'
            attendance.append(record)
        
        conn.close()
        return jsonify({'success': True, 'attendance': attendance, 'date': str(today)})
        
    except Exception as e:
        print(f"Attendance error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/system/status', methods=['GET'])
def system_status():
    # Check camera
    camera_available = False
    try:
        cap = cv2.VideoCapture(0)
        camera_available = cap.isOpened()
        cap.release()
    except:
        pass
    
    return jsonify({
        'success': True,
        'camera_available': bool(camera_available),
        'model_loaded': bool(ai_model is not None),
        'gender_detection': True,
        'analysis_type': 'Hybrid (AI + Rule-based)',
        'system_time': datetime.now().isoformat()
    })

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get information about the AI model"""
    if ai_model:
        return jsonify({
            'success': True,
            'loaded': True,
            'model_name': 'Grooming Model with Gender',
            'architecture': 'MobileNetV2',
            'tasks': ['gender', 'facial_hair', 'footwear', 'uniform', 'compliance'],
            'device': str(ai_model.device)
        })
    else:
        return jsonify({
            'success': False,
            'loaded': False,
            'error': 'AI Model not loaded - using rule-based analysis'
        })

# Fix database schema route
@app.route('/api/fix_database', methods=['POST'])
def fix_database():
    """Fix database schema issues"""
    try:
        conn = sqlite3.connect('database/students.db')
        c = conn.cursor()
        
        # Check and add missing columns
        c.execute("PRAGMA table_info(students)")
        columns = [col[1] for col in c.fetchall()]
        
        if 'roll_number' not in columns:
            print("Adding roll_number column...")
            c.execute('ALTER TABLE students ADD COLUMN roll_number INTEGER DEFAULT 0')
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Database schema fixed'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎓 STUDENT MONITORING SYSTEM WITH HYBRID AI")
    print("="*60)
    print(f"📅 Server starting at: http://127.0.0.1:5000")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"💾 Database: database/students.db")
    print(f"🤖 AI Model: {'✅ LOADED' if ai_model else '❌ NOT LOADED'}")
    print(f"🔧 Analysis: Hybrid (AI + Rule-based)")
    print("="*60)
    print("✅ All systems ready!")
    print("="*60)
    print("\nKEY FEATURES:")
    print("-"*40)
    print("1. 🤖 HYBRID ANALYSIS SYSTEM")
    print("   - AI Model: MobileNetV2 with Gender")
    print("   - Rule-based: Fallback when AI unavailable")
    print("   - Combined results for maximum accuracy")
    print("2. 👨‍🎓👩‍🎓 Gender-Aware Detection")
    print("   - Beard checks only for males")
    print("   - Different rules for each gender")
    print("   - Smart visibility checks")
    print("3. 📊 Complete Student Management")
    print("   - Student database with roll numbers")
    print("   - Attendance tracking")
    print("   - Live camera monitoring")
    print("-"*40)
    
    # Fix database on startup
    try:
        conn = sqlite3.connect('database/students.db')
        c = conn.cursor()
        c.execute("PRAGMA table_info(students)")
        columns = [col[1] for col in c.fetchall()]
        if 'roll_number' not in columns:
            print("⚠️  Fixing database schema...")
            c.execute('ALTER TABLE students ADD COLUMN roll_number INTEGER DEFAULT 0')
            conn.commit()
            print("✅ Database schema fixed!")
        conn.close()
    except Exception as e:
        print(f"⚠️  Database check error: {e}")
    
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)


# # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
# from flask import Flask, render_template, request, jsonify, Response
# import os
# import cv2
# import numpy as np
# import sqlite3
# from datetime import datetime, date
# import json
# import time
# import threading
# import base64
# import hashlib

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['SECRET_KEY'] = 'student-monitoring-system-2024'

# # Create directories
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# os.makedirs('database', exist_ok=True)
# os.makedirs('utils', exist_ok=True)

# # Global camera variables
# camera = None
# camera_active = False
# camera_lock = threading.Lock()

# # Initialize database with proper schema
# def init_db():
#     conn = sqlite3.connect('database/students.db')
#     c = conn.cursor()
    
#     # Students table - UPDATED SCHEMA
#     c.execute('''
#         CREATE TABLE IF NOT EXISTS students (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             student_id TEXT UNIQUE,
#             name TEXT,
#             class TEXT,
#             roll_number INTEGER DEFAULT 0,
#             registered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         )
#     ''')
    
#     # Check if roll_number column exists, if not add it
#     try:
#         c.execute("SELECT roll_number FROM students LIMIT 1")
#     except sqlite3.OperationalError:
#         print("Adding roll_number column to students table...")
#         c.execute('ALTER TABLE students ADD COLUMN roll_number INTEGER DEFAULT 0')
    
#     # Attendance table
#     c.execute('''
#         CREATE TABLE IF NOT EXISTS attendance (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             student_id TEXT,
#             date DATE,
#             check_in TIME,
#             grooming_status TEXT,
#             uniform_status TEXT,
#             violations TEXT
#         )
#     ''')
    
#     # Add sample students if table is empty
#     c.execute("SELECT COUNT(*) FROM students")
#     if c.fetchone()[0] == 0:
#         sample_students = [
#             ('S001', 'John Doe', '10', 1),
#             ('S002', 'Jane Smith', '10', 2),
#             ('S003', 'Robert Johnson', '11', 1),
#             ('S004', 'Emily Davis', '11', 3),
#             ('S005', 'Michael Wilson', '12', 5),
#         ]
        
#         for student in sample_students:
#             c.execute('''
#                 INSERT INTO students (student_id, name, class, roll_number) 
#                 VALUES (?, ?, ?, ?)
#             ''', student)
        
#         print("✅ Added 5 sample students")
    
#     conn.commit()
#     conn.close()
#     print("✅ Database initialized successfully!")

# init_db()

# # Import your custom model
# try:
#     from utils.model_loader import GroomingModel
#     grooming_model = GroomingModel('grooming_model_final (1).pth')
#     print("✅ Grooming Model Loaded Successfully")
# except Exception as e:
#     print(f"⚠️  Model Loading Error: {e}")
#     grooming_model = None

# # Import and setup Hugging Face Inference API
# # try:
# #     from huggingface_hub import InferenceClient
# #     # Get your free token from: https://huggingface.co/settings/tokens
# #     HF_TOKEN = "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"  # REPLACE THIS WITH YOUR ACTUAL TOKEN
# #     hf_client = InferenceClient(token=HF_TOKEN)
# #     print("✅ Hugging Face Inference Client Initialized")
# # except Exception as e:
# #     print(f"⚠️  Hugging Face Client Error: {e}")
# #     hf_client = None

# # Simple face detector using OpenCV
# class SimpleFaceDetector:
#     def __init__(self):
#         self.face_cascade = cv2.CascadeClassifier(
#             cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
#         )
#         self.known_faces = {}
    
#     def detect(self, image):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         faces = self.face_cascade.detectMultiScale(
#             gray, 
#             scaleFactor=1.1, 
#             minNeighbors=5,
#             minSize=(30, 30)
#         )
#         return faces
    
#     def identify_student(self, face_image):
#         # Simple identification - returns "Unknown" for now
#         # You can implement face recognition here if needed
#         return "Unknown"
    
    

# # Uniform and grooming analyzer
# class UniformGroomingAnalyzer:
#     def analyze(self, image, face_region):
#         x, y, w, h = face_region
        
#         results = {
#             'beard': False,
#             'ID': True,
#             'shoes': True,
#             'hair_neat': True,
#             'uniform_proper': True
#         }
        
#         try:
#             # Analyze lower face for beard (dark pixels)
#             face_roi = image[y:y+h, x:x+w]
#             if face_roi.size > 0:
#                 lower_face = face_roi[int(h*0.6):, :]
#                 gray_lower = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
#                 dark_pixels = np.sum(gray_lower < 50) / gray_lower.size
#                 if dark_pixels > 0.3:
#                     results['beard'] = True
            
#             # Analyze neck region for ID
#             neck_y = min(y + h, image.shape[0] - 1)
#             neck_h = min(50, image.shape[0] - neck_y)
#             neck_region = image[neck_y:neck_y+neck_h, x:x+w]
#             if neck_region.size > 0:
#                 hsv = cv2.cvtColor(neck_region, cv2.COLOR_BGR2HSV)
#                 # Look for ID colors (red/blue)
#                 red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
#                 blue_mask = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))
#                 ID_pixels = np.sum(red_mask > 0) + np.sum(blue_mask > 0)
#                 if ID_pixels / (neck_region.shape[0] * neck_region.shape[1]) < 0.1:
#                     results['ID'] = False
            
#             # Analyze lower region for shoes/slippers
#             feet_y = min(y + int(h * 1.8), image.shape[0] - 1)
#             feet_h = min(100, image.shape[0] - feet_y)
#             if feet_y < image.shape[0] and feet_h > 0:
#                 feet_region = image[feet_y:feet_y+feet_h, x:x+w]
#                 if feet_region.size > 0:
#                     # Check for skin color (slippers)
#                     hsv_feet = cv2.cvtColor(feet_region, cv2.COLOR_BGR2HSV)
#                     skin_mask = cv2.inRange(hsv_feet, (0, 20, 70), (20, 255, 255))
#                     skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
#                     if skin_ratio > 0.3:
#                         results['shoes'] = False
            
#             return results
            
#         except Exception as e:
#             print(f"Analysis error: {e}")
#             return results

# # Initialize detectors
# face_detector = SimpleFaceDetector()
# analyzer = UniformGroomingAnalyzer()

# # Routes
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/api/upload', methods=['POST'])
# def upload_image():
#     try:
#         if 'file' not in request.files:
#             return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({'success': False, 'error': 'No file selected'}), 400
        
#         # Save file
#         filename = f"upload_{int(time.time())}.jpg"
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         # Process image
#         result = process_image(filepath, is_upload=True)
        
#         return jsonify(result)
        
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)}), 500

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# def generate_frames():
#     global camera, camera_active
    
#     with camera_lock:
#         camera = cv2.VideoCapture(0)
#         camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
#         while camera_active:
#             success, frame = camera.read()
#             if not success:
#                 break
            
#             # Process frame
#             processed_frame = process_live_frame(frame)
            
#             # Encode frame
#             ret, buffer = cv2.imencode('.jpg', processed_frame)
#             frame_bytes = buffer.tobytes()
            
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
#             time.sleep(0.03)
        
#         camera.release()
#         camera = None

# def process_image(image_path, is_upload=False):
#     try:
#         # Load image
#         image = cv2.imread(image_path)
#         if image is None:
#             return {'success': False, 'error': 'Could not load image'}
        
#         # Resize for consistency
#         image = cv2.resize(image, (640, 480))
#         result_image = image.copy()
        
#         # Detect faces
#         faces = face_detector.detect(image)
        
#         results = []
        
#         for (x, y, w, h) in faces:
#             # Analyze grooming and uniform
#             analysis = analyzer.analyze(image, (x, y, w, h))
            
#             # Get grooming score from your model
#             grooming_result = {}
#             if grooming_model:
#                 face_roi = image[y:y+h, x:x+w]
#                 if face_roi.size > 0:
#                     grooming_result = grooming_model.predict(face_roi)
            
#             # Get analysis from Hugging Face API
#             # grooming_result = {}
#             # if hf_client:
#             #     try:
#             #         face_roi = image[y:y+h, x:x+w]
#             #         if face_roi.size > 0:
#             #             # Convert the face region to a format the API can use (base64)
#             #             _, buffer = cv2.imencode('.jpg', face_roi)
#             #             face_image_bytes = base64.b64encode(buffer).decode('utf-8')
                        
#             #             # Choose ONE of the following API calls based on what you need:
#             #             # Option A: Object Detection (finds items like 'person', 'tie', 'shoe')
#             #             api_result = hf_client.object_detection(
#             #                 image=face_image_bytes,
#             #                 model="facebook/detr-resnet-50"  # Example model
#             #             )
#             #             # Process `api_result` to check for specific items
                        
#             #             # Option B: Zero-Shot Image Classification (checks custom labels)
#             #             labels_to_check = ["neat hair", "clean shaven", "wearing a tie", "formal attire"]
#             #             api_result = hf_client.zero_shot_image_classification(
#             #                 image=face_image_bytes,
#             #                 candidate_labels=labels_to_check,
#             #                 model="openai/clip-vit-large-patch14-336"  # Example model
#             #             )
#             #             # `api_result` will be a list of scores for each label
                        
#             #             grooming_result = {"api_result": api_result}  # Store raw result for now
#             #     except Exception as api_error:
#             #         print(f"⚠️  API call failed: {api_error}")
#             #         grooming_result = {"error": str(api_error)}

#             # Identify student
#             student_id = face_detector.identify_student(image[y:y+h, x:x+w])
            
#             # Detect violations
#             violations = []
#             if analysis['beard']:
#                 violations.append('Beard detected')
#             if not analysis['ID']:
#                 violations.append('No ID')
#             if not analysis['shoes']:
#                 violations.append('Slippers/Wrong footwear')
#             if not analysis['hair_neat']:
#                 violations.append('Untidy hair')
#             if not analysis['uniform_proper']:
#                 violations.append('Improper uniform')
            
#             # Create result
#             result = {
#                 'student_id': student_id,
#                 'face_location': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
#                 'analysis': analysis,
#                 'grooming_result': grooming_result,
#                 'violations': violations,
#                 'has_issues': len(violations) > 0
#             }
#             results.append(result)
            
#             # Draw on image
#             color = (0, 255, 0) if not result['has_issues'] else (0, 0, 255)
#             cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
            
#             # Add text
#             status = "GOOD" if not result['has_issues'] else "NEEDS ATTENTION"
#             cv2.putText(result_image, f"ID: {student_id}", (x, y-40), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#             cv2.putText(result_image, f"Status: {status}", (x, y-20), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
#             # Add violation details
#             if violations:
#                 for i, violation in enumerate(violations[:3]):
#                     cv2.putText(result_image, f"- {violation}", (x, y+h+20+i*20), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
#         # Save result image
#         result_filename = f"result_{int(time.time())}.jpg"
#         result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
#         cv2.imwrite(result_path, result_image)
        
#         return {
#             'success': True,
#             'faces_detected': len(faces),
#             'results': results,
#             'result_image': f'static/uploads/{result_filename}',
#             'original_image': f'static/uploads/{os.path.basename(image_path)}'
#         }
        
#     except Exception as e:
#         return {'success': False, 'error': str(e)}

# def process_live_frame(frame):
#     try:
#         # Resize frame
#         frame = cv2.resize(frame, (640, 480))
        
#         # Detect faces
#         faces = face_detector.detect(frame)
        
#         for (x, y, w, h) in faces:
#             # Quick analysis
#             analysis = analyzer.analyze(frame, (x, y, w, h))
            
#             # Determine color
#             color = (0, 255, 0)  # Green
#             issues = []
            
#             if analysis['beard']:
#                 issues.append('BEARD')
#                 color = (0, 165, 255)  # Orange
#             if not analysis['ID']:
#                 issues.append('NO ID')
#                 color = (0, 0, 255)  # Red
#             if not analysis['shoes']:
#                 issues.append('SLIPPERS')
#                 color = (0, 0, 255)  # Red
            
#             # Draw rectangle
#             cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
#             # Display info
#             cv2.putText(frame, f"ID: Unknown", (x, y-60), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
#             if issues:
#                 cv2.putText(frame, "Issues:", (x, y-40), 
#                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#                 for i, issue in enumerate(issues[:2]):
#                     cv2.putText(frame, f"- {issue}", (x, y-20+i*20), 
#                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
#             else:
#                 cv2.putText(frame, "All Good ✓", (x, y-20), 
#                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
#         return frame
        
#     except Exception as e:
#         return frame

# @app.route('/api/camera/start', methods=['POST'])
# def start_camera():
#     global camera_active
#     camera_active = True
#     return jsonify({'success': True, 'message': 'Camera started'})

# @app.route('/api/camera/stop', methods=['POST'])
# def stop_camera():
#     global camera_active
#     camera_active = False
#     time.sleep(0.5)  # Allow last frame to be sent
#     return jsonify({'success': True, 'message': 'Camera stopped'})

# @app.route('/api/camera/capture', methods=['POST'])
# def capture_frame():
#     global camera
#     try:
#         with camera_lock:
#             if camera and camera_active:
#                 success, frame = camera.read()
#                 if success:
#                     # Save frame
#                     filename = f"capture_{int(time.time())}.jpg"
#                     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#                     cv2.imwrite(filepath, frame)
                    
#                     # Process it
#                     result = process_image(filepath, is_upload=True)
#                     return jsonify(result)
        
#         return jsonify({'success': False, 'error': 'Camera not active'}), 400
        
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)}), 500

# @app.route('/api/students', methods=['GET'])
# def get_students():
#     try:
#         conn = sqlite3.connect('database/students.db')
#         conn.row_factory = sqlite3.Row
#         c = conn.cursor()
        
#         c.execute('SELECT * FROM students ORDER BY class, roll_number')
#         students = [dict(row) for row in c.fetchall()]
        
#         conn.close()
#         return jsonify({'success': True, 'students': students})
        
#     except Exception as e:
#         print(f"Error getting students: {e}")
#         return jsonify({'success': False, 'error': str(e)}), 500

# @app.route('/api/student/add', methods=['POST'])
# def add_student():
#     try:
#         data = request.json
#         required = ['student_id', 'name', 'class']
        
#         for field in required:
#             if field not in data:
#                 return jsonify({'success': False, 'error': f'Missing {field}'}), 400
        
#         conn = sqlite3.connect('database/students.db')
#         c = conn.cursor()
        
#         try:
#             c.execute('''
#                 INSERT INTO students (student_id, name, class, roll_number)
#                 VALUES (?, ?, ?, ?)
#             ''', (
#                 data['student_id'],
#                 data['name'],
#                 data['class'],
#                 data.get('roll_number', 0)
#             ))
            
#             conn.commit()
#             return jsonify({'success': True, 'message': 'Student added successfully'})
            
#         except sqlite3.IntegrityError:
#             return jsonify({'success': False, 'error': 'Student ID already exists'}), 400
#         except Exception as e:
#             print(f"Database error: {e}")
#             return jsonify({'success': False, 'error': str(e)}), 500
#         finally:
#             conn.close()
            
#     except Exception as e:
#         print(f"Add student error: {e}")
#         return jsonify({'success': False, 'error': str(e)}), 500

# @app.route('/api/student/delete/<student_id>', methods=['DELETE'])
# def delete_student(student_id):
#     try:
#         conn = sqlite3.connect('database/students.db')
#         c = conn.cursor()
        
#         c.execute('DELETE FROM students WHERE student_id = ?', (student_id,))
#         conn.commit()
        
#         if c.rowcount > 0:
#             return jsonify({'success': True, 'message': 'Student deleted successfully'})
#         else:
#             return jsonify({'success': False, 'error': 'Student not found'}), 404
            
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)}), 500
#     finally:
#         conn.close()

# @app.route('/api/attendance/today', methods=['GET'])
# def get_today_attendance():
#     try:
#         today = date.today()
#         conn = sqlite3.connect('database/students.db')
#         conn.row_factory = sqlite3.Row
#         c = conn.cursor()
        
#         c.execute('''
#             SELECT s.*, a.check_in, a.grooming_status, a.uniform_status
#             FROM students s
#             LEFT JOIN attendance a ON s.student_id = a.student_id AND a.date = ?
#             ORDER BY s.class, s.roll_number
#         ''', (today,))
        
#         rows = c.fetchall()
#         attendance = []
        
#         for row in rows:
#             record = dict(row)
#             if record['check_in'] is None:
#                 record['status'] = 'Absent'
#             else:
#                 record['status'] = 'Present'
#             attendance.append(record)
        
#         conn.close()
#         return jsonify({'success': True, 'attendance': attendance, 'date': str(today)})
        
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)}), 500

# @app.route('/api/system/status', methods=['GET'])
# def system_status():
#     # Check camera
#     camera_available = False
#     try:
#         cap = cv2.VideoCapture(0)
#         camera_available = cap.isOpened()
#         cap.release()
#     except:
#         pass
    
#     return jsonify({
#         'success': True,
#         'camera_available': camera_available,
#         'model_loaded': grooming_model is not None,
#         # 'huggingface_client_ready': hf_client is not None,
#         'system_time': datetime.now().isoformat()
#     })

# # Fix database schema route
# @app.route('/api/fix_database', methods=['POST'])
# def fix_database():
#     """Fix database schema issues"""
#     try:
#         conn = sqlite3.connect('database/students.db')
#         c = conn.cursor()
        
#         # Check and add missing columns
#         c.execute("PRAGMA table_info(students)")
#         columns = [col[1] for col in c.fetchall()]
        
#         if 'roll_number' not in columns:
#             print("Adding roll_number column...")
#             c.execute('ALTER TABLE students ADD COLUMN roll_number INTEGER DEFAULT 0')
        
#         conn.commit()
#         conn.close()
        
#         return jsonify({'success': True, 'message': 'Database schema fixed'})
        
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)}), 500

# if __name__ == '__main__':
#     print("\n" + "="*60)
#     print("🎓 STUDENT MONITORING SYSTEM")
#     print("="*60)
#     print(f"📅 Server starting at: http://127.0.0.1:5000")
#     print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
#     print(f"💾 Database: database/students.db")
#     print("="*60)
#     print("✅ All systems ready!")
#     print("="*60)
    
#     # Fix database on startup
#     try:
#         conn = sqlite3.connect('database/students.db')
#         c = conn.cursor()
#         c.execute("PRAGMA table_info(students)")
#         columns = [col[1] for col in c.fetchall()]
#         if 'roll_number' not in columns:
#             print("⚠️  Fixing database schema...")
#             c.execute('ALTER TABLE students ADD COLUMN roll_number INTEGER DEFAULT 0')
#             conn.commit()
#             print("✅ Database schema fixed!")
#         conn.close()
#     except Exception as e:
#         print(f"⚠️  Database check error: {e}")
    
#     app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)