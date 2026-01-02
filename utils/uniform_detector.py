import cv2
import numpy as np

class UniformDetector:
    def __init__(self):
        # Define color ranges for school uniforms
        self.uniform_colors = {
            'white_shirt': {
                'lower': np.array([0, 0, 200]),
                'upper': np.array([180, 30, 255])
            },
            'blue_pants': {
                'lower': np.array([100, 150, 0]),
                'upper': np.array([140, 255, 255])
            },
            'black_shoes': {
                'lower': np.array([0, 0, 0]),
                'upper': np.array([180, 255, 50])
            },
            'gray_pants': {
                'lower': np.array([0, 0, 100]),
                'upper': np.array([180, 30, 200])
            },
            'red_tie': {
                'lower': np.array([0, 100, 100]),
                'upper': np.array([10, 255, 255])
            },
            'blue_tie': {
                'lower': np.array([100, 100, 100]),
                'upper': np.array([130, 255, 255])
            }
        }
        
        # Load shoe/slipper detection model (simplified)
        self.shoe_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lowerbody.xml')
        
        print("✅ Uniform Detector Ready")
    
    def check_uniform_detailed(self, image, face_rect):
        """Perform detailed uniform check"""
        x, y, w, h = face_rect
        
        results = {
            'has_shirt': False,
            'shirt_color': 'unknown',
            'has_pants': False,
            'pants_color': 'unknown',
            'has_shoes': False,
            'has_slippers': False,
            'has_tie': False,
            'tie_color': 'unknown',
            'shirt_tucked': None,
            'improper_dress': False,
            'score': 0,
            'issues': []
        }
        
        try:
            # Define regions for uniform parts
            height, width = image.shape[:2]
            
            # Shirt region (upper body below face)
            shirt_y_start = min(y + h, height - 1)
            shirt_y_end = min(shirt_y_start + int(h * 1.5), height - 1)
            shirt_region = image[shirt_y_start:shirt_y_end, x:x+w]
            
            # Pants region (lower body)
            pants_y_start = shirt_y_end
            pants_y_end = min(pants_y_start + int(h * 1.0), height - 1)
            pants_region = image[pants_y_start:pants_y_end, x:x+w]
            
            # Shoes/slippers region
            shoes_y_start = pants_y_end
            shoes_y_end = min(shoes_y_start + int(h * 0.5), height - 1)
            shoes_region = image[shoes_y_start:shoes_y_end, x:x+w]
            
            # Analyze each region
            if shirt_region.size > 0:
                shirt_results = self.analyze_shirt(shirt_region)
                results.update(shirt_results)
            
            if pants_region.size > 0:
                pants_results = self.analyze_pants(pants_region)
                results.update(pants_results)
            
            if shoes_region.size > 0:
                shoes_results = self.analyze_shoes(shoes_region)
                results.update(shoes_results)
            
            # Check for tie
            tie_results = self.check_tie(image, face_rect)
            results.update(tie_results)
            
            # Check if shirt is tucked
            results['shirt_tucked'] = self.check_shirt_tucked(image, face_rect)
            
            # Calculate score and issues
            results['score'] = self.calculate_uniform_score(results)
            results['issues'] = self.detect_uniform_issues(results)
            
            return results
            
        except Exception as e:
            print(f"Uniform check error: {e}")
            return results
    
    def quick_check(self, image, face_rect):
        """Quick uniform check for live view"""
        x, y, w, h = face_rect
        
        results = {
            'has_slippers': False,
            'no_tie': True,
            'improper_dress': False
        }
        
        try:
            # Quick shoe/slipper check
            height, width = image.shape[:2]
            shoes_y_start = min(y + h + int(h * 1.5), height - 1)
            shoes_y_end = min(shoes_y_start + int(h * 0.8), height - 1)
            
            if shoes_y_start < shoes_y_end:
                shoes_region = image[shoes_y_start:shoes_y_end, max(0, x-w//2):min(width, x+w+w//2)]
                
                # Convert to HSV
                hsv = cv2.cvtColor(shoes_region, cv2.COLOR_BGR2HSV)
                
                # Look for skin color (slippers)
                lower_skin = np.array([0, 20, 70])
                upper_skin = np.array([20, 255, 255])
                skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
                
                skin_percentage = np.sum(skin_mask > 0) / skin_mask.size
                
                # Look for dark colors (shoes)
                lower_dark = np.array([0, 0, 0])
                upper_dark = np.array([180, 255, 50])
                dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
                
                dark_percentage = np.sum(dark_mask > 0) / dark_mask.size
                
                results['has_slippers'] = skin_percentage > 0.3 and dark_percentage < 0.2
            
            # Quick tie check (look for colored region near neck)
            tie_y_start = max(0, y + h//2)
            tie_y_end = min(height, y + h)
            tie_x_start = max(0, x + w//4)
            tie_x_end = min(width, x + 3*w//4)
            
            if tie_y_start < tie_y_end and tie_x_start < tie_x_end:
                tie_region = image[tie_y_start:tie_y_end, tie_x_start:tie_x_end]
                
                hsv = cv2.cvtColor(tie_region, cv2.COLOR_BGR2HSV)
                
                # Look for tie colors (red or blue)
                lower_red1 = np.array([0, 100, 100])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 100, 100])
                upper_red2 = np.array([180, 255, 255])
                lower_blue = np.array([100, 100, 100])
                upper_blue = np.array([130, 255, 255])
                
                red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
                
                red_mask = cv2.bitwise_or(red_mask1, red_mask2)
                tie_mask = cv2.bitwise_or(red_mask, blue_mask)
                
                tie_percentage = np.sum(tie_mask > 0) / tie_mask.size
                results['no_tie'] = tie_percentage < 0.1
            
            return results
            
        except Exception as e:
            return results
    
    def analyze_shirt(self, shirt_region):
        """Analyze shirt color and style"""
        results = {
            'has_shirt': False,
            'shirt_color': 'unknown',
            'shirt_style': 'unknown'
        }
        
        if shirt_region.size == 0:
            return results
        
        # Convert to HSV
        hsv = cv2.cvtColor(shirt_region, cv2.COLOR_BGR2HSV)
        
        # Check for white shirt
        white_mask = cv2.inRange(hsv, 
                               self.uniform_colors['white_shirt']['lower'],
                               self.uniform_colors['white_shirt']['upper'])
        white_percentage = np.sum(white_mask > 0) / white_mask.size
        
        if white_percentage > 0.3:
            results['has_shirt'] = True
            results['shirt_color'] = 'white'
            results['shirt_style'] = 'school_shirt'
        
        return results
    
    def analyze_pants(self, pants_region):
        """Analyze pants color"""
        results = {
            'has_pants': False,
            'pants_color': 'unknown'
        }
        
        if pants_region.size == 0:
            return results
        
        # Convert to HSV
        hsv = cv2.cvtColor(pants_region, cv2.COLOR_BGR2HSV)
        
        # Check for blue pants
        blue_mask = cv2.inRange(hsv,
                              self.uniform_colors['blue_pants']['lower'],
                              self.uniform_colors['blue_pants']['upper'])
        blue_percentage = np.sum(blue_mask > 0) / blue_mask.size
        
        # Check for gray pants
        gray_mask = cv2.inRange(hsv,
                              self.uniform_colors['gray_pants']['lower'],
                              self.uniform_colors['gray_pants']['upper'])
        gray_percentage = np.sum(gray_mask > 0) / gray_mask.size
        
        if blue_percentage > 0.3:
            results['has_pants'] = True
            results['pants_color'] = 'blue'
        elif gray_percentage > 0.3:
            results['has_pants'] = True
            results['pants_color'] = 'gray'
        
        return results
    
    def analyze_shoes(self, shoes_region):
        """Analyze shoes vs slippers"""
        results = {
            'has_shoes': False,
            'has_slippers': False
        }
        
        if shoes_region.size == 0:
            return results
        
        # Convert to HSV
        hsv = cv2.cvtColor(shoes_region, cv2.COLOR_BGR2HSV)
        
        # Look for black shoes
        black_mask = cv2.inRange(hsv,
                               self.uniform_colors['black_shoes']['lower'],
                               self.uniform_colors['black_shoes']['upper'])
        black_percentage = np.sum(black_mask > 0) / black_mask.size
        
        # Look for skin color (slippers)
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_percentage = np.sum(skin_mask > 0) / skin_mask.size
        
        if black_percentage > 0.4:
            results['has_shoes'] = True
        elif skin_percentage > 0.3:
            results['has_slippers'] = True
        
        return results
    
    def check_tie(self, image, face_rect):
        """Check if tie is present"""
        x, y, w, h = face_rect
        
        results = {
            'has_tie': False,
            'tie_color': 'unknown'
        }
        
        # Define neck/tie region
        tie_y_start = max(0, y + h//2)
        tie_y_end = min(image.shape[0], y + h)
        tie_x_start = max(0, x + w//4)
        tie_x_end = min(image.shape[1], x + 3*w//4)
        
        if tie_y_start >= tie_y_end or tie_x_start >= tie_x_end:
            return results
        
        tie_region = image[tie_y_start:tie_y_end, tie_x_start:tie_x_end]
        
        if tie_region.size == 0:
            return results
        
        # Convert to HSV
        hsv = cv2.cvtColor(tie_region, cv2.COLOR_BGR2HSV)
        
        # Check for red tie
        red_mask = cv2.inRange(hsv,
                             self.uniform_colors['red_tie']['lower'],
                             self.uniform_colors['red_tie']['upper'])
        red_percentage = np.sum(red_mask > 0) / red_mask.size
        
        # Check for blue tie
        blue_mask = cv2.inRange(hsv,
                              self.uniform_colors['blue_tie']['lower'],
                              self.uniform_colors['blue_tie']['upper'])
        blue_percentage = np.sum(blue_mask > 0) / blue_mask.size
        
        if red_percentage > 0.2:
            results['has_tie'] = True
            results['tie_color'] = 'red'
        elif blue_percentage > 0.2:
            results['has_tie'] = True
            results['tie_color'] = 'blue'
        
        return results
    
    def check_shirt_tucked(self, image, face_rect):
        """Check if shirt is properly tucked"""
        # Simplified check - look for belt line or color transition
        x, y, w, h = face_rect
        
        # Look for horizontal edges around waist area
        waist_y = min(y + int(h * 1.8), image.shape[0] - 1)
        waist_region_y_start = max(0, waist_y - 20)
        waist_region_y_end = min(image.shape[0], waist_y + 20)
        waist_region = image[waist_region_y_start:waist_region_y_end, x:x+w]
        
        if waist_region.size == 0:
            return None
        
        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(waist_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for horizontal lines (belt line)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        horizontal_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        
        horizontal_percentage = np.sum(horizontal_edges > 0) / horizontal_edges.size
        
        return horizontal_percentage > 0.1
    
    def calculate_uniform_score(self, results):
        """Calculate uniform compliance score (0-10)"""
        score = 10.0
        
        # Deductions for issues
        if not results['has_shirt']:
            score -= 3.0
        if not results['has_pants']:
            score -= 3.0
        if results['has_slippers']:
            score -= 2.5
        if not results['has_shoes'] and not results['has_slippers']:
            score -= 2.0  # No footwear
        if not results['has_tie']:
            score -= 1.5
        if results['improper_dress']:
            score -= 2.0
        if results['shirt_tucked'] == False:
            score -= 0.5
        
        return round(max(0, min(10, score)), 1)
    
    def detect_uniform_issues(self, results):
        """Detect specific uniform issues"""
        issues = []
        
        if not results['has_shirt']:
            issues.append("No school shirt detected")
        if not results['has_pants']:
            issues.append("No school pants detected")
        if results['has_slippers']:
            issues.append("Wearing slippers instead of shoes")
        if not results['has_shoes'] and not results['has_slippers']:
            issues.append("No proper footwear")
        if not results['has_tie']:
            issues.append("School tie not worn")
        if results['improper_dress']:
            issues.append("Improper dress detected")
        if results['shirt_tucked'] == False:
            issues.append("Shirt not properly tucked in")
        
        return issues