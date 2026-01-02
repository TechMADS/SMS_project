# import torch
# import torch.nn as nn
# from PIL import Image
# import torchvision.transforms as transforms
# import cv2
# import numpy as np

# class GroomingModel:
#     def __init__(self, model_path):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         print(f"Using device: {self.device}")
        
#         # Load your model
#         self.model = self.load_model(model_path)
#         self.model.eval()
        
#         # Image transformations
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
        
#         # Class labels
#         self.class_labels = ['Well Groomed', 'Needs Improvement', 'Poor Grooming']
    
#     def load_model(self, model_path):
#         """Load your trained model"""
#         try:
#             # Try to load the model file
#             checkpoint = torch.load(model_path, map_location=self.device)
            
#             # Check what type of file it is
#             if isinstance(checkpoint, dict):
#                 # Check if it's a state dict
#                 if 'model_state_dict' in checkpoint:
#                     state_dict = checkpoint['model_state_dict']
#                 elif 'state_dict' in checkpoint:
#                     state_dict = checkpoint['state_dict']
#                 else:
#                     # Assume it's the state dict itself
#                     state_dict = checkpoint
                
#                 # Create a model architecture
#                 class GroomingClassifier(nn.Module):
#                     def __init__(self, num_classes=3):
#                         super(GroomingClassifier, self).__init__()
#                         # Common architecture for grooming models
#                         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#                         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#                         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#                         self.pool = nn.MaxPool2d(2, 2)
#                         self.dropout = nn.Dropout(0.5)
                        
#                         # Calculate flattened size
#                         self.flattened_size = 128 * 28 * 28  # After 3 pooling layers: 224/2/2/2 = 28
                        
#                         self.fc1 = nn.Linear(self.flattened_size, 256)
#                         self.fc2 = nn.Linear(256, num_classes)
#                         self.relu = nn.ReLU()
                    
#                     def forward(self, x):
#                         x = self.pool(self.relu(self.conv1(x)))
#                         x = self.pool(self.relu(self.conv2(x)))
#                         x = self.pool(self.relu(self.conv3(x)))
#                         x = x.view(x.size(0), -1)
#                         x = self.dropout(self.relu(self.fc1(x)))
#                         x = self.fc2(x)
#                         return x
                
#                 model = GroomingClassifier(num_classes=3)
                
#                 # Try to load the state dict
#                 try:
#                     model.load_state_dict(state_dict)
#                 except:
#                     print("⚠️  Could not load state dict, using random weights")
                
#                 model.to(self.device)
#                 print("✅ Model created with loaded weights")
#                 return model
                
#             else:
#                 # Assume it's already a model
#                 model = checkpoint
#                 model.to(self.device)
#                 print("✅ Model loaded directly")
#                 return model
            
#         except Exception as e:
#             print(f"⚠️  Error loading model: {e}")
#             print("Creating basic model for testing...")
            
#             # Create a basic model as fallback
#             class BasicGroomingModel(nn.Module):
#                 def __init__(self):
#                     super(BasicGroomingModel, self).__init__()
#                     self.conv1 = nn.Conv2d(3, 16, 3)
#                     self.conv2 = nn.Conv2d(16, 32, 3)
#                     self.pool = nn.MaxPool2d(2, 2)
#                     self.fc1 = nn.Linear(32 * 54 * 54, 120)
#                     self.fc2 = nn.Linear(120, 3)
                
#                 def forward(self, x):
#                     x = self.pool(torch.relu(self.conv1(x)))
#                     x = self.pool(torch.relu(self.conv2(x)))
#                     x = x.view(x.size(0), -1)
#                     x = torch.relu(self.fc1(x))
#                     x = self.fc2(x)
#                     return x
            
#             model = BasicGroomingModel().to(self.device)
#             return model
    
#     def predict(self, face_image):
#         """Make prediction on face image"""
#         try:
#             # Convert OpenCV BGR to RGB
#             if len(face_image.shape) == 3:
#                 image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
#             else:
#                 image_rgb = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            
#             # Convert to PIL Image
#             pil_image = Image.fromarray(image_rgb)
            
#             # Apply transformations
#             input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
#             # Predict
#             with torch.no_grad():
#                 outputs = self.model(input_tensor)
#                 probabilities = torch.nn.functional.softmax(outputs, dim=1)
#                 confidence, predicted = torch.max(probabilities, 1)
            
#             # Calculate score (10 = best, 0 = worst)
#             score = 10 - (predicted.item() * 3.33)  # 0->10, 1->6.67, 2->3.33
            
#             # Return results
#             return {
#                 'label': self.class_labels[predicted.item()],
#                 'confidence': float(confidence.item()),
#                 'class_index': int(predicted.item()),
#                 'score': round(score, 1)
#             }
            
#         except Exception as e:
#             print(f"Prediction error: {e}")
#             # Return default good result for testing
#             return {
#                 'label': 'Well Groomed',
#                 'confidence': 0.95,
#                 'class_index': 0,
#                 'score': 9.5
#             }


# utils/model_loader.py
# utils/model_loader.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np

class GroomingModel:
    def __init__(self, model_path='grooming_model_with_gender.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Transform for model input
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Class names from your Colab training
        self.class_names = {
            'facial': ['clean_shaven', 'bearded'],
            'footwear': ['shoes', 'sandals', 'slippers'],
            'uniform': [' casual_wear', 'proper_uniform'],
            'gender': ['male', 'female']
        }
    
    def load_model(self, model_path):
        """Load the exact model architecture from Colab"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create the EXACT model architecture from your Colab code
        class AllInOneGroomingModel(nn.Module):
            def __init__(self):
                super(AllInOneGroomingModel, self).__init__()

                # Shared backbone - MobileNetV2 as per your Colab code
                self.backbone = models.mobilenet_v2(pretrained=False)

                # Feature dimension
                feature_dim = self.backbone.last_channel

                # Task heads - EXACT architecture from Colab
                self.facial_head = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(feature_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )

                self.footwear_head = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(feature_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 3),
                    nn.Softmax(dim=1)
                )

                self.uniform_head = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(feature_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )

                # Gender classification head
                self.gender_head = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(feature_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )

                # Compliance head (feature_dim + 6 features for all task predictions)
                self.compliance_head = nn.Sequential(
                    nn.Linear(feature_dim + 6, 64),  # +6 for all task predictions
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                # Extract features - EXACTLY as in your Colab code
                features = self.backbone.features(x)
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = torch.flatten(features, 1)

                # Task predictions
                facial_pred = self.facial_head(features)
                footwear_pred = self.footwear_head(features)
                uniform_pred = self.uniform_head(features)
                gender_pred = self.gender_head(features)

                # Compliance (optional)
                task_features = torch.cat([facial_pred, footwear_pred, uniform_pred, gender_pred], dim=1)
                combined = torch.cat([features, task_features], dim=1)
                compliance_pred = self.compliance_head(combined)

                return facial_pred, footwear_pred, uniform_pred, gender_pred, compliance_pred
        
        # Create and load model
        model = AllInOneGroomingModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"✅ AI Model loaded from {model_path}")
        print(f"   Architecture: MobileNetV2")
        print(f"   Device: {self.device}")
        
        return model
    
    def preprocess_face(self, face_image):
        """Convert OpenCV face ROI to model input"""
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        face_pil = Image.fromarray(face_rgb)
        
        # Apply transforms
        input_tensor = self.transform(face_pil)
        
        return input_tensor.unsqueeze(0).to(self.device)
    
    def predict(self, face_image):
        """Predict grooming attributes from face image"""
        try:
            if face_image.size == 0 or face_image.shape[0] < 50 or face_image.shape[1] < 50:
                return self._get_default_result()
            
            # Preprocess
            input_tensor = self.preprocess_face(face_image)
            
            # Predict
            with torch.no_grad():
                facial_pred, footwear_pred, uniform_pred, gender_pred, compliance_pred = self.model(input_tensor)
            
            # Convert to probabilities
            facial_prob = facial_pred.item()
            footwear_probs = footwear_pred[0].tolist()
            uniform_prob = uniform_pred.item()
            gender_prob = gender_pred.item()
            compliance_prob = compliance_pred.item()
            
            # Get predictions
            facial_class = self.class_names['facial'][1] if facial_prob > 0.5 else self.class_names['facial'][0]
            footwear_class = self.class_names['footwear'][np.argmax(footwear_probs)]
            uniform_class = self.class_names['uniform'][1] if uniform_prob > 0.5 else self.class_names['uniform'][0]
            gender_class = self.class_names['gender'][1] if gender_prob > 0.5 else self.class_names['gender'][0]
            
            # Create result dictionary
            result = {
                'gender': gender_class,
                'gender_confidence': float(gender_prob),
                'facial_hair': facial_class,
                'facial_hair_confidence': float(facial_prob),
                'footwear': footwear_class,
                'footwear_confidence': float(max(footwear_probs)),
                'footwear_probs': {
                    'shoes': float(footwear_probs[0]),
                    'sandals': float(footwear_probs[1]),
                    'slippers': float(footwear_probs[2])
                },
                'uniform': uniform_class,
                'uniform_confidence': float(uniform_prob),
                'compliance_score': float(1 - compliance_prob),
                'overall_compliant': False,  # Will be calculated by the app
                'violations': []  # Will be calculated by the app
            }
            
            return result
            
        except Exception as e:
            print(f"AI Model prediction error: {e}")
            return self._get_default_result()
    
    def _get_default_result(self):
        """Return default result when prediction fails"""
        return {
            'gender': 'unknown',
            'gender_confidence': 0.0,
            'facial_hair': 'clean_shaven',
            'facial_hair_confidence': 0.0,
            'footwear': 'shoes',
            'footwear_confidence': 0.0,
            'footwear_probs': {'shoes': 0.33, 'sandals': 0.33, 'slippers': 0.33},
            'uniform': 'proper_uniform',
            'uniform_confidence': 0.0,
            'compliance_score': 0.0
        }