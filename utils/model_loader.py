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
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
                ]
        )
        
        # Class names from your Colab training
        self.class_names = {
            'facial': ['clean_shaven', 'bearded'],
            'footwear': ['shoes', 'sandals', 'slippers'],
            'uniform': ['casual_wear', 'proper_uniform'],
            'gender': ['male', 'female']
        }
    
    def load_model(self, model_path):
        """Load the exact model architecture from Colab"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create the EXACT model architecture from your Colab code
        class AllInOneGroomingModel(nn.Module):
            def __init__(self):
                super(AllInOneGroomingModel, self).__init__()

                self.backbone = models.mobilenet_v2(pretrained=False)
                feature_dim = self.backbone.last_channel
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

                self.gender_head = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(feature_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )

                self.compliance_head = nn.Sequential(
                    nn.Linear(feature_dim + 6, 64), 
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):

                features = self.backbone.features(x)
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = torch.flatten(features, 1)

                facial_pred = self.facial_head(features)
                footwear_pred = self.footwear_head(features)
                uniform_pred = self.uniform_head(features)
                gender_pred = self.gender_head(features)

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

        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        input_tensor = self.transform(face_pil)
        
        return input_tensor.unsqueeze(0).to(self.device)
    
    def predict(self, face_image):
        """Predict grooming attributes from face image"""
        try:
            if face_image.size == 0 or face_image.shape[0] < 50 or face_image.shape[1] < 50:
                return self._get_default_result()
            
            input_tensor = self.preprocess_face(face_image)
            
            with torch.no_grad():
                facial_pred, footwear_pred, uniform_pred, gender_pred, compliance_pred = self.model(input_tensor)
            
            facial_prob = facial_pred.item()
            footwear_probs = footwear_pred[0].tolist()
            uniform_prob = uniform_pred.item()
            gender_prob = gender_pred.item()
            compliance_prob = compliance_pred.item()
            
            facial_class = self.class_names['facial'][1] if facial_prob > 0.5 else self.class_names['facial'][0]
            footwear_class = self.class_names['footwear'][np.argmax(footwear_probs)]
            uniform_class = self.class_names['uniform'][1] if uniform_prob > 0.5 else self.class_names['uniform'][0]
            gender_class = self.class_names['gender'][1] if gender_prob > 0.5 else self.class_names['gender'][0]
            
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
                'overall_compliant': False, 
                'violations': [] 
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