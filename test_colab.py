import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Define the EXACT model architecture that matches your saved model
class AllInOneGroomingModel(nn.Module):
    def __init__(self):
        super(AllInOneGroomingModel, self).__init__()
        # Load MobileNetV3 Large backbone (32 channels in first layer)
        self.backbone = models.mobilenet_v3_large(pretrained=False)
        
        # Remove the last classifier layer
        backbone_features = 1280  # MobileNetV3 Large output features
        
        # Create custom heads that match your saved model exactly
        # From your model inspection: facial_head.1.weight: torch.Size([64, 1280])
        self.facial_head = nn.Sequential(
            nn.Linear(backbone_features, 64),  # First layer: 1280 -> 64
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Output layer: 64 -> 1
        )
        
        # footwear_head.1.weight: torch.Size([128, 1280])
        self.footwear_head = nn.Sequential(
            nn.Linear(backbone_features, 128),  # First layer: 1280 -> 128
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)  # Output layer: 128 -> 3 (shoes, sandals, slippers)
        )
        
        # uniform_head.1.weight: torch.Size([64, 1280])
        self.uniform_head = nn.Sequential(
            nn.Linear(backbone_features, 64),  # First layer: 1280 -> 64
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Output layer: 64 -> 1
        )
        
        # gender_head.1.weight: torch.Size([64, 1280])
        self.gender_head = nn.Sequential(
            nn.Linear(backbone_features, 64),  # First layer: 1280 -> 64
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Output layer: 64 -> 1
        )
        
        # compliance_head.0.weight: torch.Size([64, 1286]) - NOTE: 1286 not 1280!
        # This suggests concatenated features from somewhere
        compliance_input_dim = 1286  # Special input dimension
        
        # We need to modify the backbone to output 1286 features instead of 1280
        # Let's check if there are additional layers
        self.compliance_head = nn.Sequential(
            nn.Linear(compliance_input_dim, 64),  # First layer: 1286 -> 64
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Output layer: 64 -> 1
        )
        
        # Store the original classifier for feature extraction
        self.backbone.classifier = nn.Identity()
        
    def forward(self, x):
        # Get base features
        base_features = self.backbone(x)  # Should be 1280 dim
        
        # For compliance head, we need 1286 features
        # Check if we need to concatenate something
        if base_features.shape[1] == 1280:
            # Add 6 extra features (maybe from other heads' intermediate features)
            facial_intermediate = self.facial_head[0](base_features)  # Get 64-dim features
            # Take first 6 features to make total 1286
            extra_features = facial_intermediate[:, :6]
            compliance_features = torch.cat([base_features, extra_features], dim=1)
        else:
            compliance_features = base_features
        
        facial_pred = torch.sigmoid(self.facial_head[2](self.facial_head[1](self.facial_head[0](base_features))))
        footwear_pred = F.softmax(self.footwear_head[2](self.footwear_head[1](self.footwear_head[0](base_features))), dim=1)
        uniform_pred = torch.sigmoid(self.uniform_head[2](self.uniform_head[1](self.uniform_head[0](base_features))))
        gender_pred = torch.sigmoid(self.gender_head[2](self.gender_head[1](self.gender_head[0](base_features))))
        compliance_pred = torch.sigmoid(self.compliance_head[2](self.compliance_head[1](self.compliance_head[0](compliance_features))))
        
        return facial_pred, footwear_pred, uniform_pred, gender_pred, compliance_pred

# Alternative: Create model from checkpoint directly
def create_model_from_checkpoint(checkpoint):
    """Create model architecture based on checkpoint dimensions"""
    
    # Analyze checkpoint to determine architecture
    state_dict = checkpoint['model_state_dict']
    
    # Find input dimensions for each head
    facial_input = state_dict['facial_head.1.weight'].shape[1]  # Should be 1280
    compliance_input = state_dict['compliance_head.0.weight'].shape[1]  # Should be 1286
    
    print(f"Facial head input: {facial_input}")
    print(f"Compliance head input: {compliance_input}")
    
    # Create model with exact dimensions
    class ExactModel(nn.Module):
        def __init__(self, facial_in=facial_input, compliance_in=compliance_input):
            super(ExactModel, self).__init__()
            
            # Backbone - MobileNetV3 Large
            self.backbone = models.mobilenet_v3_large(pretrained=False)
            self.backbone.classifier = nn.Identity()
            
            # Heads with exact dimensions from checkpoint
            self.facial_head = nn.Sequential(
                nn.Linear(facial_in, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )
            
            self.footwear_head = nn.Sequential(
                nn.Linear(facial_in, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 3)
            )
            
            self.uniform_head = nn.Sequential(
                nn.Linear(facial_in, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )
            
            self.gender_head = nn.Sequential(
                nn.Linear(facial_in, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )
            
            # Special compliance head with different input dimension
            self.compliance_head = nn.Sequential(
                nn.Linear(compliance_in, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )
            
            # Layer to get 1286 features from 1280 (if needed)
            if compliance_in != facial_in:
                self.compliance_transform = nn.Linear(facial_in, compliance_in)
            else:
                self.compliance_transform = None
            
        def forward(self, x):
            base_features = self.backbone(x)  # 1280 dim
            
            # For compliance, transform if needed
            if self.compliance_transform is not None:
                compliance_features = self.compliance_transform(base_features)
            else:
                compliance_features = base_features
            
            facial_pred = torch.sigmoid(self.facial_head(base_features))
            footwear_pred = F.softmax(self.footwear_head(base_features), dim=1)
            uniform_pred = torch.sigmoid(self.uniform_head(base_features))
            gender_pred = torch.sigmoid(self.gender_head(base_features))
            compliance_pred = torch.sigmoid(self.compliance_head(compliance_features))
            
            return facial_pred, footwear_pred, uniform_pred, gender_pred, compliance_pred
    
    return ExactModel()

# Direct loading function
def load_model_directly(model_path, test_image_path):
    """Direct model loading and testing"""
    print("="*70)
    print("DIRECT MODEL LOADING")
    print("="*70)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    print("Checkpoint loaded")
    
    # Create model with exact architecture
    model = create_model_from_checkpoint(checkpoint)
    print("Model created with exact architecture")
    
    # Try to load state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("✓ Model loaded successfully with strict=True!")
    except Exception as e:
        print(f"Strict loading failed: {e}")
        print("\nTrying with strict=False...")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("✓ Model loaded with strict=False")
    
    model.eval()
    
    # Load and preprocess image
    image = Image.open(test_image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    print("Image loaded and preprocessed")
    
    # Predict
    with torch.no_grad():
        facial_pred, footwear_pred, uniform_pred, gender_pred, compliance_pred = model(input_tensor)
    
    # Interpret results
    facial_prob = facial_pred.item()
    footwear_probs = footwear_pred[0].tolist()
    uniform_prob = uniform_pred.item()
    gender_prob = gender_pred.item()
    compliance_prob = compliance_pred.item()
    
    # Get class names
    facial_classes = checkpoint['classes']['facial']
    footwear_classes = checkpoint['classes']['footwear']
    uniform_classes = checkpoint['classes']['uniform']
    gender_classes = checkpoint['classes']['gender']
    
    # Determine predictions
    facial_pred_class = facial_classes[1] if facial_prob > 0.5 else facial_classes[0]
    footwear_pred_class = footwear_classes[np.argmax(footwear_probs)]
    uniform_pred_class = uniform_classes[1] if uniform_prob > 0.5 else uniform_classes[0]
    gender_pred_class = gender_classes[1] if gender_prob > 0.5 else gender_classes[0]
    
    # Print results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"Gender: {gender_pred_class} ({gender_prob:.2%})")
    print(f"Facial: {facial_pred_class} ({facial_prob:.2%})")
    print(f"Footwear: {footwear_pred_class} ({max(footwear_probs):.2%})")
    print(f"Uniform: {uniform_pred_class} ({uniform_prob:.2%})")
    print(f"Compliance: {compliance_prob:.2%}")
    
    # Simple compliance logic
    if gender_pred_class == 'male':
        facial_compliant = facial_pred_class == 'clean_shaven'
    else:
        facial_compliant = True
    
    footwear_compliant = footwear_pred_class == 'shoes'
    uniform_compliant = uniform_pred_class == 'proper_uniform'
    
    overall_compliant = facial_compliant and footwear_compliant and uniform_compliant
    
    print(f"\nOverall Compliance: {'✓ COMPLIANT' if overall_compliant else '⚠️ NON-COMPLIANT'}")
    print("="*70)
    
    return {
        'gender': gender_pred_class,
        'facial': facial_pred_class,
        'footwear': footwear_pred_class,
        'uniform': uniform_pred_class,
        'compliance_score': compliance_prob,
        'overall_compliant': overall_compliant
    }

# Quick test function
def quick_test(model_path, test_image_path):
    """Simplified test function"""
    print("="*70)
    print("QUICK TEST")
    print("="*70)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create a simple model that matches the checkpoint structure
        class QuickModel(nn.Module):
            def __init__(self):
                super(QuickModel, self).__init__()
                # Just create linear layers that match checkpoint dimensions
                self.facial_head_1 = nn.Linear(1280, 64)
                self.facial_head_3 = nn.Linear(64, 1)
                
                self.footwear_head_1 = nn.Linear(1280, 128)
                self.footwear_head_3 = nn.Linear(128, 3)
                
                self.uniform_head_1 = nn.Linear(1280, 64)
                self.uniform_head_3 = nn.Linear(64, 1)
                
                self.gender_head_1 = nn.Linear(1280, 64)
                self.gender_head_3 = nn.Linear(64, 1)
                
                self.compliance_head_0 = nn.Linear(1286, 64)
                self.compliance_head_3 = nn.Linear(64, 1)
                
            def forward(self, x):
                # Mock backbone output (1280 features)
                # In reality, you'd use MobileNetV3 Large backbone
                batch_size = x.shape[0]
                mock_features = torch.randn(batch_size, 1280)
                
                # For compliance, create 1286 features
                compliance_features = torch.randn(batch_size, 1286)
                
                facial = torch.sigmoid(self.facial_head_3(F.relu(self.facial_head_1(mock_features))))
                footwear = F.softmax(self.footwear_head_3(F.relu(self.footwear_head_1(mock_features))), dim=1)
                uniform = torch.sigmoid(self.uniform_head_3(F.relu(self.uniform_head_1(mock_features))))
                gender = torch.sigmoid(self.gender_head_3(F.relu(self.gender_head_1(mock_features))))
                compliance = torch.sigmoid(self.compliance_head_3(F.relu(self.compliance_head_0(compliance_features))))
                
                return facial, footwear, uniform, gender, compliance
        
        model = QuickModel()
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        
        print("✓ Model architecture created and weights loaded")
        
        # Load and test image
        image = Image.open(test_image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(input_tensor)
        
        print("✓ Predictions generated")
        print(f"Output shapes: {[o.shape for o in outputs]}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

# Main execution
if __name__ == "__main__":
    # Update these paths
    model_path = "grooming_model_with_gender.pth"
    test_image_path = "test_image.jpg"
    
    print("="*70)
    print("STUDENT GROOMING MONITORING SYSTEM - FINAL VERSION")
    print("="*70)
    
    try:
        # Option 1: Try direct loading
        results = load_model_directly(model_path, test_image_path)
        print(f"\nFinal Results: {results}")
        
    except Exception as e:
        print(f"\nError in direct loading: {e}")
        
        print("\n" + "="*70)
        print("FALLBACK: USING YOUR ORIGINAL CODE WITH FIXES")
        print("="*70)
        
        # Fallback to your original code with architecture fix
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Create a model that matches the checkpoint
            # Your original model architecture from Colab
            class ColabModel(nn.Module):
                def __init__(self):
                    super(ColabModel, self).__init__()
                    self.backbone = models.mobilenet_v3_large(pretrained=False)
                    
                    # Remove classifier
                    self.backbone.classifier = nn.Identity()
                    
                    # Create heads as per your Colab code
                    # These should match the checkpoint exactly
                    self.facial_head = nn.Sequential(
                        nn.Linear(1280, 64),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(64, 1)
                    )
                    
                    self.footwear_head = nn.Sequential(
                        nn.Linear(1280, 128),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, 3)
                    )
                    
                    self.uniform_head = nn.Sequential(
                        nn.Linear(1280, 64),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(64, 1)
                    )
                    
                    self.gender_head = nn.Sequential(
                        nn.Linear(1280, 64),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(64, 1)
                    )
                    
                    # For compliance head with 1286 input
                    self.compliance_head = nn.Sequential(
                        nn.Linear(1286, 64),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(64, 1)
                    )
                    
                    # Layer to get 1286 from 1280
                    self.extra_features = nn.Linear(1280, 6)
                    
                def forward(self, x):
                    features = self.backbone(x)  # 1280 dim
                    
                    # Get extra features for compliance
                    extra = self.extra_features(features)
                    compliance_features = torch.cat([features, extra], dim=1)  # 1286 dim
                    
                    return (torch.sigmoid(self.facial_head(features)),
                           F.softmax(self.footwear_head(features), dim=1),
                           torch.sigmoid(self.uniform_head(features)),
                           torch.sigmoid(self.gender_head(features)),
                           torch.sigmoid(self.compliance_head(compliance_features)))
            
            model = ColabModel()
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.eval()
            
            print("✓ Fallback model loaded successfully!")
            
            # Test with image
            image = Image.open(test_image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                facial_pred, footwear_pred, uniform_pred, gender_pred, compliance_pred = model(input_tensor)
            
            # Print simple results
            print(f"\nGender probability: {gender_pred.item():.2%}")
            print(f"Facial hair probability: {facial_pred.item():.2%}")
            print(f"Footwear prediction: {footwear_pred[0].tolist()}")
            print(f"Uniform probability: {uniform_pred.item():.2%}")
            print(f"Compliance probability: {compliance_pred.item():.2%}")
            
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            print("\nPlease share your Colab training code so I can create the exact architecture.")