import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np

class ReIDFeatureExtractor:
    def __init__(self, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        # Load pre-trained ResNet18 and remove the classification head
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.model = self.model.to(self.device).eval()
        
        self.transform = T.Compose([
            T.Resize((256, 128)), # Standard ReID size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return np.zeros(512, dtype=np.float32)
            
        img = Image.fromarray(crop[..., ::-1]) # BGR to RGB
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feat = self.model(img_t).flatten().cpu().numpy()
            
        # L2 normalization
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
            
        return feat
