import torch
import torch.nn as nn

class CNNInferenceNet(nn.Module):
    """
    Standard PyTorch Convolutional Neural Network for CPU vs GPU inference benchmark.
    Input: (B, 3, 64, 64) image tensor.
    Output: (B, 10) class logits.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 32x32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 16x16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 8x8
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

def get_model(device: str = "cpu") -> CNNInferenceNet:
    model = CNNInferenceNet()
    model.eval()
    return model.to(device)
