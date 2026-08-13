import os
import torch
import torch.nn as nn

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
FP32_PATH = os.path.join(MODEL_DIR, "production_fp32.pt")

class ProductionInferenceNet(nn.Module):
    """
    Production PyTorch Neural Network for Capstone System Design.
    Input: (B, 128) feature vector.
    Output: (B, 10) class logits.
    """
    def __init__(self, input_dim: int = 128, hidden_dim: int = 512, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def get_or_create_fp32_model(model_path: str = FP32_PATH) -> ProductionInferenceNet:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model = ProductionInferenceNet()
    if not os.path.exists(model_path):
        torch.save(model.state_dict(), model_path)
    else:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
