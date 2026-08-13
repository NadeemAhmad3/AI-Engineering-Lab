import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
FP32_PATH = os.path.join(MODEL_DIR, "fp32_model.pt")

class DeepQuantizableNet(nn.Module):
    """
    Multi-layer PyTorch classification network (~2M parameters) designed for FP32, FP16, and INT8 quantization benchmarks.
    Input: (B, 128) feature vector.
    Output: (B, 5) class logits.
    """
    def __init__(self, input_dim: int = 128, hidden_dim: int = 512, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def get_dataset(n_samples: int = 5000, input_dim: int = 128, n_classes: int = 5):
    """Generates synthetic classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=input_dim,
        n_informative=80,
        n_classes=n_classes,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long()
    )

def train_and_save_fp32_model(model_path: str = FP32_PATH) -> DeepQuantizableNet:
    """Trains FP32 model baseline and saves state dict."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    print("Training FP32 PyTorch baseline model for Day 7 quantization lab...")
    X_train, y_train, X_test, y_test = get_dataset()
    
    model = DeepQuantizableNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in range(5):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    torch.save(model.state_dict(), model_path)
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"FP32 Model saved to {model_path} (Size: {file_size_mb:.2f} MB)")
    return model

if __name__ == "__main__":
    train_and_save_fp32_model()
