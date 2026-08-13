import os
import sys
import torch
import numpy as np
from typing import Dict, Tuple

DAY7_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY7_DIR not in sys.path:
    sys.path.insert(0, DAY7_DIR)

from src.model import get_dataset, DeepQuantizableNet
from src.quantize import load_fp32_model, create_fp16_model, create_int8_model

def evaluate_model_accuracy(model: torch.nn.Module, is_fp16: bool = False) -> float:
    """Evaluates classification accuracy on synthetic test dataset."""
    _, _, X_test, y_test = get_dataset(n_samples=2000)
    
    if is_fp16:
        X_test = X_test.half()
        
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = torch.argmax(logits, dim=1)
        correct = (preds == y_test).sum().item()
        acc = (correct / len(y_test)) * 100.0
    return round(acc, 2)

def evaluate_all_precisions() -> Dict[str, Dict]:
    fp32_model = load_fp32_model()
    fp16_model = create_fp16_model()
    int8_model = create_int8_model()
    
    fp32_acc = evaluate_model_accuracy(fp32_model, is_fp16=False)
    fp16_acc = evaluate_model_accuracy(fp16_model, is_fp16=True)
    int8_acc = evaluate_model_accuracy(int8_model, is_fp16=False)
    
    return {
        "fp32": {
            "accuracy": fp32_acc,
            "accuracy_drop": 0.0,
            "meets_1pct_budget": True
        },
        "fp16": {
            "accuracy": fp16_acc,
            "accuracy_drop": round(fp32_acc - fp16_acc, 2),
            "meets_1pct_budget": (fp32_acc - fp16_acc) <= 1.0
        },
        "int8": {
            "accuracy": int8_acc,
            "accuracy_drop": round(fp32_acc - int8_acc, 2),
            "meets_1pct_budget": (fp32_acc - int8_acc) <= 1.0
        }
    }

if __name__ == "__main__":
    res = evaluate_all_precisions()
    print("Evaluation Results:", res)
