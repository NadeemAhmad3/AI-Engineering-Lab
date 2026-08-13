import os
import sys
import torch
import torch.nn as nn

DAY10_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY10_DIR not in sys.path:
    sys.path.insert(0, DAY10_DIR)

from inference.model import get_or_create_fp32_model, FP32_PATH, MODEL_DIR

INT8_PATH = os.path.join(MODEL_DIR, "production_int8.pt")

def get_quantized_int8_model(fp32_path: str = FP32_PATH, int8_path: str = INT8_PATH) -> nn.Module:
    """Applies dynamic INT8 quantization to linear layers."""
    model = get_or_create_fp32_model(fp32_path)
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    os.makedirs(os.path.dirname(int8_path), exist_ok=True)
    torch.save(quantized_model, int8_path)
    return quantized_model
