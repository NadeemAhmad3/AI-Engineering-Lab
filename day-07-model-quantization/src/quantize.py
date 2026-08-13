import os
import sys
import torch
import torch.nn as nn

DAY7_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY7_DIR not in sys.path:
    sys.path.insert(0, DAY7_DIR)

from src.model import DeepQuantizableNet, train_and_save_fp32_model, FP32_PATH, MODEL_DIR

FP16_PATH = os.path.join(MODEL_DIR, "fp16_model.pt")
INT8_PATH = os.path.join(MODEL_DIR, "int8_model.pt")

def load_fp32_model(model_path: str = FP32_PATH) -> DeepQuantizableNet:
    model = DeepQuantizableNet()
    if not os.path.exists(model_path):
        train_and_save_fp32_model(model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def create_fp16_model(fp32_path: str = FP32_PATH, fp16_path: str = FP16_PATH) -> DeepQuantizableNet:
    """Converts PyTorch model weights to FP16 half precision."""
    model = load_fp32_model(fp32_path)
    model_fp16 = model.half()
    torch.save(model_fp16.state_dict(), fp16_path)
    size_mb = os.path.getsize(fp16_path) / (1024 * 1024)
    print(f"[Quantize Engine] FP16 Model saved to {fp16_path} (Size: {size_mb:.2f} MB)")
    return model_fp16

def create_int8_model(fp32_path: str = FP32_PATH, int8_path: str = INT8_PATH) -> nn.Module:
    """Applies PyTorch dynamic INT8 quantization to Linear layers."""
    model = load_fp32_model(fp32_path)
    
    # Quantize linear layers to qint8 dynamically
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    
    torch.save(quantized_model, int8_path)
    size_mb = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"[Quantize Engine] INT8 Quantized Model saved to {int8_path} (Size: {size_mb:.2f} MB)")
    return quantized_model

def get_all_models():
    fp32 = load_fp32_model()
    fp16 = create_fp16_model()
    int8 = create_int8_model()
    return {"fp32": fp32, "fp16": fp16, "int8": int8}

if __name__ == "__main__":
    get_all_models()
