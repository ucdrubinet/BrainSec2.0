from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torch

def model_size_mb(model):
    total = 0
    for name, param in model.named_parameters():
        total += param.numel() * param.element_size()
    return total / (1024 ** 2)

def quantize_model(model, layers=None, dtype=torch.qint8):
    return torch.quantization.quantize_dynamic(model, layers, dtype=dtype)



# Load merged model (weights + adapters fused)
config = SegformerConfig.from_pretrained("/home/ajinkya/BS_two/src/ft_models/merged_lora_segformer")
model = SegformerForSemanticSegmentation.from_pretrained("/home/ajinkya/BS_two/src/ft_models/merged_lora_segformer")
print(f"Main Model size: {model_size_mb(model):.2f} MB")


# Quantize (Linear layers only)
model_quant_linear = quantize_model(model, {torch.nn.Linear})
print(f"Quantized (Linear only) size: {model_size_mb(model_quant_linear):.2f} MB")

# Quantize (Linear + LayerNorm)
model_quant_ln = quantize_model(model, {torch.nn.Linear, torch.nn.LayerNorm})
print(f"Quantized (Linear + LayerNorm) size: {model_size_mb(model_quant_ln):.2f} MB")

# Quantize (all layers)
model_quant_all = quantize_model(model, None)
print(f"Quantized (All layers) size: {model_size_mb(model_quant_all):.2f} MB")


# Save state_dict + config
torch.save(model_quant_all.state_dict(), "/home/ajinkya/BS_two/src/squeezemodel/quant_model_all.pth")
config.save_pretrained("/home/ajinkya/BS_two/src/squeezemodel/quant_model_all_config")

# Load back the quantized model to verify
print(f"Model size: {model_size_mb(model_quant):.2f} MB")


