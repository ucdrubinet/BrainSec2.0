'''convert trained segformer model to onnx format to run on CPU Engines like MacOS'''
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torch

# Load merged model (lora)
config = SegformerConfig.from_pretrained("/home/ajinkya/BS_two/src/ft_models/merged_lora_segformer")
model = SegformerForSemanticSegmentation.from_pretrained("/home/ajinkya/BS_two/src/ft_models/merged_lora_segformer") #merged model path

# Define dummy input
dummy_input = torch.randn(1, 3, 512, 512)  # Adjust if needed
onnx_path = "/home/ajinkya/BS_two/src/squeezemodel/segformer_lora.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=12
)
print(f"ONNX model saved at: {onnx_path}")
