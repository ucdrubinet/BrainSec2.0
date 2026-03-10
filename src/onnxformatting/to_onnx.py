'''convert trained segformer model to onnx format to run on CPU Engines like MacOS'''
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torch
import argparse

# CLI argument parser
parser = argparse.ArgumentParser(description="Convert SegFormer model to ONNX format")
parser.add_argument("model_dir", help="Directory containing the model (config and weights)")
parser.add_argument("--output", type=str, required=True, help="Output path for ONNX model")
parser.add_argument("--input-size", type=int, default=512, help="Input image size (default: 512)")
parser.add_argument("--batch-size", type=int, default=1, help="Batch size for dummy input (default: 1)")
parser.add_argument("--opset-version", type=int, default=12, help="ONNX opset version (default: 12)")
args = parser.parse_args()

# Load merged model (lora)
print(f"Loading model from: {args.model_dir}")
config = SegformerConfig.from_pretrained(args.model_dir)
model = SegformerForSemanticSegmentation.from_pretrained(args.model_dir)

# Define dummy input
dummy_input = torch.randn(args.batch_size, 3, args.input_size, args.input_size)
print(f"Converting to ONNX with input shape: {dummy_input.shape}")

torch.onnx.export(
    model,
    dummy_input,
    args.output,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=args.opset_version
)
print(f"ONNX model saved at: {args.output}")
