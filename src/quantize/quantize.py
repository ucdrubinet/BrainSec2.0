from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torch
import argparse
from pathlib import Path


def model_size_mb(model):
    """Calculate total model size in MB."""
    total = 0
    for name, param in model.named_parameters():
        total += param.numel() * param.element_size()
    return total / (1024 ** 2)


def quantize_model(model, layers=None, dtype=torch.qint8):
    """Quantize model using dynamic quantization."""
    return torch.quantization.quantize_dynamic(model, layers, dtype=dtype)


def main():
    parser = argparse.ArgumentParser(description="Quantize SegFormer model")
    parser.add_argument("model_dir", help="Directory containing model config and weights")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for quantized models")
    parser.add_argument("--dtype", type=str, default="qint8", choices=["qint8", "float16"], 
                        help="Quantization dtype (default: qint8)")
    args = parser.parse_args()
    
    # Convert dtype string to torch type
    dtype = torch.qint8 if args.dtype == "qint8" else torch.float16
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from: {args.model_dir}")
    config = SegformerConfig.from_pretrained(args.model_dir)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_dir)
    original_size = model_size_mb(model)
    print(f"Original model size: {original_size:.2f} MB")
    
    # Quantize with different configurations
    quantization_configs = [
        ("linear_only", {torch.nn.Linear}),
        ("linear_layernorm", {torch.nn.Linear, torch.nn.LayerNorm}),
        ("all_layers", None)
    ]
    
    quantized_models = {}
    for config_name, layers in quantization_configs:
        print(f"\nQuantizing {config_name}...")
        quantized_models[config_name] = quantize_model(model, layers, dtype=dtype)
        size = model_size_mb(quantized_models[config_name])
        reduction = ((original_size - size) / original_size) * 100
        print(f"Size: {size:.2f} MB (reduction: {reduction:.1f}%)")
    
    # Save all quantized models
    print(f"\nSaving quantized models to: {output_dir}")
    for config_name, quantized_model in quantized_models.items():
        model_path = output_dir / f"quant_{config_name}.pth"
        torch.save(quantized_model.state_dict(), model_path)
        print(f"Saved {model_path}")
        
        config_path = output_dir / f"quant_{config_name}_config"
        config.save_pretrained(config_path)
        print(f"Saved config to {config_path}")
    
    print("\nQuantization complete!")


if __name__ == "__main__":
    main()


