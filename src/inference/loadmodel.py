''' Script to load Segformer models with various configurations (pretrained, finetuned, lora, qlora) for inference '''
import torch
from transformers import SegformerForSemanticSegmentation, SegformerConfig, BitsAndBytesConfig
from peft import PeftModel
import os
import argparse


def load_model(
    model_type,
    device,
    model_dir=None,
    base_model_dir=None
):
    """
    Load a Segformer pretrained/finetuned model based on the specified type.
    
    Args:
        model_type: Type of model ("pretrained", "finetuned_full", "finetuned_lastlayer", "lora", "qlora")
        device: Torch device to load model on
        model_dir: Directory containing model files
        base_model_dir: Base model directory for lora/qlora (required for lora/qlora models)
        
    Returns:
        Loaded model
    """
    if model_type == "pretrained":
        if not model_dir:
            raise ValueError("model_dir is required for pretrained model")
        
        config_path = os.path.join(model_dir, "config.json")
        config = SegformerConfig.from_pretrained(config_path)
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_dir,
            config=config,
            local_files_only=True
        )
        model.to(device)
        model.eval()
        print(f"Loaded pretrained model from {model_dir}")
        return model

    elif model_type == "finetuned_full":
        if not model_dir:
            raise ValueError("model_dir is required for finetuned_full model")
        
        config_path = os.path.join(model_dir, "config.json")
        state_dict_path = os.path.join(model_dir, "finetuned_full.pth")
        
        config = SegformerConfig.from_pretrained(config_path)
        model = SegformerForSemanticSegmentation(config)
        state_dict = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Loaded finetuned_full model from {model_dir}")
        return model

    elif model_type == "finetuned_lastlayer":
        if not model_dir:
            raise ValueError("model_dir is required for finetuned_lastlayer model")
        
        config_path = os.path.join(model_dir, "config.json")
        state_dict_path = os.path.join(model_dir, "finetuned_lastlayer.pth")
        
        config = SegformerConfig.from_pretrained(config_path)
        model = SegformerForSemanticSegmentation(config)
        state_dict = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Loaded finetuned_lastlayer model from {model_dir}")
        return model

    elif model_type == "lora":
        if not model_dir:
            raise ValueError("model_dir is required for lora model")
        if not base_model_dir:
            raise ValueError("base_model_dir is required for lora model")
        
        config_path = os.path.join(base_model_dir, "config.json")
        config = SegformerConfig.from_pretrained(config_path)
        base_model = SegformerForSemanticSegmentation.from_pretrained(
            base_model_dir,
            config=config,
            local_files_only=True
        )
        model = PeftModel.from_pretrained(base_model, model_dir)
        model.to(device)
        model.eval()
        print(f"Loaded lora model from {model_dir}")
        return model

    elif model_type == "qlora":
        if not model_dir:
            raise ValueError("model_dir is required for qlora model")
        if not base_model_dir:
            raise ValueError("base_model_dir is required for qlora model")
        
        config_path = os.path.join(base_model_dir, "config.json")
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
        )
        config = SegformerConfig.from_pretrained(config_path)
        base_model = SegformerForSemanticSegmentation.from_pretrained(
            base_model_dir,
            config=config,
            local_files_only=True
        )
        model = PeftModel.from_pretrained(base_model, model_dir)
        model.to(device)
        model.eval()
        print(f"Loaded qlora model from {model_dir}")
        return model

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Supported types: pretrained, finetuned_full, finetuned_lastlayer, lora, qlora")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Segformer model with various configurations")
    parser.add_argument("model_type", choices=["pretrained", "finetuned_full", "finetuned_lastlayer", "lora", "qlora"], 
                        help="Type of model to load")
    parser.add_argument("model_dir", help="Directory containing model files")
    parser.add_argument("--base-model-dir", default=None, help="Base model directory (required for lora/qlora models)")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"], help="Device to load model on (default: cuda if available, else cpu)")
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = load_model(
        model_type=args.model_type,
        device=device,
        model_dir=args.model_dir,
        base_model_dir=args.base_model_dir
    )
    print("Model loaded successfully!")
