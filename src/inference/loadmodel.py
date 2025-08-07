import torch
from transformers import SegformerForSemanticSegmentation, SegformerConfig, BitsAndBytesConfig
from peft import PeftModel
import os

def load_model(
    model_type,
    device,
    model_dir=None
):
    """
    Load a Segformer pretrained/finetuned model based on the specified type.
    """

    if model_dir is None and model_type != "lora" and model_type != "qlora":
        raise ValueError("model_dir must be provided for all model types except finetuned_lastlayer.")
    config_path = os.path.join(model_dir, "config.json") if model_dir else None
    config = SegformerConfig.from_pretrained(config_path) if config_path else None


    if model_type == "pretrained":
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_dir,
            config=config,
            local_files_only=True
        )
        model.to(device)
        model.eval()
        return model

    elif model_type == "finetuned_full":
        # config.json is in the same directory as the .pth file
        config = SegformerConfig.from_pretrained(config_path)
        model = SegformerForSemanticSegmentation(config)
        state_dict_path="/home/ajinkya/segmentation/BrainSec2.0/models/ft_models/finetuned_full/finetuned_full.pth"
        state_dict = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    elif model_type == "finetuned_lastlayer":
        config = SegformerConfig.from_pretrained(config_path)
        model = SegformerForSemanticSegmentation(config)
        state_dict_path="/home/ajinkya/segmentation/BrainSec2.0/models/ft_models/finetuned_lastlayer/finetuned_lastlayer.pth"
        state_dict = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    elif model_type == "lora":

        config = SegformerConfig.from_pretrained("/home/ajinkya/BS_two/models/models/segformer_JC/config.json")
        base_model = SegformerForSemanticSegmentation.from_pretrained(
            "/home/ajinkya/BS_two/models/models/segformer_JC",  # Directory containing model files
            config=config,                           # Load the custom config
            local_files_only=True                  # Ensure only local files are used
        )
        adapter_path="/home/ajinkya/segmentation/BrainSec2.0/models/ft_models/finetuned_lora"
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.to(device)
        model.eval()
        return model

    elif model_type == "qlora":

        if bnb_config is None:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        config = SegformerConfig.from_pretrained("/home/ajinkya/BS_two/models/models/segformer_JC/config.json")
        base_model = SegformerForSemanticSegmentation.from_pretrained(
            "/home/ajinkya/segmentation/BrainSec2.0/models/segformer_JC",
            config=config,
            quantization_config=bnb_config,
            local_files_only=True
        )
        adapter_path="/home/ajinkya/segmentation/BrainSec2.0/models/ft_models/finetuned_qlora"
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.to(device)
        model.eval()
        return model

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        model_type="pretrained",
        device=device,
        model_dir="/home/ajinkya/BS_two/models/models/segformer_JC"
    )
