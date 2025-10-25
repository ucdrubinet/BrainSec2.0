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

    if model_type == "pretrained":
        config_path = os.path.join(model_dir, "config.json") if model_dir else None
        config = SegformerConfig.from_pretrained(config_path) if config_path else None
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_dir,
            config=config,
            local_files_only=True
        )
        model.to(device)
        model.eval()
        return model

    elif model_type == "finetuned_full":
        config_path = os.path.join(model_dir, "config.json") if model_dir else None
        config = SegformerConfig.from_pretrained(config_path) if config_path else None
        model = SegformerForSemanticSegmentation(config)
        state_dict_path = os.path.join(model_dir, "finetuned_full.pth")
        state_dict = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    elif model_type == "finetuned_lastlayer":
        config_path = os.path.join(model_dir, "config.json") if model_dir else None
        config = SegformerConfig.from_pretrained(config_path) if config_path else None
        model = SegformerForSemanticSegmentation(config)
        state_dict_path = os.path.join(model_dir, "finetuned_lastlayer.pth")
        state_dict = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    elif model_type == "lora":
        config = SegformerConfig.from_pretrained("/home/ajinkya/segmentation/BrainSec2.0/models/segformer_pretrained/config.json")
        model_dir1 = "/home/ajinkya/segmentation/BrainSec2.0/models/segformer_pretrained"
        base_model = SegformerForSemanticSegmentation.from_pretrained(
            model_dir1,
            config=config,
            local_files_only=True
        )
        adapter_path = os.path.join(model_dir)
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.to(device)
        model.eval()
        return model

    elif model_type == "qlora":

        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
        )
        config = SegformerConfig.from_pretrained("/home/ajinkya/segmentation/BrainSec2.0/models/segformer_pretrained/config.json")
        model_dir1 = "/home/ajinkya/segmentation/BrainSec2.0/models/segformer_pretrained"
        base_model = SegformerForSemanticSegmentation.from_pretrained(
            model_dir1,
            config=config,
            local_files_only=True
        )
        adapter_path=model_dir
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.to(device)
        model.eval()
        return model

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        model_type="pretrained",
        device=device,
        model_dir="/home/ajinkya/BS_two/models/models/segformer_pretrained"
    )
