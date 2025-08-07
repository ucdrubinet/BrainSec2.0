#merge LoRA weights into a base Segformer model for easier export to ONNX
from transformers import SegformerForSemanticSegmentation,SegformerConfig
from peft import PeftModel
config = SegformerConfig.from_pretrained("/home/ajinkya/BS_two/models/models/segformer_JC/config.json")
base_model = SegformerForSemanticSegmentation.from_pretrained(
    "/home/ajinkya/BS_two/models/models/segformer_JC",  # Directory containing model files
    config=config,                           # Load the custom config
    local_files_only=True                  # Ensure only local files are used
)
lora_model = PeftModel.from_pretrained(base_model, "/home/ajinkya/BS_two/src/ft_models/finetuned_lora")
# Merge LoRA into the base weights
merged_model = lora_model.merge_and_unload()
# Save clean base model
merged_model.save_pretrained("/home/ajinkya/BS_two/src/ft_models/merged_lora_segformer")
