import torch
from transformers import SegformerForSemanticSegmentation, SegformerConfig

def load_model(model_dir, config_path, device):
    config = SegformerConfig.from_pretrained(config_path)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
        device_map="cuda"
    )
    model.to(device)
    return model


def prepare_lora_model(model_path, config_path, device):
    # from transformers import SegformerForSemanticSegmentation, SegformerConfig
    from peft import get_peft_model, LoraConfig

    config = SegformerConfig.from_pretrained(config_path)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_path,
        config=config,
        local_files_only=True
    )
    peft_config = LoraConfig(
        inference_mode=False,
        r=32,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"]
    )
    lora_model = get_peft_model(model, peft_config)
    lora_model.to(device)
    return lora_model

def prepare_qlora_model(model_path, config_path, device):
    from transformers import BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig

    config = SegformerConfig.from_pretrained(config_path)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_path,
        config=config,
        local_files_only=True,
        quantization_config=bnb_config
    )
    peft_config = LoraConfig(
        inference_mode=False,
        r=32,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"]
    )
    qlora_model = get_peft_model(model, peft_config)
    qlora_model.to(device)
    return qlora_model