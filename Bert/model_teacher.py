import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json
import os

from Bert.config import MODEL_NAME
from Bert.modeling_bert import BertClassifier, BertConfig

def build_teacher(path=None):
    """
    Builds the custom BERT nn.Module and loads weights from the HF repo.
    """
    model_id = path or MODEL_NAME
    
    # Textattack bert-base-uncased-SST-2 config
    config = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        num_labels=2,
    )
    
    teacher = BertClassifier(config)
    
    # Download and load the weights
    # We download the pytorch_model.bin since it doesn't have safetensors
    weights_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
    
    # Load state dict
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    
    # The custom module uses the exact same keys as the HF model.
    # The only difference might be some position_ids that are buffers, not parameters.
    # We load with strict=False to handle any missing/extra buffers.
    missing, unexpected = teacher.load_state_dict(state_dict, strict=False)
    
    # Ensure no crucial parameter is missing
    assert len([k for k in missing if "weight" in k or "bias" in k]) == 0, f"Missing crucial weights: {missing}"
    
    # We must freeze the teacher
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
        
    return teacher
