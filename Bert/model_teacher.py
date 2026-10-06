import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json
import os

from Bert.config import MODEL_NAME
from Bert.modeling_bert import BertClassifier, BertConfig


def build_teacher(path=None):
    model_id = path or MODEL_NAME
    config = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        num_labels=2,
    )

    teacher = BertClassifier(config)
    weights_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    missing, unexpected = teacher.load_state_dict(state_dict, strict=False)

    assert len([k for k in missing if "weight" in k or "bias" in k]) == 0, (
        f"Missing crucial weights: {missing}"
    )

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    return teacher
