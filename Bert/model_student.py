import torch
from Bert.modeling_bert import BertClassifier, BertConfig
from Bert.config import STUDENT_HIDDEN_SIZE, STUDENT_HEADS, STUDENT_INTERMEDIATE, NUM_LABELS

def build_student(num_layers=4):
    """
    Builds a randomly initialized student BERT nn.Module.
    """
    config = BertConfig(
        vocab_size=30522,
        hidden_size=STUDENT_HIDDEN_SIZE,
        num_hidden_layers=num_layers,
        num_attention_heads=STUDENT_HEADS,
        intermediate_size=STUDENT_INTERMEDIATE,
        num_labels=NUM_LABELS,
    )
    
    student = BertClassifier(config)
    return student
