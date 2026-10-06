import os

MODEL_NAME = "textattack/bert-base-uncased-SST-2"
NUM_LABELS = 2
MAX_LENGTH = 128
SEED = 42
NUM_STUDENT_LAYERS = 4
STUDENT_HIDDEN_SIZE = 384
STUDENT_HEADS = 12
STUDENT_INTERMEDIATE = 1536
NUM_RELATION_HEADS = 12

NUM_EPOCHS = 5
LEARNING_RATE = 2e-4
BATCH_SIZE = 32

ALPHA_CE = 1.0
ALPHA_ATTENTION = 1.0
ALPHA_VALUE = 1.0

OUTPUT_ROOT = "./outputs"
TEACHER_DIR = os.path.join(OUTPUT_ROOT, "teacher")

def student_dir(mode, num_layers):
    return os.path.join(OUTPUT_ROOT, f"student_{mode}_{num_layers}L")

def is_trained(directory):
    return os.path.exists(os.path.join(directory, "config.json"))
