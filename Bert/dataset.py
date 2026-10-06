from datasets import load_dataset
from transformers import AutoTokenizer
from Bert.config import MODEL_NAME, MAX_LENGTH

def load_sst2():
    # Student must use teacher's tokenizer so sequences align
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    raw_dataset = load_dataset("nyu-mll/glue", "sst2")

    def tokenize_batch(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=MAX_LENGTH
        )
    
    # We want train, dev (from train), validation (glue val)
    # The reference used dataset["validation"] for evaluation, so we'll hold out a small split 
    # from train as 'dev' for checkpoint selection if we want, or just stick to the reference 
    # and use glue val for eval. The reference uses GLUE val directly.
    # To keep it aligned with the reference structure, we'll return the standard splits.
    
    tokenized = raw_dataset.map(
        tokenize_batch, batched=True, remove_columns=["sentence", "idx"]
    )
    
    return tokenized, tokenizer
