from datasets import load_dataset
from transformers import AutoTokenizer
from Bert.config import MODEL_NAME, MAX_LENGTH


def load_sst2():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    raw_dataset = load_dataset("nyu-mll/glue", "sst2")

    def tokenize_batch(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=MAX_LENGTH)

    tokenized = raw_dataset.map(
        tokenize_batch, batched=True, remove_columns=["sentence", "idx"]
    )

    return tokenized, tokenizer
