import json
import os
import numpy as np
import torch
import evaluate
from transformers import TrainingArguments
from Bert.config import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, SEED

_accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    return _accuracy.compute(predictions=predictions, references=labels)


def make_training_args(output_dir, epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=128,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to="none",
        remove_unused_columns=False,
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def save_history(trainer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
