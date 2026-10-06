import argparse
import torch
import torch.nn.functional as F
from transformers import DataCollatorWithPadding, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput

from Bert.config import NUM_EPOCHS, TEACHER_DIR, student_dir, ALPHA_CE, ALPHA_ATTENTION, ALPHA_VALUE, NUM_RELATION_HEADS
from Bert.dataset import load_sst2
from Bert.model_student import build_student
from Bert.model_teacher import build_teacher
from Bert.utils import compute_metrics, count_parameters, make_training_args, save_history
from Bert.minilm import relation_log_probs, masked_relation_kl

class MiniLMTrainer(Trainer):
    def __init__(self, *args, teacher, alpha_ce, alpha_attention, alpha_value, num_relation_heads, **kwargs):
        super().__init__(*args, **kwargs)
        # Teacher is already on CPU/GPU depending on initialization, 
        # but Trainer handles placing it on the right device usually.
        # Let's ensure it's on the right device during compute_loss.
        self.teacher = teacher
        self.alpha_ce = alpha_ce
        self.alpha_attention = alpha_attention
        self.alpha_value = alpha_value
        self.num_relation_heads = num_relation_heads

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Move teacher to device if needed (first pass)
        device = inputs["attention_mask"].device
        if self.teacher.bert.embeddings.word_embeddings.weight.device != device:
            self.teacher = self.teacher.to(device)
            
        student_out = model(**inputs)
        ce_loss = student_out.loss
        student_qkv = student_out.qkv[-1] # Last layer

        with torch.no_grad():
            teacher_out = self.teacher(**inputs)
            teacher_qkv = teacher_out.qkv[-1] # Last layer

        attention_mask = inputs["attention_mask"]

        # Compute MiniLM losses in fp32
        with torch.autocast(device_type=attention_mask.device.type, enabled=False):
            # Teacher tensors need to be explicitly cast if they were fp16 from some autocast context, 
            # though here they are inside the autocast=False block.
            t_attn = relation_log_probs(teacher_qkv.query.detach(), teacher_qkv.key.detach(), self.num_relation_heads, attention_mask)
            t_val = relation_log_probs(teacher_qkv.value.detach(), teacher_qkv.value.detach(), self.num_relation_heads, attention_mask)
            
            s_attn = relation_log_probs(student_qkv.query, student_qkv.key, self.num_relation_heads, attention_mask)
            s_val = relation_log_probs(student_qkv.value, student_qkv.value, self.num_relation_heads, attention_mask)

            attention_kl = masked_relation_kl(t_attn, s_attn, attention_mask)
            value_relation_kl = masked_relation_kl(t_val, s_val, attention_mask)

        loss = (self.alpha_ce * ce_loss) + (self.alpha_attention * attention_kl) + (self.alpha_value * value_relation_kl)

        if return_outputs:
            # We must return a SequenceClassifierOutput or tuple
            return loss, SequenceClassifierOutput(loss=loss, logits=student_out.logits)
        return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    
    output_dir = student_dir("distill", args.layers)

    dataset, tokenizer = load_sst2()
    
    print("Loading teacher...")
    teacher = build_teacher(TEACHER_DIR if is_trained(TEACHER_DIR) else None)
    
    print("Building student...")
    student = build_student(num_layers=args.layers)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # In HF Trainer, the evaluation splits could be large, but GLUE val is small.
    # To avoid bias we split train to create a dev set.
    # But since the reference uses dataset["validation"] for evaluation, we will too.
    # (We are copying the layout and logic of the reference).
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    trainer = MiniLMTrainer(
        model=student,
        args=make_training_args(output_dir, epochs=args.epochs),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        teacher=teacher,
        alpha_ce=ALPHA_CE,
        alpha_attention=ALPHA_ATTENTION,
        alpha_value=ALPHA_VALUE,
        num_relation_heads=NUM_RELATION_HEADS,
    )

    print("Starting distillation...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_history(trainer, output_dir)

    metrics = trainer.evaluate()
    print(f"[student_distill_{args.layers}L] accuracy={metrics['eval_accuracy']:.4f} "
          f"params={count_parameters(student) / 1e6:.1f}M")

if __name__ == "__main__":
    from Bert.config import is_trained
    main()
