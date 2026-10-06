# MiniLM Task-Specific Distillation for SST-2

This repository implements relation-based Knowledge Distillation (MiniLM) to compress a 109M parameter BERT classifier into a lightweight 19.2M parameter student (~82% parameter reduction) while achieving **12,618 samples/second** inference throughput on a single NVIDIA T4 GPU.

## Core Architecture & Innovation

Standard knowledge distillation matches output logits or intermediate hidden layers directly. Matching hidden states requires the student and teacher to share identical hidden dimensions ($d = 768$) unless parameter-heavy linear projection layers are introduced.

MiniLM bypasses dimension alignment by transferring self-attention relations from the teacher's final layer:
1. **Self-Attention Distribution Transfer:** Captures query-key attention interactions:
   $$A = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$
2. **Value-Relation Transfer:** Captures semantic token relationships using value-vector dot products:
   $$VR = \text{Softmax}\left(\frac{VV^T}{\sqrt{d_k}}\right)$$

Because both $A$ and $VR$ matrices have dimensions (`seq_len` × `seq_len`), the hidden dimension mathematically cancels out. This enables distilling directly from a 12-layer, 768-dimensional teacher to a 4-layer, 384-dimensional student without projection matrices, resulting in massive computational speedups.

---

## Benchmark Results (SST-2 Sentiment Analysis)

| Metric | Teacher Baseline | MiniLM Student (This Repo) |
| :--- | :--- | :--- |
| **Layers** | 12 | 4 |
| **Hidden Size** | 768 | 384 |
| **Parameters** | 109.5M | **19.2M (-82.5%)** |
| **SST-2 Accuracy** | 92.4% | 81.10% |
| **Throughput (T4)**| ~3,000 samples/s | **12,618 samples/s** |

**Business Application:** This pipeline is optimized for edge deployment and high-volume text processing where latency and compute costs are primary constraints, successfully trading an ~11% accuracy drop for a >4x speedup and massive memory reduction.

---

## Repository Structure

```text
├── BERT/
│   ├── benchmark.py         # Inference throughput & latency evaluation
│   ├── config.py            # Hyperparameters and model configuration
│   ├── dataset.py           # SST-2 data loading and tokenization
│   ├── minilm.py            # Core MiniLM relation-extraction logic
│   ├── model_student.py     # 4-layer student architecture definition
│   ├── model_teacher.py     # 12-layer teacher architecture wrapper
│   ├── modeling_bert.py     # Custom BERT modifications
│   ├── train_distill.py     # Multi-GPU Accelerate training loop
│   └── utils.py             # Evaluation and metric calculation
├── outputs/                 # Saved student model weights (Ignored in git)
├── infer.py                 # Interactive command-line inference script
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
