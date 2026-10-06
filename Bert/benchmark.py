import argparse
import os
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def run_benchmark(model_path, batch_size=128, iterations=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on {device.upper()} (Batch Size: {batch_size})...")

    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    sample_text = [
        "The cinematography in this film was absolutely breathtaking and"
        " perfectly paced."
    ] * batch_size
    inputs = tokenizer(
        sample_text, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    # GPU warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(**inputs)

    if device == "cuda":
        torch.cuda.synchronize()

    # Timed run
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(**inputs)

    if device == "cuda":
        torch.cuda.synchronize()
    total_time = time.time() - start_time

    total_samples = batch_size * iterations
    throughput = total_samples / total_time
    params = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"Model Parameters: {params:.2f}M")
    print(f"Total Time:       {total_time:.2f} s")
    print(f"Throughput:       {throughput:.2f} samples/sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="./outputs/student_distill_4L",
        help="Path to student checkpoint",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    run_benchmark(args.model_path, args.batch_size, args.iterations)
