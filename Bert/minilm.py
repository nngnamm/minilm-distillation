import math
import torch
import torch.nn.functional as F

def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """[B, T, H] -> [B, num_heads, T, H // num_heads]."""
    batch, seq_len, hidden = x.shape
    return x.view(batch, seq_len, num_heads, hidden // num_heads).transpose(1, 2)

def relation_log_probs(
    left: torch.Tensor,
    right: torch.Tensor,
    num_heads: int,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """log softmax(left_h @ right_h^T / sqrt(d_h)) over the key axis."""
    left_h = split_heads(left.float(), num_heads)
    right_h = split_heads(right.float(), num_heads)
    scores = left_h @ right_h.transpose(-1, -2) / math.sqrt(left_h.size(-1))
    key_mask = attention_mask[:, None, None, :].bool()  # [B, 1, 1, T]
    scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
    return F.log_softmax(scores, dim=-1)

def masked_relation_kl(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """KL(teacher || student) for [B, R, T, T] relation distributions."""
    key_mask = attention_mask[:, None, None, :].bool()  # [B, 1, 1, T]
    teacher_probs = teacher_log_probs.exp()
    pointwise = teacher_probs * (teacher_log_probs - student_log_probs)
    kl_per_query = pointwise.masked_fill(~key_mask, 0.0).sum(dim=-1)  # [B, R, T]

    query_mask = attention_mask[:, None, :].to(kl_per_query.dtype)  # [B, 1, T]
    num_heads = kl_per_query.size(1)
    kl_per_example = (kl_per_query * query_mask).sum(dim=(1, 2))
    valid_per_example = query_mask.sum(dim=(1, 2)).clamp_min(1.0) * num_heads
    return (kl_per_example / valid_per_example).mean()
