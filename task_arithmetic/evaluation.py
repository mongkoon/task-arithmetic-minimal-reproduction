"""Constrained generation-free scoring of text-to-text label candidates.

No arbitrary answer parsing: evaluate the normalized conditional log likelihood
of each label (including EOS), then choose the largest score.
"""
from __future__ import annotations
from collections import Counter
import math
import torch
from torch.nn import functional as F
from .config import TASKS, prompt, path_for
from .data import read_rows


@torch.inference_mode()
def evaluate_classifier(model, tokenizer, task: str, split: str, *, limit: int,
                        batch_size: int = 8, max_input_length: int = 128):
    if TASKS[task]["objective"] == "lm":
        raise ValueError("Only supervised classification tasks can be evaluated")
    rows = read_rows(path_for(task, split))[:limit]
    if not rows:
        raise ValueError(f"Empty {task}/{split}")
    labels = TASKS[task]["labels"]
    device = next(model.parameters()).device
    label_ids = tokenizer(text_target=labels, padding=True, return_tensors="pt")["input_ids"].to(device)
    label_mask = label_ids.ne(tokenizer.pad_token_id)
    n_labels = len(labels)
    correct, count = 0, 0
    gold_probs, gold_token_nll = [], []
    predictions = []
    model.eval()
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        enc = tokenizer([prompt(task, x["text"]) for x in batch], padding=True,
                        truncation=True, max_length=max_input_length, return_tensors="pt").to(device)
        bsz = len(batch)
        ids = enc["input_ids"].repeat_interleave(n_labels, dim=0)
        masks = enc["attention_mask"].repeat_interleave(n_labels, dim=0)
        ids_y = label_ids.repeat(bsz, 1)
        mask_y = label_mask.repeat(bsz, 1)
        # Decoder starts with pad token, then teacher-forces every label token.
        decoder_inputs = model._shift_right(ids_y)
        logits = model(input_ids=ids, attention_mask=masks,
                       decoder_input_ids=decoder_inputs, use_cache=False).logits
        token_logp = F.log_softmax(logits, dim=-1).gather(-1, ids_y.unsqueeze(-1)).squeeze(-1)
        sequence_score = (token_logp * mask_y).sum(dim=-1) / mask_y.sum(dim=-1).clamp(min=1)
        scores = sequence_score.reshape(bsz, n_labels)
        probabilities = scores.softmax(dim=-1)
        predicted = scores.argmax(dim=-1).cpu().tolist()
        truth = [labels.index(row["label"]) for row in batch]
        predictions.extend([{"gold": labels[t], "predicted": labels[p]} for t,p in zip(truth,predicted)])
        correct += sum(a == b for a,b in zip(predicted,truth))
        count += bsz
        gold_probs.extend(probabilities[range(bsz),truth].cpu().tolist())
        gold_token_nll.extend((-scores[range(bsz),truth]).cpu().tolist())
    return {
        "accuracy": correct/count,
        "mean_gold_label_probability": sum(gold_probs)/count,
        "mean_gold_token_nll": sum(gold_token_nll)/count,
        "examples": count, "correct": correct,
        "prediction_sample": predictions[:8],
        "evaluation_method": "length-normalized teacher-forced candidate label log-likelihood (with EOS)",
    }
