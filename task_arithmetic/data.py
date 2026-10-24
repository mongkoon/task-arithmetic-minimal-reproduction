"""Dataset IO and true consecutive-chunk text language-modeling pairs."""
from __future__ import annotations
import json
from pathlib import Path
from .config import TASKS, path_for, prompt


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts/prepare_data.py first")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def language_model_pairs(rows: list[dict], tokenizer, chunk_tokens: int = 96) -> list[dict]:
    """Make consecutive fixed-length token chunks (input chunk i -> target chunk i+1).

    This is a small-data proxy of the paper's 128-token consecutive-chunk LM objective.
    Domain data are unlabeled; do not mix evaluation/test reviews into training.
    """
    if chunk_tokens < 8:
        raise ValueError("chunk_tokens must be >=8")
    tokens = []
    for item in rows:
        ids = tokenizer(item["text"], add_special_tokens=False, truncation=False)["input_ids"]
        tokens.extend(ids)
        tokens.append(tokenizer.eos_token_id)
    chunks = [tokens[i:i + chunk_tokens] for i in range(0, len(tokens), chunk_tokens)]
    chunks = [chunk for chunk in chunks if len(chunk) == chunk_tokens]
    out = []
    for source, target in zip(chunks[:-1], chunks[1:]):
        out.append({"input": "continue: " + tokenizer.decode(source, skip_special_tokens=True),
                    "target": tokenizer.decode(target, skip_special_tokens=True)})
    if not out:
        raise ValueError("Not enough LM text for at least two full chunks. Increase --lm-train.")
    return out


def examples(task: str, split: str, tokenizer=None, chunk_tokens: int = 96) -> list[dict]:
    rows = read_rows(path_for(task, split))
    if TASKS[task]["objective"] == "lm":
        if tokenizer is None:
            raise ValueError("Tokenizer needed for LM chunk formation")
        return language_model_pairs(rows, tokenizer, chunk_tokens)
    return [{"input": prompt(task, item["text"]), "target": item["label"]} for item in rows]


def collate(batch: list[dict], tokenizer, max_input_length=128, max_target_length=32):
    import torch
    enc = tokenizer([r["input"] for r in batch], padding=True, truncation=True,
                    max_length=max_input_length, return_tensors="pt")
    targets = tokenizer(text_target=[r["target"] for r in batch], padding=True,
                        truncation=True, max_length=max_target_length, return_tensors="pt")
    labels = targets["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    enc["labels"] = labels
    return enc
