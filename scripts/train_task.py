#!/usr/bin/env python3
"""Fine-tune one full-weight T5-small checkpoint from the SAME initialization."""
from __future__ import annotations
import argparse
import json
import random
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from task_arithmetic.config import BASE_MODEL, TASKS, CHECKPOINTS, path_for
from task_arithmetic.data import examples, collate


def validate(model, batches, device, max_batches=15):
    model.eval()
    losses = []
    with torch.inference_mode():
        for i, batch in enumerate(batches):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            losses.append(float(model(**batch).loss))
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=[k for k, x in TASKS.items() if not x.get("eval_only")])
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--max-steps", type=int, default=180)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--chunk-tokens", type=int, default=96)
    p.add_argument("--max-input-length", type=int, default=128)
    p.add_argument("--max-target-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.max_steps <= 0 or args.batch_size <= 0 or args.grad_accum <= 0:
        p.error("max-steps, batch-size and grad-accum must be positive")
    folder = CHECKPOINTS / args.task
    if folder.exists() and not args.overwrite:
        raise FileExistsError(f"{folder} already exists. Use --overwrite to retrain intentionally.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train = examples(args.task, "train", tokenizer, args.chunk_tokens)
    val = examples(args.task, "val", tokenizer, args.chunk_tokens)
    collator = partial(collate, tokenizer=tokenizer, max_input_length=args.max_input_length,
                       max_target_length=args.max_target_length)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model).to(device)
    model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=min(10, args.max_steps//10),
                                                 num_training_steps=args.max_steps)
    model.train()
    total_loss = 0.0
    step = 0
    batches = iter(loader)
    opt.zero_grad(set_to_none=True)
    while step < args.max_steps:
        for _ in range(args.grad_accum):
            try:
                batch = next(batches)
            except StopIteration:
                batches = iter(loader)
                batch = next(batches)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            (loss / args.grad_accum).backward()
            total_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step == 1 or step % 25 == 0 or step == args.max_steps:
            print(f"[{args.task}] {step}/{args.max_steps}, last_loss={float(loss):.3f}", flush=True)
    val_loss = validate(model, val_loader, device)
    model.config.use_cache = True
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder, safe_serialization=True)
    tokenizer.save_pretrained(folder)
    metadata = {**vars(args), "device": str(device), "train_examples": len(train),
                "validation_examples": len(val), "validation_loss": val_loss,
                "mean_microbatch_loss": total_loss/(args.max_steps*args.grad_accum),
                "note": "Full-parameter fine-tuning; all tasks independently initialized from the SAME base-model."}
    (folder / "train_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {folder}, validation_loss={val_loss:.4f}")

if __name__ == "__main__":
    main()
