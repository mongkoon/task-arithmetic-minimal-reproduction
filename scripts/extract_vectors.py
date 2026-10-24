#!/usr/bin/env python3
"""Export four task vectors and verify exact lambda=1 reconstruction."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM
from task_arithmetic.core import ArithmeticModel, extract_vector, save_vector, vector_norm
from task_arithmetic.config import BASE_MODEL, CHECKPOINTS, VECTORS, TASKS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--task", choices=[k for k, v in TASKS.items() if not v.get("eval_only")], default=None,
                   help="Export just one vector; default all four")
    args = p.parse_args()
    tasks = [args.task] if args.task else [k for k, v in TASKS.items() if not v.get("eval_only")]
    base = AutoModelForSeq2SeqLM.from_pretrained(args.base_model).cpu().float()
    editor = ArithmeticModel(base)
    VECTORS.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        ckpt = CHECKPOINTS / task
        if not (ckpt / "train_metadata.json").exists():
            raise FileNotFoundError(f"No training metadata for {task}: run scripts/train_task.py first")
        meta = json.loads((ckpt / "train_metadata.json").read_text())
        if meta["base_model"] != args.base_model:
            raise ValueError(f"{task} trained from {meta['base_model']}, not {args.base_model}")
        tuned = AutoModelForSeq2SeqLM.from_pretrained(ckpt).cpu().float()
        delta = extract_vector(base, tuned)
        editor.register(task, delta)
        editor.apply({task: 1.0})
        reconstruction_error = editor.max_abs_error_vs(tuned)
        editor.restore()
        if reconstruction_error > 5e-5:
            raise RuntimeError(f"Reconstruction failed for {task}: max error {reconstruction_error}")
        metadata = {"base_model": args.base_model, "task": task, "checkpoint": str(ckpt),
                    "norm_l2": vector_norm(delta), "lambda_1_reconstruction_max_error": reconstruction_error,
                    "parameters_counted_once": len(delta)}
        save_vector(delta, VECTORS / f"{task}.pt", metadata)
        (VECTORS / f"{task}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"{task}: vector norm={metadata['norm_l2']:.5f}; error={reconstruction_error:.9g}")
        del tuned, delta
    print(f"Exported vectors to {VECTORS}")

if __name__ == "__main__":
    main()
