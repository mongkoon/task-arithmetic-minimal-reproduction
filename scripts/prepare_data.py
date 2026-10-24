#!/usr/bin/env python3
"""Stream reproducible small subsets of public Hugging Face datasets."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path
from task_arithmetic.config import DATA, TASKS, path_for
from task_arithmetic.data import write_rows

SOURCES = {
    "amazon": ("fancyzhx/amazon_polarity", "title", "content"),
    "yelp": ("fancyzhx/yelp_polarity", "text", None),
    "agnews": ("fancyzhx/ag_news", "text", None),
}
LABELS = {"amazon": ["negative", "positive"], "yelp": ["negative", "positive"],
          "agnews": ["world", "sports", "business", "science and technology"]}


def to_row(domain: str, item: dict) -> dict:
    _, first, second = SOURCES[domain]
    text = (str(item.get(first, "")) + (". " + str(item.get(second, "")) if second else "")).strip()
    label_id = int(item["label"])
    return {"text": text, "label": LABELS[domain][label_id]}


def sample_remote(domain: str, split: str, n: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    source = SOURCES[domain][0]
    ds = load_dataset(source, split=split, streaming=True)
    # Shuffle buffer is bounded and seed is fixed. No downloading the full dataset.
    ds = ds.shuffle(seed=seed, buffer_size=2000)
    found = []
    for item in ds:
        row = to_row(domain, item)
        if len(row["text"]) >= 15:
            found.append(row)
        if len(found) == n:
            break
    if len(found) != n:
        raise RuntimeError(f"Only {len(found)} samples from {domain}/{split}; wanted {n}")
    return found


def fixture_rows(domain: str, split: str, n: int, seed: int) -> list[dict]:
    # Offline fixture tests the plumbing only; its results must NEVER be presented as empirical evidence.
    phrases = {
        "amazon": ["This product stopped working immediately and I regret purchasing it.",
                   "This product works perfectly and exceeded my expectations."],
        "yelp": ["The restaurant served cold food and the staff were very rude.",
                 "The restaurant served delicious food and the staff were delightful."],
        "agnews": ["Diplomats met to negotiate a new international agreement.",
                   "The football team secured a dramatic championship victory.",
                   "Markets rose after the central bank announced its decision.",
                   "Researchers developed a new breakthrough in computer technology."],
    }
    rng = random.Random(f"{domain}:{split}:{seed}")
    result = []
    for i in range(n):
        idx = rng.randrange(len(phrases[domain]))
        result.append({"text": phrases[domain][idx] + f" Example {i} {domain}.",
                       "label": LABELS[domain][idx]})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["real", "fixture"], default="real")
    p.add_argument("--train", type=int, default=1600)
    p.add_argument("--val", type=int, default=200)
    p.add_argument("--test", type=int, default=250)
    p.add_argument("--lm-train", type=int, default=2500)
    p.add_argument("--lm-val", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if min(args.train, args.val, args.test, args.lm_train, args.lm_val) < 1:
        p.error("All sample counts must be positive")
    DATA.mkdir(parents=True, exist_ok=True)
    def get(domain, split, n):
        return (sample_remote if args.mode == "real" else fixture_rows)(domain, split, n, args.seed)
    # Disjoint train and validation samples are obtained by ONE stream and split by index.
    amazon = get("amazon", "train", max(args.train+args.val, args.lm_train+args.lm_val))
    agnews = get("agnews", "train", args.train+args.val)
    yelp = get("yelp", "train", args.lm_train+args.lm_val)
    for domain, task, rows, ntr, nva in [
        ("amazon", "amazon_sent", amazon, args.train, args.val),
        ("agnews", "agnews_topic", agnews, args.train, args.val),
        ("amazon", "amazon_lm", amazon, args.lm_train, args.lm_val),
        ("yelp", "yelp_lm", yelp, args.lm_train, args.lm_val),
    ]:
        training, validation = rows[:ntr], rows[ntr:ntr+nva]
        if task.endswith("_lm"):
            training = [{"text": x["text"]} for x in training]
            validation = [{"text": x["text"]} for x in validation]
        write_rows(path_for(task, "train"), training)
        write_rows(path_for(task, "val"), validation)
        print(f"{task}: train={len(training)} val={len(validation)}")
    for domain, task in [("amazon", "amazon_sent"), ("agnews", "agnews_topic"), ("yelp", "yelp_sent")]:
        test = get(domain, "test", args.test)
        write_rows(path_for(task, "test"), test)
        print(f"{task}: test={len(test)}")
    provenance = {"mode": args.mode, "seed": args.seed,
                  "train_per_supervised_task": args.train, "val_per_supervised_task": args.val,
                  "test_per_classification_task": args.test, "lm_train_reviews": args.lm_train,
                  "lm_val_reviews": args.lm_val,
                  "dataset_sources": {key: val[0] for key, val in SOURCES.items()},
                  "note": "Yelp sentiment labels exist ONLY in held-out test JSONL. Yelp LM rows contain only text. "
                          "Fixture mode is not scientific evidence."}
    (DATA / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"\nSaved {DATA}; mode={args.mode}")

if __name__ == "__main__":
    main()
