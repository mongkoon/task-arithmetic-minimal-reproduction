#!/usr/bin/env python3
"""Evaluate Negation, Addition, Analogy. NO target Yelp labels for tuning."""
from __future__ import annotations
import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from task_arithmetic.config import BASE_MODEL, RESULTS, VECTORS, DATA
from task_arithmetic.core import ArithmeticModel, load_vector, vector_cosine
from task_arithmetic.evaluation import evaluate_classifier

NEGATIVE_ALPHAS = [0.0, 0.25, 0.5, 1.0]
ADDITION_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SENT_ALPHAS = [0.5, 0.75, 1.0]
ANALOGY_DOMAIN_ALPHAS = [0.0, 0.1, 0.25, 0.5]
PREDECLARED_NEG_ALPHA = 0.5
PREDECLARED_DOMAIN_ALPHA = 0.25


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate_case(editor, tokenizer, coeffs, task, split, limit, batch_size, memo):
    # Cache evaluation of identical (model, task, split) to save expensive passes.
    key = (tuple(sorted((k,float(v)) for k,v in coeffs.items() if v)), task, split, limit)
    if key not in memo:
        editor.apply(coeffs)
        memo[key] = evaluate_classifier(editor.model, tokenizer, task, split,
                                        limit=limit, batch_size=batch_size)
    return memo[key]


def record(bucket, editor, tokenizer, name, coeffs, tasks, split, limit, batch_size, memo, extra=None):
    for task in tasks:
        metrics = evaluate_case(editor, tokenizer, coeffs, task, split, limit, batch_size, memo)
        row = {"experiment": bucket["experiment"], "variant": name, "split": split,
               "task": task, "coefficients": coeffs,
               "accuracy": metrics["accuracy"],
               "mean_gold_label_probability": metrics["mean_gold_label_probability"],
               "mean_gold_token_nll": metrics["mean_gold_token_nll"],
               "examples": metrics["examples"], "correct": metrics["correct"],
               "prediction_sample": metrics["prediction_sample"]}
        if extra:
            row.update(extra)
        bucket["rows"].append(row)
        print(f"  {bucket['experiment']:8} {split:4} {name:24} {task:13} accuracy={metrics['accuracy']:.3f}", flush=True)


def _score_pair(rows, name):
    found = {r["task"]:r for r in rows if r["variant"] == name and r["split"] == "val"}
    return found["amazon_sent"]["accuracy"], found["agnews_topic"]["accuracy"]


def run(editor, tokenizer, val_size, test_size, batch_size):
    memo = {}
    neg = {"experiment": "negation", "rows": [], "primary_alpha": PREDECLARED_NEG_ALPHA}
    print("\n--- Negation: theta0 - alpha * tau_amazon_sent ---")
    # Counterfactual base / target / control comparison; control task may be weak at base.
    for alpha in NEGATIVE_ALPHAS:
        record(neg, editor, tokenizer, f"neg_alpha_{alpha:g}", {"amazon_sent": -alpha},
               ["amazon_sent", "agnews_topic"], "test", test_size, batch_size, memo,
               {"alpha": alpha, "predeclared": alpha == PREDECLARED_NEG_ALPHA})
    # Specialized supervised Amazon reference helps interpret performance floor.
    record(neg, editor, tokenizer, "amazon_finetuned", {"amazon_sent": 1.0},
           ["amazon_sent", "agnews_topic"], "test", test_size, batch_size, memo)

    add = {"experiment": "addition", "rows": [], "alpha_grid": ADDITION_ALPHAS}
    print("\n--- Addition: theta0 + a*(tau_amazon_sent + tau_agnews_topic) ---")
    # Validation selection only; sealed test used once for winner + comparator rows.
    for alpha in ADDITION_ALPHAS:
        record(add, editor, tokenizer, f"add_alpha_{alpha:g}",
               {"amazon_sent": alpha, "agnews_topic": alpha},
               ["amazon_sent", "agnews_topic"], "val", val_size, batch_size, memo,
               {"alpha": alpha})
    # Pick best balance against specialized checkpoints; use normalized fraction
    # of their validation accuracies, not either test set.
    record(add, editor, tokenizer, "amazon_finetuned", {"amazon_sent": 1.0},
           ["amazon_sent", "agnews_topic"], "val", val_size, batch_size, memo)
    record(add, editor, tokenizer, "agnews_finetuned", {"agnews_topic": 1.0},
           ["amazon_sent", "agnews_topic"], "val", val_size, batch_size, memo)
    reference_a = next(r["accuracy"] for r in add["rows"] if r["variant"] == "amazon_finetuned" and r["task"] == "amazon_sent")
    reference_b = next(r["accuracy"] for r in add["rows"] if r["variant"] == "agnews_finetuned" and r["task"] == "agnews_topic")
    def normalized_pair_score(alpha):
        a,b = _score_pair(add["rows"], f"add_alpha_{alpha:g}")
        return 0.5*(a/max(reference_a,1e-6) + b/max(reference_b,1e-6))
    best_alpha = max(ADDITION_ALPHAS, key=normalized_pair_score)
    add["selected_alpha_from_validation"] = best_alpha
    add["validation_selection_score"] = normalized_pair_score(best_alpha)
    for name, coeffs in [
        ("base", {}), ("amazon_finetuned", {"amazon_sent": 1.0}),
        ("agnews_finetuned", {"agnews_topic": 1.0}),
        ("addition_selected", {"amazon_sent": best_alpha, "agnews_topic": best_alpha}),
    ]:
        record(add, editor, tokenizer, name, coeffs, ["amazon_sent", "agnews_topic"],
               "test", test_size, batch_size, memo)

    analogy = {"experiment": "analogy", "rows": [], "predeclared_domain_alpha": PREDECLARED_DOMAIN_ALPHA,
               "target_label_usage": "Yelp labels are used exclusively for final test evaluation, never for training or hyperparameter selection."}
    print("\n--- Analogy: source sentiment + (Yelp LM - Amazon LM) ---")
    for sentiment_alpha in SENT_ALPHAS:
        record(analogy, editor, tokenizer, f"source_val_{sentiment_alpha:g}",
               {"amazon_sent": sentiment_alpha}, ["amazon_sent"], "val", val_size,
               batch_size, memo, {"sentiment_alpha": sentiment_alpha})
    best_sent_alpha = max(SENT_ALPHAS, key=lambda a: next(r["accuracy"] for r in analogy["rows"]
                          if r["variant"] == f"source_val_{a:g}" and r["split"] == "val"))
    analogy["source_sentiment_alpha_chosen_on_Amazon_validation"] = best_sent_alpha
    record(analogy, editor, tokenizer, "base", {}, ["yelp_sent", "amazon_sent"],
           "test", test_size, batch_size, memo)
    for alpha in ANALOGY_DOMAIN_ALPHAS:
        coeffs = {"amazon_sent": best_sent_alpha, "yelp_lm": alpha, "amazon_lm": -alpha}
        record(analogy, editor, tokenizer, f"analogy_domain_{alpha:g}", coeffs,
               ["yelp_sent", "amazon_sent"], "test", test_size, batch_size, memo,
               {"sentiment_alpha": best_sent_alpha, "domain_alpha": alpha,
                "predeclared": alpha == PREDECLARED_DOMAIN_ALPHA,
                "exploratory": alpha != PREDECLARED_DOMAIN_ALPHA})
    # Domain-only negative control, fixed coefficient.
    record(analogy, editor, tokenizer, "domain_only_control",
           {"yelp_lm": PREDECLARED_DOMAIN_ALPHA, "amazon_lm": -PREDECLARED_DOMAIN_ALPHA},
           ["yelp_sent"], "test", test_size, batch_size, memo)
    return neg, add, analogy


def write_rows_csv(path, buckets):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["experiment", "variant", "split", "task", "alpha", "sentiment_alpha", "domain_alpha",
            "predeclared", "exploratory", "accuracy", "mean_gold_label_probability", "mean_gold_token_nll",
            "examples", "correct", "coefficients"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for bucket in buckets:
            for row in bucket["rows"]:
                w.writerow({k:json.dumps(v, ensure_ascii=False) if k=="coefficients" else v for k,v in row.items() if k in cols})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--val-size", type=int, default=80)
    p.add_argument("--test-size", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--allow-fixture", action="store_true", help="Only for pipeline debugging; non-scientific")
    args = p.parse_args()
    provenance = json.loads((DATA / "provenance.json").read_text())
    if provenance["mode"] != "real" and not args.allow_fixture:
        raise RuntimeError("Cannot report fixture results as scientific evidence. Use --allow-fixture for smoke tests ONLY.")
    if args.test_size > provenance["test_per_classification_task"]:
        p.error("test-size must not exceed prepared test size")
    if args.val_size > provenance["val_per_supervised_task"]:
        p.error("val-size must not exceed prepared validation size")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model).to(device).float()
    model.eval()
    editor = ArithmeticModel(model)
    deltas = {}
    vmetas = {}
    for task in ["amazon_sent", "agnews_topic", "amazon_lm", "yelp_lm"]:
        delta, metadata = load_vector(VECTORS / f"{task}.pt", expected_base=args.base_model)
        editor.register(task, delta)
        deltas[task] = delta
        vmetas[task] = metadata
    print(f"Device: {device}; dataset mode: {provenance['mode']}")
    neg, add, ana = run(editor, tokenizer, args.val_size, args.test_size, args.batch_size)
    similarities = {}
    keys = list(deltas)
    for i, first in enumerate(keys):
        for second in keys[i+1:]:
            similarities[f"{first}__{second}"] = vector_cosine(deltas[first], deltas[second])
    payload = {
        "status": "completed", "run_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_mode": provenance["mode"], "scientific_evidence": provenance["mode"] == "real",
        "base_model": args.base_model, "settings": vars(args),
        "dataset_provenance": provenance,
        "vector_metadata": vmetas, "cosine_similarities": similarities,
        "experiments": {b["experiment"]:b for b in (neg,add,ana)},
        "notes": ["The paper's full-scale results are NOT reproduced here; this is a small-sample replication.",
                  "Addition coefficient selected only on held-out Amazon/AG News validation data.",
                  "Analogy source coefficient selected only using Amazon validation labels; Yelp labels are used for test only.",
                  "Analogy domain=0.25 predeclared primary; other domain coefficients exploratory, not independent confirmations.",
                  "Candidate-label classification uses length-normalized conditional log-probabilities, not free generation."]}
    RESULTS.mkdir(exist_ok=True)
    save_json(RESULTS / "run_summary.json", payload)
    write_rows_csv(RESULTS / "metrics.csv", [neg,add,ana])
    print(f"\nSaved {RESULTS / 'run_summary.json'} and {RESULTS / 'metrics.csv'}")

if __name__ == "__main__":
    main()
