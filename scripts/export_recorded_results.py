#!/usr/bin/env python3
"""Derive compact public tables from the archived original Colab run.

The source is results/run_summary.json. This script neither trains a model
nor infers measurements from notebook display text. Original raw artifacts
are left untouched.
"""
from __future__ import annotations
import argparse
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("amazon_sent", "agnews_topic", "amazon_lm", "yelp_lm")


def load_run(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "completed" or data.get("dataset_mode") != "real":
        raise ValueError("Only completed real-data runs can be exported")
    if not isinstance(data.get("experiments"), dict):
        raise ValueError("Missing experiment measurements")
    return data


def flatten_run(data: dict):
    metrics = []
    for key in ("negation", "addition", "analogy"):
        bucket = data["experiments"][key]
        for row in bucket["rows"]:
            metrics.append({
                "experiment": key, "split": row["split"],
                "variant": row["variant"], "task": row["task"],
                "accuracy": row["accuracy"], "correct": row["correct"],
                "examples": row["examples"],
            })
    vectors = [
        {"task": task,
         "norm_l2": data["vector_metadata"][task]["norm_l2"],
         "reconstruction_max_abs_error": data["vector_metadata"][task]["reconstruction_max_abs_error"],
         "tensor_count": data["vector_metadata"][task]["tensor_count"]}
        for task in TASKS
    ]
    cosines = []
    for key, value in data["cosine_similarities"].items():
        left, right = key.split("__", 1)
        cosines.append({"vector_a": left, "vector_b": right, "cosine_similarity": value})
    return metrics, vectors, cosines


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def export(data: dict, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    metrics, vectors, cosines = flatten_run(data)
    write_csv(dest / "metrics.csv", metrics,
              ["experiment", "split", "variant", "task", "accuracy", "correct", "examples"])
    write_csv(dest / "vectors.csv", vectors,
              ["task", "norm_l2", "reconstruction_max_abs_error", "tensor_count"])
    write_csv(dest / "cosine_similarities.csv", cosines,
              ["vector_a", "vector_b", "cosine_similarity"])
    (dest / "recorded_run.json").write_text(json.dumps({
        "source_file": "results/run_summary.json",
        "source_type": "original saved Colab result file",
        "run_utc": data["run_utc"],
        "status": data["status"],
        "base_model": data["base_model"],
        "dataset_mode": data["dataset_mode"],
        "settings": data["settings"],
        "dataset_provenance": data["dataset_provenance"],
        "vector_metadata": data["vector_metadata"],
        "cosine_similarities": data["cosine_similarities"],
        "experiment_metadata": {key: {k:v for k,v in section.items() if k != "rows"}
                                for key,section in data["experiments"].items()},
        "evaluation_count": len(metrics),
    }, indent=2, ensure_ascii=False)+'\n',encoding="utf-8")
    # Mirror the unchanged original figures into the compact published bundle.
    source_figures = dest.parent / "figures"
    published_figures = dest / "figures"
    published_figures.mkdir(exist_ok=True)
    for name in ("negation.png", "addition.png", "analogy.png"):
        src = source_figures / name
        if src.is_file():
            shutil.copyfile(src, published_figures / name)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"{entry['vector_a']} / {entry['vector_b']}" for entry in cosines]
    similarities = [entry['cosine_similarity'] for entry in cosines]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(labels, similarities)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Cosine similarity")
    ax.set_title("Task-vector directions: pairwise cosine similarities")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(published_figures / "vector_cosines.png", dpi=160)
    plt.close(fig)
    return len(metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/run_summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/published")
    args = parser.parse_args()
    rows = export(load_run(args.source), args.output)
    print(f"Exported {rows} exact evaluation measurements from {args.source} to {args.output}")


if __name__ == "__main__":
    main()
