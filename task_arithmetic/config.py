"""Task definitions and compact reproducible defaults."""
from pathlib import Path

BASE_MODEL = "google-t5/t5-small"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
CHECKPOINTS = ROOT / "artifacts" / "checkpoints"
VECTORS = ROOT / "artifacts" / "vectors"
RESULTS = ROOT / "results"

TASKS = {
    "amazon_sent": {"domain": "amazon", "objective": "sentiment", "labels": ["negative", "positive"]},
    "agnews_topic": {"domain": "agnews", "objective": "topic", "labels": ["world", "sports", "business", "science and technology"]},
    "amazon_lm": {"domain": "amazon", "objective": "lm", "labels": None},
    "yelp_lm": {"domain": "yelp", "objective": "lm", "labels": None},
    "yelp_sent": {"domain": "yelp", "objective": "sentiment", "labels": ["negative", "positive"], "eval_only": True},
}


def path_for(task: str, split: str) -> Path:
    return DATA / f"{task}_{split}.jsonl"


def prompt(task: str, text: str) -> str:
    objective = TASKS[task]["objective"]
    if objective == "sentiment":
        return f"sentiment: {text}"
    if objective == "topic":
        return f"topic: {text}"
    raise ValueError(f"LM rows are tokenized separately: {task}")
