"""Plotting smoke test with synthetic data, not empirical measurements."""
import json
from scripts import plot_results


def _row(variant, task, score):
    return {"variant": variant, "task": task, "split": "test", "accuracy": score}


def test_plot_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(plot_results, "RESULTS", tmp_path)
    neg = [
        _row(f"neg_alpha_{alpha:g}", task, 0.6 - alpha * 0.1)
        for alpha in [0.0, 0.25, 0.5, 1.0]
        for task in ["amazon_sent", "agnews_topic"]
    ]
    add = [
        _row(variant, task, 0.5)
        for variant in ["base", "amazon_finetuned", "agnews_finetuned", "addition_selected"]
        for task in ["amazon_sent", "agnews_topic"]
    ]
    analogy = [
        _row(f"analogy_domain_{alpha:g}", task, 0.5 + 0.1 * alpha)
        for alpha in [0.0, 0.1, 0.25, 0.5]
        for task in ["yelp_sent", "amazon_sent"]
    ]
    payload = {"experiments": {
        "negation": {"rows": neg},
        "addition": {"rows": add, "selected_alpha_from_validation": 0.5},
        "analogy": {"rows": analogy},
    }}
    (tmp_path / "run_summary.json").write_text(json.dumps(payload))
    plot_results.main()
    for name in ["negation", "addition", "analogy"]:
        assert (tmp_path / "figures" / f"{name}.png").stat().st_size > 1000
