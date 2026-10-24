"""Verify the archived raw run and compact derived summaries."""
import csv
import json
from pathlib import Path
from scripts.export_recorded_results import load_run, flatten_run, export

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def test_original_run_has_complete_measurements():
    run = load_run(RESULTS / "run_summary.json")
    assert run["status"] == "completed"
    assert run["base_model"] == "google-t5/t5-small"
    assert run["settings"]["seed"] == 42
    assert run["dataset_provenance"]["mode"] == "real"
    assert run["experiments"]["addition"]["selected_alpha_from_validation"] == 1.0
    assert run["experiments"]["analogy"]["source_sentiment_alpha_chosen_on_Amazon_validation"] == 0.5
    rows, vectors, cosines = flatten_run(run)
    assert len(rows) == 46
    assert len(vectors) == 4
    assert len(cosines) == 6
    observed = {(x['experiment'], x['variant'], x['task'], x['split']): x for x in rows}
    assert observed['negation', 'neg_alpha_0.5', 'amazon_sent', 'test']['correct'] == 169
    assert observed['addition', 'addition_selected', 'agnews_topic', 'test']['correct'] == 144
    assert observed['analogy', 'analogy_domain_0.25', 'yelp_sent', 'test']['correct'] == 212
    for x in rows:
        assert x['accuracy'] == x['correct'] / x['examples']


def test_raw_metrics_match_full_json():
    run = load_run(RESULTS / 'run_summary.json')
    rows, _, _ = flatten_run(run)
    with (RESULTS / 'metrics.csv').open(newline='', encoding='utf-8') as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 46
    index = {(x['experiment'],x['split'],x['variant'],x['task']):x for x in rows}
    for csv_row in csv_rows:
        key = tuple(csv_row[k] for k in ('experiment','split','variant','task'))
        expected = index[key]
        assert float(csv_row['accuracy']) == expected['accuracy']
        assert int(csv_row['correct']) == expected['correct']
        assert int(csv_row['examples']) == expected['examples']


def test_published_summaries_derive_from_original(tmp_path):
    run = load_run(RESULTS / 'run_summary.json')
    assert export(run,tmp_path) == 46
    for name in ['metrics.csv', 'vectors.csv', 'cosine_similarities.csv', 'recorded_run.json']:
        assert (tmp_path/name).read_bytes() == (RESULTS/'published'/name).read_bytes()


def test_archived_figures_available():
    for name in ('negation', 'addition', 'analogy'):
        assert (RESULTS/'figures'/f'{name}.png').stat().st_size > 1000


def test_original_artifact_checksums():
    import hashlib
    provenance=json.loads((RESULTS/'provenance.json').read_text(encoding='utf-8'))
    for relative_path, expected_hash in provenance['sha256'].items():
        assert hashlib.sha256((RESULTS/relative_path).read_bytes()).hexdigest()==expected_hash
