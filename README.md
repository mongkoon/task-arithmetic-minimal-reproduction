# Task Arithmetic: Negation, Addition, and Analogy

**English** | [한국어](README.ko.md)

An independent, small-scale investigation of **task vectors in model weight space** using [`google-t5/t5-small`](https://huggingface.co/google-t5/t5-small), based on [*Editing Models with Task Arithmetic* (Ilharco et al., ICLR 2023)](https://arxiv.org/abs/2212.04089).

This repository includes **one completed Google Colab experiment**: the executable notebook with its recorded outputs, the **original evaluation JSON and CSV downloaded from Colab**, original PNG figures, executable training/evaluation scripts, and bilingual analysis. The project is a limited implementation, **not an exact reproduction of the entire paper**.

**[Results and interpretation](results/README.md) · [결과 및 해석](results/README.ko.md) · [Executed notebook](notebooks/run_in_colab.ipynb)**

## Research question and operations

For the common pretrained initialization $\theta_0$ and separately fine-tuned checkpoints $\theta_t$, define a task vector as:

$$\tau_t = \theta_t - \theta_0.$$

| Operation | Applied weight edit | Target evaluation |
|---|---|---|
| **Negation** | $\theta_0 - \alpha\tau_{\mathrm{AmazonSent}}$ | Amazon sentiment; AG News as a control |
| **Addition** | $\theta_0 + \alpha(\tau_{\mathrm{AmazonSent}}+\tau_{\mathrm{AGNews}})$ | Both classification tasks in a single model |
| **Analogy** | $\theta_0+\beta\tau_{\mathrm{AmazonSent}}+\gamma(\tau_{\mathrm{YelpLM}}-\tau_{\mathrm{AmazonLM}})$ | Yelp sentiment without using labeled Yelp data for tuning |

All edits start independently from the original base model parameters. The experiment also checks whether adding a task vector at coefficient 1 reconstructs the corresponding fine-tuned checkpoint.

## Recorded outcomes

One run with **seed 42**, four separately fine-tuned checkpoints, and **250 held-out test examples per task** yielded the following results:

| Operation | Observation | Interpretation |
|---|---|---|
| **Negation** ($\alpha=0.5$) | Amazon accuracy **56.0% → 67.6%**; AG News control **30.8% → 26.0%** | Targeted forgetting **not demonstrated** |
| **Addition** ($\alpha=1.0$) | Amazon **84.8%**, AG News **57.6%**, vs base **56.0%** and **30.8%** | Both improved over base; below respective specialists **87.6%** and **77.2%** |
| **Analogy** ($\beta=0.5$, $\gamma=0.25$) | Yelp **84.0% → 84.8%** vs source-only transfer | Net gain of **+0.8 percentage points (2/250)**; inconclusive in one run |

![Addition: original Colab figure](results/figures/addition.png)

The conclusion is **mixed**, not a confirmation of all three effects. A parameter difference is numerically computable and checkpoint reconstruction works, but that alone does not establish independent, linearly separable task capabilities. See the [full results analysis](results/README.md) for baselines, coefficient selection, vector norms, and limitations.

## Experiment configuration

| Setting | Recorded value |
|---|---|
| Base model | `google-t5/t5-small` |
| Fine-tuned tasks | `amazon_sent`, `agnews_topic`, `amazon_lm`, `yelp_lm` |
| Datasets | [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity), [AG News](https://huggingface.co/datasets/fancyzhx/ag_news), [Yelp Polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity) |
| Training | 180 optimization steps per checkpoint; batch size 4; gradient accumulation 2; learning rate `1e-4` |
| Supervised data | 1,600 training and 200 validation examples per task |
| Domain LM data | 2,500 unlabeled reviews per domain for training; 300 for validation |
| Evaluation | 80 examples per validation task for coefficient selection; 250 examples per test task |
| Classifier | Length-normalized conditional label log-probability (EOS included) |
| Domain LM approximation | Consecutive 96-token chunk pairs in a seq2seq proxy objective |
| Recorded device | NVIDIA Tesla T4 (Google Colab) |

Yelp sentiment labels are used for **final held-out evaluation only**, not fine-tuning or coefficient selection. The simplified LM proxy differs from the original paper's precise language-modeling procedure.

## Repository structure

```text
README.md                          # English project overview
README.ko.md                       # Korean project overview
notebooks/run_in_colab.ipynb       # Executed, explanatory Python notebook
scripts/                           # CLI pipeline and result-table exporter
task_arithmetic/                   # Shared model/data/evaluation code
tests/                             # Unit and archived-data integrity checks
results/
  README.md                        # English results, tables and analysis
  README.ko.md                     # Korean results, tables and analysis
  run_summary.json                 # ORIGINAL full Colab JSON (complete recorded run)
  metrics.csv                      # ORIGINAL detailed Colab metrics (46 rows)
  figures/
    negation.png                    # ORIGINAL Colab figure
    addition.png                    # ORIGINAL Colab figure
    analogy.png                     # ORIGINAL Colab figure
  published/                       # Derived compact CSV/metadata summaries
    metrics.csv
    vectors.csv
    cosine_similarities.csv
    recorded_run.json
    figures/                      # Original-chart mirrors and derived cosine figure
artifacts/checkpoints/             # Local-only large model checkpoints
data/processed/                     # Local-only prepared data
```

**Provenance:** `results/run_summary.json`, `results/metrics.csv`, and the three PNG files in `results/figures/` are the **original archived Colab outputs**, not back-filled estimates from notebook text. The `results/published/` directory contains compact derivatives of the original JSON. The full trained model checkpoints and complete example-level predictions were not supplied and are not claimed to be available.

## Running and verifying

With Python 3.10–3.12 and a GPU recommended:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
bash scripts/run_all.sh
```

The [Colab notebook](notebooks/run_in_colab.ipynb) shows the training loop, parameter subtraction, all three weight edits, evaluation, and plotting as explicit code cells. The equivalent CLI entry points are located in `scripts/`. A new execution requires model/dataset downloads; it may **overwrite the versioned `results/` files** in the local checkout. Archive previous results before intentionally publishing another run.

To regenerate **only the derived compact tables** from the archived original JSON, without training or model downloads:

```bash
python scripts/export_recorded_results.py
```

The experiment's original charts are retained as-is; the exporter does not overwrite them. The `results/published/figures/vector_cosines.png` supplemental plot is generated from the recorded vector cosine values.

## Limitations

- One random seed and 250-item test sets do **not** establish significance or robust generalization.
- Base AG News accuracy is low, limiting its value as a selective-forgetting control.
- Addition improves both tasks relative to base but has substantial interference relative to specialized checkpoints.
- The analogy gain is only **2 net correct predictions**; complete paired example-level predictions are unavailable for a paired significance test.
- The experiment uses shortened fine-tuning and a simplified LM objective, so outcomes are not directly comparable to the full-scale original paper.
- Compatible model architecture and identical pretrained initialization are required for the tested full-parameter arithmetic; standalone LoRA adapter deltas are not used here.

## Reference and license

Ilharco, G., et al. (2023). [*Editing Models with Task Arithmetic*](https://arxiv.org/abs/2212.04089), ICLR 2023. [Original project](https://github.com/mlfoundations/task_vectors).

Independent implementation; not affiliated with the original authors. See [LICENSE](LICENSE).
