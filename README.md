# Task Arithmetic: Negation, Addition, and Analogy

**English** | [한국어](README.ko.md)

A minimal, independent implementation of **weight-space task-vector arithmetic**, inspired by [*Editing Models with Task Arithmetic* (Ilharco et al., ICLR 2023)](https://arxiv.org/abs/2212.04089).

The implementation uses **T5-small** and four independently fine-tuned full-parameter checkpoints derived from the same initialization. It examines whether simple parameter-space arithmetic can modify task performance without additional training at merge time.

> **Scope:** Small-scale experimental implementation, not a reproduction of the full paper. No empirical results are included in this repository. Model training and benchmark execution are required to obtain results.

## Operations

Given a shared pretrained parameter vector $\theta_0$ and independently fine-tuned parameters $\theta_t$, the task vector is

$$\tau_t = \theta_t - \theta_0.$$

| Operation | Weight-space edit | Evaluation |
|---|---|---|
| Negation | $\theta_0 - \alpha\tau_{\mathrm{AmazonSent}}$ | Amazon sentiment accuracy and AG News control accuracy |
| Addition | $\theta_0 + \alpha(\tau_{\mathrm{AmazonSent}} + \tau_{\mathrm{AGNews}})$ | Both tasks, compared against base and specialized models |
| Analogy | $\theta_0 + \beta\tau_{\mathrm{AmazonSent}} + \gamma(\tau_{\mathrm{YelpLM}} - \tau_{\mathrm{AmazonLM}})$ | Held-out Yelp sentiment accuracy, compared against source-only transfer |

All edits are applied independently from the original parameter state. Reconstruction of each fine-tuned checkpoint at coefficient 1 is checked numerically.

## Experimental setup

- **Backbone:** [`google-t5/t5-small`](https://huggingface.co/google-t5/t5-small).
- **Supervised datasets:** [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) and [AG News](https://huggingface.co/datasets/fancyzhx/ag_news).
- **Unlabeled domain adaptation:** Amazon and [Yelp Polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity) texts; labels are discarded before LM fine-tuning.
- **Analogy evaluation:** Yelp held-out labeled *test* split only. Target labels are not used for training or hyperparameter selection.
- **Language-modeling objective:** A reduced consecutive-token-chunk seq2seq proxy (96 tokens by default), which differs from the original paper's exact setup.
- **Classifier evaluation:** Length-normalized conditional label log-probabilities, including EOS, rather than free-form text generation.
- **Validation:** Addition coefficients are selected on labeled Amazon and AG News validation data; the source sentiment coefficient is selected on Amazon validation data. Analogy domain coefficient `0.25` is prespecified as the primary comparison; other domain coefficients are exploratory.
- **Compute:** Python 3.10–3.12 and PyTorch; a GPU is recommended. Public datasets and base weights are downloaded at runtime.

## Reproducibility

The default pipeline can be invoked from the repository root with:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
bash scripts/run_all.sh
```

The scripts may also be run individually: `prepare_data`, `train_task`, `extract_vectors`, `run_experiments`, and `plot_results` under `scripts/`. Use `--help` for their arguments. The pipeline performs four full-parameter fine-tunes from the same base checkpoint; runtime and GPU memory requirements vary by environment.

Outputs written locally (ignored by Git):

```text
data/processed/             # prepared dataset subsets
artifacts/checkpoints/      # four fine-tuned checkpoints
artifacts/vectors/          # task-vector tensors and metadata
results/run_summary.json    # measured metrics and configurations
results/metrics.csv         # flattened metrics
results/figures/            # negation, addition and analogy plots
```

`--mode fixture` is only for debugging the pipeline with synthetic inputs. **Fixture outputs are not empirical evidence.** The default mode uses real public datasets.

## Limitations

- Small-sample experiments with a single random seed do not establish generalization or statistical significance.
- Task-vector arithmetic is defined by weight differences; computing a delta alone does not show that a behavioral capability is linearly isolated.
- Strong evidence for negation requires preserving unaffected task performance, which may be difficult to assess when the base model already has poor zero-shot accuracy on a control task.
- Addition may exhibit interference across task vectors; accuracy gains are not guaranteed.
- Analogy is an empirical transfer hypothesis. In particular, a small proxy LM objective and limited training may fail to yield improvements.
- Task vectors require compatible model architectures and a common fine-tuning initialization; LoRA adapters or quantized tensors are not treated as full-model deltas here.

## Source

Ilharco, G., et al. (2023). [*Editing Models with Task Arithmetic*](https://arxiv.org/abs/2212.04089). ICLR 2023. Original implementation: [mlfoundations/task_vectors](https://github.com/mlfoundations/task_vectors).

This repository is an independent educational implementation and is not affiliated with the paper's authors.

## License

See [LICENSE](LICENSE).
