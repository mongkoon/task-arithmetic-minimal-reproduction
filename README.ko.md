# Task Arithmetic: Negation, Addition, and Analogy

[English](README.md) | **한국어**

[《Editing Models with Task Arithmetic》(Ilharco 외, ICLR 2023)](https://arxiv.org/abs/2212.04089)에서 제안한 **가중치 공간(Weight Space)의 Task Vector 산술 연산**을 독립적으로 구현한 최소 규모의 실험 프로젝트입니다.

동일한 초기 가중치에서 출발한 **T5-small** 모델을 네 가지 작업에 대해 각각 전체 파라미터 미세조정(Full-Parameter Fine-Tuning)하여 체크포인트를 생성합니다. 이후 모델 병합 시점에 추가 학습을 수행하지 않고, 단순한 가중치 산술 연산만으로 작업 성능을 변화시킬 수 있는지 살펴봅니다.

> **연구 범위:** 원 논문 전체를 재현하는 것이 아닌 소규모 실험 구현입니다. 이 저장소에는 실제 측정 결과가 포함되어 있지 않으며, 결과를 얻으려면 모델 학습과 벤치마크 평가를 직접 실행해야 합니다.

## 산술 연산 (Operations)

공통 사전학습 모델의 파라미터를 $\theta_0$, 동일한 초기 모델을 작업 $t$에 대해 미세조정한 파라미터를 $\theta_t$라고 할 때, Task Vector는 다음과 같이 정의합니다.

$$\tau_t = \theta_t - \theta_0.$$

| 연산 | 가중치 공간 연산 | 평가 내용 |
|---|---|---|
| Negation (부정) | $\theta_0 - \alpha\tau_{\mathrm{AmazonSent}}$ | Amazon 감성 분류 정확도와 대조 작업인 AG News 정확도 |
| Addition (덧셈) | $\theta_0 + \alpha(\tau_{\mathrm{AmazonSent}} + \tau_{\mathrm{AGNews}})$ | 두 작업의 정확도를 원본 및 각 작업의 전용 미세조정 모델과 비교 |
| Analogy (유추) | $\theta_0 + \beta\tau_{\mathrm{AmazonSent}} + \gamma(\tau_{\mathrm{YelpLM}} - \tau_{\mathrm{AmazonLM}})$ | 별도로 확보한 Yelp 감성 분류 테스트 데이터에서의 정확도를 원천 도메인 벡터만 사용한 전이 결과와 비교 |

모든 가중치 수정은 **원본 파라미터 상태에서 독립적으로** 수행합니다. 또한 각 Task Vector에 계수 1을 적용했을 때 해당 미세조정 체크포인트가 수치적으로 복원되는지 검증합니다.

## 실험 설정 (Experimental Setup)

- **기반 모델(Backbone):** [`google-t5/t5-small`](https://huggingface.co/google-t5/t5-small).
- **지도학습 데이터셋:** [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity), [AG News](https://huggingface.co/datasets/fancyzhx/ag_news).
- **레이블 없는 도메인 적응:** Amazon 및 [Yelp Polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity)의 텍스트를 사용하며, 언어 모델링(LM) 미세조정 전에 레이블은 제외합니다.
- **Analogy 평가:** 별도로 보류한 Yelp의 정답 레이블이 포함된 *테스트(test)* 분할만 사용합니다. 목표 도메인의 레이블은 학습이나 하이퍼파라미터 선택에 사용하지 않습니다.
- **언어 모델링 학습 목표:** 연속 토큰을 일정 길이로 나눈 청크(chunk)를 활용하는 축소형 Seq2Seq 대리 학습 목표(기본 96토큰)를 사용합니다. 이는 원 논문의 정확한 실험 설정과 다릅니다.
- **분류 평가:** 자유 형식의 텍스트 생성 결과가 아니라 EOS 토큰을 포함한 레이블의 **길이 정규화 조건부 로그확률**을 사용합니다.
- **검증 및 계수 선택:** Addition의 계수는 정답 레이블이 있는 Amazon 및 AG News 검증 데이터로 선택합니다. 원천 감성 분류 벡터의 계수는 Amazon 검증 데이터로 선택합니다. Analogy의 도메인 계수 `0.25`는 주 비교 조건으로 사전 지정하며, 다른 도메인 계수는 탐색적 분석에만 사용합니다.
- **실행 환경:** Python 3.10–3.12 및 PyTorch. GPU 사용을 권장하며, 공개 데이터셋과 기반 모델 가중치는 실행 시 다운로드됩니다.

## 재현 방법 (Reproducibility)

저장소 최상위 디렉터리에서 아래 명령어로 기본 실험 파이프라인을 실행할 수 있습니다.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
bash scripts/run_all.sh
```

`scripts/` 디렉터리의 `prepare_data`, `train_task`, `extract_vectors`, `run_experiments`, `plot_results`를 각각 실행할 수도 있습니다. 개별 스크립트의 인수는 `--help`로 확인하세요. 전체 파이프라인은 동일한 기반 체크포인트에서 출발해 네 번의 전체 파라미터 미세조정을 수행하며, 소요 시간과 GPU 메모리 요구량은 실행 환경에 따라 달라집니다.

로컬에 생성되는 결과 파일은 아래와 같으며 Git 추적에서는 제외됩니다.

```text
data/processed/             # 준비된 데이터셋 일부
artifacts/checkpoints/      # 미세조정된 체크포인트 4개
artifacts/vectors/          # Task Vector 텐서 및 메타데이터
results/run_summary.json    # 측정된 지표와 실험 설정
results/metrics.csv         # 표 형태로 정리한 평가 지표
results/figures/            # Negation, Addition, Analogy 그래프
```

`--mode fixture`는 합성 입력을 사용하여 실험 파이프라인을 디버깅하기 위한 옵션일 뿐입니다. **Fixture 실행 결과는 실증적 증거가 아닙니다.** 기본 실행 모드는 실제 공개 데이터셋을 사용합니다.

## 한계 (Limitations)

- 소량의 표본과 단일 난수 시드(Random Seed)를 사용한 실험만으로는 일반화 성능이나 통계적 유의성을 입증할 수 없습니다.
- Task Vector 산술 연산은 가중치 차이로 정의됩니다. 가중치 차이를 계산할 수 있다는 사실만으로 특정 행동 능력이 선형적으로 분리되어 있음을 입증할 수는 없습니다.
- Negation의 효과를 강하게 뒷받침하려면 목표와 무관한 작업의 성능이 유지되는지도 확인해야 합니다. 그러나 기반 모델이 대조 작업에서 이미 낮은 Zero-shot 정확도를 보인다면 이러한 보존 여부를 평가하기 어려울 수 있습니다.
- Addition에서는 서로 다른 Task Vector 사이에 간섭이 발생할 수 있으므로 정확도 향상이 보장되지 않습니다.
- Analogy는 실험적으로 검증해야 하는 전이 가설입니다. 특히 축소형 LM 대리 학습 목표와 제한적인 학습만으로는 성능 향상이 나타나지 않을 수 있습니다.
- Task Vector를 산술적으로 결합하려면 모델 아키텍처가 호환되고 미세조정 초기 가중치가 같아야 합니다. 이 프로젝트에서는 LoRA 어댑터나 양자화된 텐서를 전체 모델의 가중치 차이로 취급하지 않습니다.

## 참고 문헌 (Source)

Ilharco, G., et al. (2023). [*Editing Models with Task Arithmetic*](https://arxiv.org/abs/2212.04089). ICLR 2023. 원 논문 구현: [mlfoundations/task_vectors](https://github.com/mlfoundations/task_vectors).

이 저장소는 학습과 실험을 위한 독립적인 구현이며, 원 논문 저자들과 공식적인 관련이 없습니다.

## 라이선스 (License)

[LICENSE](LICENSE) 파일을 참고하세요.
