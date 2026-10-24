# Task Arithmetic: Negation, Addition, and Analogy

[English](README.md) | **한국어**

이 저장소는 [*Editing Models with Task Arithmetic* (Ilharco 외, ICLR 2023)](https://arxiv.org/abs/2212.04089)을 바탕으로 [`google-t5/t5-small`](https://huggingface.co/google-t5/t5-small)의 **가중치 공간에서 Task Vector를 산술적으로 조작하는 실험**을 독립적으로 구현한 소규모 프로젝트입니다.

**실행 완료된 Google Colab 실험 1회**를 포함합니다. 실행 기록이 보존된 Notebook, **Colab에서 직접 내려받은 원본 평가 JSON·CSV**, 원본 PNG 그래프, 학습·평가 스크립트, 영문·국문 결과 분석 문서로 구성됩니다. 원 논문 전체를 정확하게 재현한 프로젝트는 아닙니다.

**[실험 결과와 해석](results/README.ko.md) · [Results and interpretation](results/README.md) · [실행 완료 Notebook](notebooks/run_in_colab.ipynb)**

## 실험 목적과 산술 연산

모든 작업이 동일한 사전학습 가중치 $\theta_0$에서 시작해 각각 독립적으로 미세조정되었다고 할 때, Task Vector를 다음과 같이 정의합니다.

$$\tau_t=\theta_t-\theta_0.$$

| 연산 | 가중치 변환 | 평가 대상 |
|---|---|---|
| **Negation (부정)** | $\theta_0-\alpha\tau_{\mathrm{AmazonSent}}$ | Amazon 감성 분석, AG News 대조 작업 |
| **Addition (덧셈)** | $\theta_0+\alpha(\tau_{\mathrm{AmazonSent}}+\tau_{\mathrm{AGNews}})$ | 하나의 모델에서 두 가지 분류 작업 |
| **Analogy (유추)** | $\theta_0+\beta\tau_{\mathrm{AmazonSent}}+\gamma(\tau_{\mathrm{YelpLM}}-\tau_{\mathrm{AmazonLM}})$ | Yelp 정답 레이블을 학습에 사용하지 않은 감성 분석 전이 |

각 연산은 매번 원본 모델 가중치에서 독립적으로 적용합니다. 또한 단일 Task Vector를 계수 1로 더하면 대응하는 미세조정 체크포인트가 수치적으로 복원되는지 확인합니다.

## 실제 실행 결과

**Random seed 42**, 독립적으로 미세조정한 모델 4개, **작업별 테스트 샘플 250개**로 실행한 결과입니다.

| 연산 | 관찰된 결과 | 해석 |
|---|---|---|
| **Negation** ($\alpha=0.5$) | Amazon 정확도 **56.0% → 67.6%**, AG News **30.8% → 26.0%** | 목표 작업의 선택적 망각 **확인되지 않음** |
| **Addition** ($\alpha=1.0$) | Amazon **84.8%**, AG News **57.6%** (원본 **56.0%**, **30.8%**) | 두 작업 모두 원본보다 개선, 작업별 전용 모델(**87.6%**, **77.2%**)에는 미달 |
| **Analogy** ($\beta=0.5$, $\gamma=0.25$) | Yelp **84.0% → 84.8%** (도메인 보정 없는 전이 대비) | **+0.8%p**, 250개 중 정답 수 순증가 2개. 효과 판단은 유보 |

![Addition: Colab 원본 결과 그래프](results/figures/addition.png)

**세 연산 모두 성공했다는 결론이 아닙니다.** 가중치 차이의 계산과 복원이 가능하다는 사실은 확인했지만, 그것만으로 특정 작업 능력이 독립적인 선형 방향에 분리되어 있음을 증명할 수는 없습니다. 상세한 비교와 한계는 [결과 분석 문서](results/README.ko.md)에 정리했습니다.

## 실험 환경과 설정

| 항목 | 실제 실행 값 |
|---|---|
| 기반 모델 | `google-t5/t5-small` |
| 미세조정 작업 | `amazon_sent`, `agnews_topic`, `amazon_lm`, `yelp_lm` |
| 데이터셋 | [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity), [AG News](https://huggingface.co/datasets/fancyzhx/ag_news), [Yelp Polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity) |
| 학습 | 모델당 180 스텝, 배치 크기 4, 그래디언트 누적 2회, 학습률 `1e-4` |
| 지도학습 데이터 | 작업별 학습 1,600개, 검증 200개 |
| 도메인 언어 모델링 | 도메인별 레이블 없는 리뷰 학습 2,500개, 검증 300개 |
| 평가 | 계수 선택용 검증 작업별 80개, 테스트 작업별 250개 |
| 분류 점수 | 정답 후보 레이블의 길이 정규화 조건부 로그확률(EOS 포함) |
| 언어 모델링 근사 | 96토큰 연속 청크를 이용한 seq2seq 대리 목적함수 |
| 실행 장치 | NVIDIA Tesla T4 (Google Colab) |

Yelp 감성 분석 레이블은 **최종 홀드아웃 테스트**에만 사용했으며 학습이나 계수 선택에는 사용하지 않았습니다. 축약된 언어 모델링 목적함수는 원 논문의 정확한 학습 방식과 다릅니다.

## 저장소 구조

```text
README.md                          # 영문 프로젝트 소개
README.ko.md                       # 국문 프로젝트 소개
notebooks/run_in_colab.ipynb       # 코드와 실행 출력이 포함된 Notebook
scripts/                           # 명령줄 실행 및 결과 테이블 생성
task_arithmetic/                   # 모델·데이터·평가 공통 코드
tests/                             # 단위 테스트 및 원본 결과 무결성 검증
results/
  README.md                        # 영문 결과 및 분석
  README.ko.md                     # 국문 결과 및 분석
  run_summary.json                 # Colab 원본 전체 실험 결과 JSON
  metrics.csv                      # Colab 원본 상세 평가 CSV (46건)
  figures/
    negation.png                    # Colab 원본 그래프
    addition.png                    # Colab 원본 그래프
    analogy.png                     # Colab 원본 그래프
  published/                       # 원본 JSON에서 파생한 요약 자료
    metrics.csv
    vectors.csv
    cosine_similarities.csv
    recorded_run.json
    figures/                      # 원본 그림 복사본 및 파생 유사도 그래프
artifacts/checkpoints/             # Git에 포함하지 않는 대형 모델
data/processed/                   # Git에 포함하지 않는 전처리 데이터
```

**출처:** `results/run_summary.json`, `results/metrics.csv`, `results/figures/`의 그래프 3개는 **Colab에서 직접 보관한 원본 파일**입니다. Notebook 화면 출력에서 수치를 추정해 만든 파일이 아닙니다. `results/published/`는 원본 JSON을 바탕으로 생성한 요약 자료입니다. 학습된 전체 모델 체크포인트와 테스트 샘플 전체의 개별 예측 결과는 이번 저장소에 포함하지 않습니다.

## 실행과 검증

Python 3.10–3.12 환경에서 GPU 사용을 권장합니다.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
bash scripts/run_all.sh
```

[Colab Notebook](notebooks/run_in_colab.ipynb)에는 학습 루프, 모델 가중치 차이 계산, 세 연산의 적용, 평가, 시각화가 실제 Python 코드 셀로 제시되어 있습니다. `scripts/`에는 같은 목적의 명령줄 실행 프로그램이 있습니다. 재실행 시 공개 모델과 데이터셋 다운로드가 필요하며, 로컬의 **`results/` 파일들을 덮어쓸 수 있으므로**, 다른 실행 결과를 공개할 때는 이전 기록을 별도로 보존해야 합니다.

**모델 재학습 없이 원본 JSON에서 공개용 요약 자료만 재생성**하려면 다음 명령을 사용합니다.

```bash
python scripts/export_recorded_results.py
```

실험 당시 생성된 원본 그래프 3개는 수정하지 않습니다. `results/published/figures/vector_cosines.png`는 원본 벡터 유사도 값으로 만든 추가 시각화입니다.

## 한계

- 단일 학습 시드와 작업별 테스트 250개만으로 통계적 유의성이나 다른 환경에서의 일반화 성능을 확정할 수 없습니다.
- 기반 모델의 AG News 정확도가 낮아 Negation의 선택적 망각을 평가하기 위한 대조 작업으로 한계가 있습니다.
- Addition은 원본 모델보다 두 작업의 성능을 높였지만 각 작업 전용 모델 대비 간섭이 나타났습니다.
- Analogy의 차이는 **정답 개수 순증가 2개**뿐입니다. 대응 표본 검정에 필요한 전체 샘플별 예측 기록도 없습니다.
- 축약된 미세조정 절차와 언어 모델링 방식은 원 논문 전체 실험과 다릅니다.
- 동일한 초기 가중치에서 출발한 호환 가능한 전체 파라미터 모델을 대상으로 하며, LoRA 어댑터만의 가중치 변화를 전체 모델의 Task Vector로 취급하지 않습니다.

## 참고 문헌 및 라이선스

Ilharco, G. 외 (2023). [*Editing Models with Task Arithmetic*](https://arxiv.org/abs/2212.04089), ICLR 2023. [원 논문 프로젝트](https://github.com/mlfoundations/task_vectors).

본 저장소는 독립적인 구현이며 원 논문 저자의 공식 저장소가 아닙니다. 라이선스는 [LICENSE](LICENSE)를 참고하세요.
