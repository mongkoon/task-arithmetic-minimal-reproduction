# Task Arithmetic 실험 결과 및 분석

[English](README.md) | **한국어** · [프로젝트 소개](../README.ko.md) · [실행 완료 Colab Notebook](../notebooks/run_in_colab.ipynb)

## 핵심 결과

이 폴더는 T5-small을 이용한 **실제 공개 데이터 기반의 Google Colab 실행 결과 1회**를 보관합니다. [`run_summary.json`](run_summary.json), [`metrics.csv`](metrics.csv), [`figures/`](figures/)의 PNG 3개는 **실험 환경에서 직접 다운로드한 원본 파일**입니다. Notebook의 출력 문자열을 읽어 추정하거나 다시 만든 파일이 아닙니다. [`published/`](published/)에는 원본 JSON에서 추출한 간결한 요약 자료가 있습니다.

| 연산 | 대표 관찰 결과 | 해석 |
|---|---|---|
| **Negation** | $\alpha=0.5$에서 Amazon **56.0% → 67.6%**, AG News 대조 작업 **30.8% → 26.0%** | **선택적 망각 확인 실패** |
| **Addition** | 결합 모델에서 Amazon **84.8%**, AG News **57.6%** | **두 작업 모두 원본 모델보다 개선**, 전용 모델과는 성능 차이 존재 |
| **Analogy** | Amazon 전이만 사용했을 때와 비교하여 Yelp **84.0% → 84.8%** | **+0.8%p 순증가**, 효과 확정은 어려움 |

아래 테스트는 모두 **작업별 250개 샘플**로 평가했습니다. 학습 시드는 `42` 한 개이며 전체 실험도 1회입니다. 따라서 원 논문 전체 재현이나 통계적으로 검증된 결과로 해석하지 않습니다.

## 1. 실험 출처와 설정

| 항목 | 원본 기록에서 확인한 값 |
|---|---|
| 완료 시각 | `2026-10-24T18:03:36.189579+00:00` (UTC) |
| 실행 상태 | `completed`, 실제 공개 데이터 사용 |
| 기반 모델 | `google-t5/t5-small` |
| 실행 장치 | NVIDIA Tesla T4, Google Colab (Notebook 기록) |
| 독립 미세조정 모델 | `amazon_sent`, `agnews_topic`, `amazon_lm`, `yelp_lm` |
| 지도학습 데이터 | 작업별 학습 1,600개, 검증 200개 |
| 도메인 LM 데이터 | 도메인별 레이블 없는 리뷰 학습 2,500개, 검증 300개 |
| 학습 | 모델당 180 스텝, 학습률 0.0001, 배치 크기 4, 그래디언트 누적 2 |
| 평가 | 계수 선택용 검증 80개/작업, 최종 테스트 250개/작업 |
| LM 근사 | 96토큰 연속 청크를 seq2seq 대리 목적함수로 학습 |
| 분류 평가 | 정답 후보 레이블의 길이 정규화 조건부 로그확률(EOS 포함) |

데이터는 [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity), [AG News](https://huggingface.co/datasets/fancyzhx/ag_news), [Yelp Polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity)에서 준비했습니다. Yelp의 감성 레이블은 도메인 언어 모델링 학습에서 제외했고 **계수 선택에도 사용하지 않았으며**, 홀드아웃 테스트에만 사용했습니다. 데이터 출처와 개수는 원본 JSON에도 기록되어 있습니다.

## 2. Task Vector 추출과 수치 검증

동일한 기반 모델에서 출발한 각 작업의 가중치 차이를 다음과 같이 정의합니다.

$$\tau_t=\theta_t-\theta_0.$$

| 벡터 | L2 Norm | 가중치 복원 최대 절대오차 | 텐서 개수 |
|---|---:|---:|---:|
| Amazon 감성 분석 | 10.852879 | $2.3283\times10^{-10}$ | 131 |
| AG News 분류 | 12.699175 | $2.3283\times10^{-10}$ | 131 |
| Amazon LM | 13.390650 | $4.6566\times10^{-10}$ | 131 |
| Yelp LM | 13.436141 | $4.6566\times10^{-10}$ | 131 |

$\theta_0+\tau_t\approx\theta_t$ 복원 검증은 성공했습니다. 이는 가중치 연산 구현의 수치적 일관성을 확인한 것이며, 특정 능력이 의미적으로 분리된 한 방향에 존재한다는 증명은 아닙니다.

**벡터 쌍의 코사인 유사도**는 다음과 같습니다.

| 벡터 조합 | 유사도 |
|---|---:|
| Amazon 감성 ↔ AG News | 0.544976 |
| Amazon 감성 ↔ Amazon LM | 0.512805 |
| Amazon 감성 ↔ Yelp LM | 0.505949 |
| AG News ↔ Amazon LM | 0.450671 |
| AG News ↔ Yelp LM | 0.445465 |
| Amazon LM ↔ Yelp LM | **0.705692** |

![Task Vector 코사인 유사도](published/figures/vector_cosines.png)

두 LM 벡터 사이의 유사도가 가장 높았습니다. 공통 학습 목적과 부합하는 관찰이지만, 이 수치만으로 Addition의 간섭 원인을 설명하거나 인과관계를 입증할 수는 없습니다.

## 3. Negation — 선택적 성능 저하 미재현

$$\theta_{\mathrm{neg}}=\theta_0-\alpha\tau_{\mathrm{AmazonSent}}.$$

| 계수 $\alpha$ | Amazon 목표 작업 | AG News 대조 작업 | Amazon 정답/250 | AG News 정답/250 |
|---:|---:|---:|---:|---:|
| 0 (원본) | 56.0% | 30.8% | 140 | 77 |
| 0.25 | 70.8% | 26.0% | 177 | 65 |
| **0.5 (사전 지정)** | **67.6%** | **26.0%** | **169** | **65** |
| 1.0 | 59.6% | 26.0% | 149 | 65 |

참고로 Amazon 전용 미세조정 모델은 Amazon 87.6%, AG News 29.2%를 기록했습니다. 이는 음수 벡터를 적용한 모델이 아니라 별도의 비교 모델입니다.

![Colab 원본 Negation 그래프](figures/negation.png)

**해석:** 테스트한 모든 음수 계수에서 Amazon 정확도가 원본 56.0%보다 **상승**했습니다. 사전에 정한 $\alpha=0.5$에서는 목표 작업이 **+11.6%p**, 대조 작업이 **−4.8%p**였습니다. 따라서 이번 설정에서는 Task Vector 부정을 통한 **선택적 망각 효과가 재현되지 않았습니다.** 또한 4개 범주를 분류하는 AG News의 원본 정확도 자체가 30.8%로 낮아, 대조 작업의 보존 여부를 평가하기에 제한적입니다.

원본 JSON에는 `mean_gold_token_nll`과 `mean_gold_label_probability`도 기록되어 있습니다. 예를 들어 $\alpha=0.5$에서 Amazon 정확도는 올랐지만 정답 토큰 NLL은 **10.2893 → 19.5659**로 증가했습니다. 정확도는 정해진 후보 레이블의 상대적 순위로 결정되고, NLL은 정답 레이블 토큰의 절대 로그확률에 기반하므로 서로 동일한 지표가 아닙니다.

## 4. Addition — 두 작업 성능의 동시 개선

$$\theta_{\mathrm{add}}=\theta_0+\alpha(\tau_{\mathrm{AmazonSent}}+\tau_{\mathrm{AGNews}}).$$

하나의 공통 계수 $\alpha$를 **Amazon 및 AG News 검증 데이터**로 선택했습니다. 후보는 0, 0.25, 0.5, 0.75, 1.0이었고 각 작업 전용 모델의 검증 정확도로 정규화한 점수의 평균을 사용했습니다. 선택된 값은 **$\alpha=1.0$**이며 정규화 검증 점수는 `0.8705153`입니다. 테스트 데이터는 이 계수 선택에 사용하지 않았습니다.

| 홀드아웃 테스트 모델 | Amazon 감성 분석 | AG News |
|---|---:|---:|
| 원본 모델 | 56.0% (140/250) | 30.8% (77/250) |
| Amazon 전용 모델 | 87.6% (219/250) | 29.2% (73/250) |
| AG News 전용 모델 | 70.0% (175/250) | 77.2% (193/250) |
| **Addition ($\alpha=1.0$)** | **84.8% (212/250)** | **57.6% (144/250)** |

![Colab 원본 Addition 그래프](figures/addition.png)

**해석:** 하나의 결합 모델에서 Amazon은 원본 대비 **+28.8%p**, AG News는 **+26.8%p** 높아졌습니다. 하지만 각 전용 모델과 비교하면 Amazon에서 **−2.8%p**, AG News에서 **−19.6%p** 낮았습니다. 따라서 **원본 모델 대비 다중 작업 성능 개선**은 확인되지만, 전용 모델과 동등한 성능이나 간섭 없는 벡터 결합까지 확인된 것은 아닙니다.

## 5. Analogy — 작지만 불확실한 도메인 보정 효과

$$\theta_{\mathrm{analogy}}=\theta_0+\beta\tau_{\mathrm{AmazonSent}}+\gamma(\tau_{\mathrm{YelpLM}}-\tau_{\mathrm{AmazonLM}}).$$

감성 분석 벡터 계수 $\beta=0.5$는 **Amazon 검증 데이터만** 사용해 선택했습니다. 후보 0.5, 0.75, 1.0은 모두 Amazon 검증 80개 중 73개를 맞혀 **91.25%**였고, 구현의 동률 시 첫 번째 값을 선택하는 규칙으로 0.5가 채택됐습니다. 도메인 보정 계수 **$\gamma=0.25$는 Yelp 테스트 전에 지정한 기본 조건**입니다. 나머지 값은 탐색적 민감도 비교입니다.

| 조건 | Yelp 홀드아웃 테스트 | Amazon 테스트 |
|---|---:|---:|
| 원본 모델 | 59.2% (148/250) | 56.0% (140/250) |
| Amazon 감성 분석만 적용 ($\gamma=0$) | **84.0% (210/250)** | 87.6% (219/250) |
| $\gamma=0.1$ (탐색) | 84.4% (211/250) | 88.0% (220/250) |
| **$\gamma=0.25$ (사전 지정)** | **84.8% (212/250)** | **88.0% (220/250)** |
| $\gamma=0.5$ (탐색) | 84.4% (211/250) | 88.4% (221/250) |
| 도메인 차이 벡터만 적용 | **58.4% (146/250)** | — |

![Colab 원본 Analogy 그래프](figures/analogy.png)

**해석:** Yelp 정확도의 대부분의 향상(59.2% → 84.0%)은 Amazon 감성 분석 벡터만으로 얻었습니다. 여기에 사전 지정한 도메인 보정 항을 더하자 **210개에서 212개**로 총 정답 수가 **2개 순증가(+0.8%p)**했습니다. 도메인 보정 벡터만 적용한 대조군은 Yelp 정확도를 높이지 못했습니다. 따라서 이번 단일 실행에서 Analogy의 효과는 **미확정**입니다. 정답 수의 순증가가 2개라는 사실은 오직 같은 두 샘플만 오답에서 정답으로 바뀌었다는 뜻이 아닙니다.

## 6. 해석상의 한계

1. **단일 실행·단일 시드:** 여러 시드의 반복 실험이나 신뢰구간·유의성 검정이 없습니다.
2. **전체 샘플별 대응 비교 불가:** 원본 JSON의 각 평가 행에는 대표 예측 8개(`prediction_sample`)만 있고 전체 250개 샘플의 대응 예측은 없습니다. 따라서 현재 보관한 자료만으로 McNemar 검정을 재구성할 수 없습니다.
3. **LM 학습 목적의 차이:** 도메인 LM은 96토큰 연속 청크를 이용한 seq2seq 근사이므로 원 논문과 완전히 동일한 실험이 아닙니다.
4. **Negation 대조군의 제약:** 기반 모델의 AG News 정확도가 낮아 선택적 성능 보존 평가가 제한적입니다.
5. **계수 비교의 의미:** Negation·Analogy 기본 계수는 사전에 지정했으나, 탐색적으로 그린 곡선을 독립적인 반복 실험 결과로 볼 수는 없습니다.
6. **벡터와 의미적 능력의 구분:** 수치적으로 체크포인트를 복원했다고 해서 개별 작업 능력이 하나의 선형 방향으로 완전히 분리되어 있다는 증명은 아닙니다.

## 7. 저장소에 포함된 결과 파일

| 파일 | 의미 |
|---|---|
| [`run_summary.json`](run_summary.json) | **Colab 원본 JSON**: 46개 평가 기록, 정답 개수, 점수 통계, 각 행의 대표 예측 8개, 학습 설정, 벡터 정보 |
| [`metrics.csv`](metrics.csv) | **Colab 원본 CSV**, 46행의 상세 정확도·정답 수·확률·손실 지표 |
| [`figures/negation.png`](figures/negation.png) | **원본 Negation 그래프** |
| [`figures/addition.png`](figures/addition.png) | **원본 Addition 그래프** |
| [`figures/analogy.png`](figures/analogy.png) | **원본 Analogy 그래프** |
| [`provenance.json`](provenance.json) | 원본 파일 SHA-256 체크섬과 실행 식별 정보 |
| [`published/metrics.csv`](published/metrics.csv) | 원본 JSON에서 추출한 간결한 46행 정확도·정답 수 요약 |
| [`published/vectors.csv`](published/vectors.csv) | 원본 JSON의 벡터 노름, 복원 오차, 텐서 개수 |
| [`published/cosine_similarities.csv`](published/cosine_similarities.csv) | 원본 JSON의 코사인 유사도 6개 |
| [`published/recorded_run.json`](published/recorded_run.json) | 원본 JSON에서 추출한 실행 설정·메타데이터 |
| [`published/figures/vector_cosines.png`](published/figures/vector_cosines.png) | 원본 유사도 수치로 생성한 보조 그림 |
| `published/figures/{negation,addition,analogy}.png` | 기존 저장소 경로와 호환되도록 보존한 원본 그래프 3개의 동일한 복사본 |
| [`../notebooks/run_in_colab.ipynb`](../notebooks/run_in_colab.ipynb) | 실행 코드와 당시 출력을 보존한 Notebook |

**모델 재학습 없이 원본 JSON에서 요약 테이블과 원본 그림의 복사본을 재생성**할 수 있습니다.

```bash
python scripts/export_recorded_results.py
```

이 명령은 원본 JSON·CSV와 원본 그래프 3개를 변경하지 않습니다. 반면 `bash scripts/run_all.sh`로 전체 학습을 새로 실행하면 작업 폴더의 동일한 `results/` 경로를 덮어쓸 수 있으므로 새 실행은 별도로 보관한 뒤 의도적으로 버전 관리해야 합니다.

## 참고 문헌

Ilharco, G. 외 (2023). [*Editing Models with Task Arithmetic*](https://arxiv.org/abs/2212.04089). ICLR 2023. 본 실험은 도메인 LM 근사 방식이 다른 독립적인 소규모 구현입니다.
