# Relaxed RB Allocation Experiments

Wireless federated learning의 사용자 선택 및 RB(Resource Block) 할당 문제를 대상으로, 논문식 Hungarian hard matching과 continuous relaxation 기반 대안을 비교하는 실험 저장소입니다.

## Part 1. 문제와 제안

### 배경

- 대상 문제
  - Wireless FL에서 각 round마다 학습에 참여할 사용자와 RB를 함께 선택해야 합니다.
  - 원래 문제는 binary user selection, binary RB allocation, 송신 전력 최적화를 포함하는 non-convex MINLP입니다.
- 베이스 논문 접근
  - FL 수렴률 분석으로 목적함수를 단순화합니다.
  - 각 사용자-RB 조합의 최적 송신 전력 `P*_{i,n}`을 먼저 계산합니다.
  - 남은 RB 할당은 Hungarian bipartite matching으로 풉니다.
- 문제점
  - 최종 할당이 0/1 hard assignment라 differentiable pipeline으로 연결하기 어렵습니다.
  - continuous relaxation을 해도 최종 시스템이 0/1 할당을 요구하면 rounding/projection이 필요합니다.
  - rounding/projection 품질은 상황에 따라 Hungarian 대비 objective gap을 크게 만들 수 있습니다.

### 아이디어 제시

- 공통 출발점
  - `P*_{i,n}`과 `qhat_{i,n}`을 미리 계산하고 상수로 둡니다.
  - binary allocation을 continuous variable로 완화합니다.

```text
r_{i,n} in {0,1}  ->  x_{i,n} in [0,1]
```

- 두 가지 실험 축
  - Hard 복원 실험: relaxation 또는 soft score를 만든 뒤 rounding/projection으로 0/1 assignment를 복원합니다.
  - Continuous soft 실험: rounding 없이 `X in [0,1]` 자체를 time-sharing 또는 probabilistic allocation으로 평가합니다.

### 아이디어의 핵심 개념 설명

- 선형화된 목적함수

```text
min_X sum_i K_i (1 - sum_n x_{i,n} + sum_n x_{i,n} qhat_{i,n})
```

- 해석
  - `K_i`: 사용자 `i`의 sample count.
  - `qhat_{i,n}`: `P*_{i,n}`에서의 packet error probability.
  - `x_{i,n}`: 사용자 `i`가 RB `n`을 사용하는 정도.
- 핵심 관찰
  - `P*`와 `qhat`을 고정하면 objective와 assignment 제약이 선형입니다.
  - LP relaxation은 convex problem입니다.
  - 다만 assignment polytope의 extreme point가 integral이므로, 단순 LP는 실제로 Hungarian과 같은 hard solution으로 수렴할 수 있습니다.

### 아이디어의 구조/아키텍처 설명

- 데이터 생성
  - `src/relaxed_ra/instance.py`: 사용자 수, RB 수, channel gain, sample count, packet error matrix 생성.
- Solver 계층
  - `Hungarian`: 논문식 hard matching 기준선.
  - `LP-Relax+Projection`: LP relaxation 후 matching projection.
  - `Greedy-Cost`: cost 기준 greedy hard assignment.
  - `Entropy-Relax+Greedy/Projection`: entropy soft matrix 생성 후 hard 복원.
  - `Hybrid-Score-Greedy`: soft score와 원래 cost utility를 혼합한 greedy 복원.
  - `Continuous-LP(HiGHS)`: `X in [0,1]` LP를 solver로 직접 풂.
  - `Soft-Entropy-KKT`: entropy-regularized convex 문제를 KKT/Sinkhorn scaling으로 풂.
- Paper-style 재현 계층
  - 논문 Fig.4-Fig.10의 실험 축을 따라 linear regression loss, matching work, convergence gap, digit accuracy를 평가합니다.
  - 논문 baseline 3종과 새 relaxation baseline 4종을 같은 FL loop에 적용합니다.
- 실험 실행
  - `run_experiments.py`: hard assignment 복원 실험.
  - `run_continuous_experiments.py`: continuous soft allocation 실험.
  - `run_paper_reproduction.py`: 논문 figure 구조를 따른 세 번째 비교 실험.
- 산출물
  - CSV: raw result와 summary.
  - SVG plot: objective gap, runtime, fractional mass ratio.

### 사용한 수학적 기법

- LP relaxation
  - `x_{i,n} in [0,1]`, row/column assignment constraint를 갖는 convex LP입니다.
  - 구현: `scipy.optimize.linprog(method="highs")`.
- Hungarian matching
  - hard assignment 기준선입니다.
  - 구현: `scipy.optimize.linear_sum_assignment`.
- Matching projection
  - soft/relaxed matrix를 다시 feasible 0/1 assignment로 복원합니다.
- Entropy regularization
  - soft allocation을 만들기 위해 다음 convex surrogate를 사용합니다.

```text
min_X <C, X> + tau * sum_{i,n} x_{i,n}(log x_{i,n} - 1)
s.t.  sum_n x_{i,n} <= 1,  sum_i x_{i,n} <= 1,  x_{i,n} >= 0
```

- KKT/Sinkhorn scaling
  - dummy user/RB를 추가해 부등식 제약을 balanced scaling 문제로 바꿉니다.
  - KKT stationarity는 다음 형태입니다.

```text
x_{i,n} = u_i * exp(-C_{i,n} / tau) * v_n
```

  - `Soft-Entropy-KKT`는 이 식을 기반으로 Sinkhorn iteration을 수행합니다.
  - 일반 nonlinear solver는 사용하지 않았습니다.

### 기존 방법과의 차이점

| 구분 | 베이스 논문 | Hard 복원 실험 | Continuous soft 실험 |
|---|---|---|---|
| 최종 allocation | 0/1 hard | 0/1 hard | `[0,1]` soft |
| 핵심 solver | Hungarian | LP, entropy, greedy, projection | LP, KKT/Sinkhorn |
| convex성 | matching 단계는 combinatorial | relaxation 단계만 convex 가능 | soft allocation을 인정하면 convex pipeline 가능 |
| 목적 | 논문 기준선 재현 | relaxation 후 hard 복원 품질 평가 | hard 복원 없이 soft allocation 자체 평가 |
| 주요 리스크 | binary 제약 | rounding/projection 손실 | 실제 시스템이 soft/time-sharing을 허용해야 함 |

세 번째 실험에서는 논문 baseline을 다음 이름으로 구현했습니다.

- `Paper-Hungarian`: 논문 제안 방식의 Hungarian RB/user assignment.
- `Paper-OptUser-RandomRB`: user selection은 cost 기준으로 고르되 RB는 random으로 제한.
- `Paper-RandomUserRB`: user selection과 RB allocation을 모두 random으로 결정.
- `Paper-Wireless-PER`: FL sample 수를 무시하고 packet error rate만 최소화.

추가 비교군은 `LP-Relax+Projection`, `Hybrid-Score-Greedy(alpha=0.25)`, `Continuous-LP(HiGHS)`, `Soft-Entropy-KKT(tau=1)`입니다.

## Part 2. 실험과 평가

### 실험 실행 방법

- 전체 테스트

```powershell
python -m unittest discover -v
```

- Hard assignment 복원 실험

```powershell
python run_experiments.py --output results --seeds 5
```

- Continuous soft allocation 실험

```powershell
python run_continuous_experiments.py --output results/continuous --seeds 5
```

- Paper-style 재현 실험

```powershell
python run_paper_reproduction.py --output results/paper --seeds 3
```

- 빠른 smoke run

```powershell
python run_experiments.py --output results_quick --seeds 2 --quick
python run_continuous_experiments.py --output results_continuous_quick --seeds 2 --quick
python run_paper_reproduction.py --output results_paper_quick --seeds 1 --quick
```

- 주요 결과 파일
  - `results/raw_results.csv`
  - `results/summary.csv`
  - `results/continuous/raw_results.csv`
  - `results/continuous/summary.csv`
  - `results/paper/paper_raw_results.csv`
  - `results/paper/paper_summary.csv`
- 주요 plot
  - [hard objective gap](results/plots/objective_gap_by_users.svg)
  - [hard runtime](results/plots/runtime_by_users.svg)
  - [hard convergence gap](results/plots/convergence_gap_by_rbs.svg)
  - [continuous objective gap](results/continuous/plots/soft_objective_gap_by_users.svg)
  - [continuous fractional mass](results/continuous/plots/fractional_mass_by_users.svg)
  - [continuous runtime](results/continuous/plots/soft_runtime_by_users.svg)
  - [paper Fig.4-style linear loss](results/paper/plots/paper_fig4_linear_loss_by_samples.svg)
  - [paper Fig.5-style number of iterations](results/paper/plots/paper_fig5_matching_work_by_users.svg)
  - [paper Fig.6-style convergence gap](results/paper/plots/paper_fig6_convergence_gap_by_users.svg)
  - [paper Fig.7-style digit accuracy by iterations](results/paper/plots/paper_fig7_digit_accuracy_by_iterations.svg)
  - [paper Fig.8-style digit accuracy by users](results/paper/plots/paper_fig8_digit_accuracy_by_users.svg)
  - [paper Fig.9-style digit accuracy by RBs](results/paper/plots/paper_fig9_digit_accuracy_by_rbs.svg)
  - [paper Fig.10-style digit examples](results/paper/plots/paper_fig10_digit_examples.svg)

### 실험 환경

- 실행 환경
  - OS: Windows 11 `10.0.26200`
  - CPU: Intel Core Ultra 5 226V, 8 cores / 8 logical processors
  - Memory: 약 16 GB
  - GPU: 사용하지 않음
- Python 및 의존성
  - Python `3.14.0`
  - NumPy `2.4.3`
  - SciPy `1.17.1`
  - scikit-learn `1.8.0`: `load_digits`, `train_test_split` 사용
- 공통 실험 파라미터
  - seeds: `5`
  - 사용자 수 sweep: `U = [10, 15, 20, 30, 50, 80]`, `R = 12`
  - RB 수 sweep: `R = [5, 9, 12, 15, 20]`, `U = 15`
- Paper-style 실험 파라미터
  - seeds: `3`
  - Fig.5 사용자 수 sweep: `U = [3, 5, 10, 15, 20, 25]`
  - Fig.6/Fig.8 사용자 수 sweep: `U = [3, 6, 9, 12, 15, 18]`, `R = 12`
  - Fig.9 RB 수 sweep: `R = [3, 6, 9, 12]`, `U = 15`
  - Fig.4 samples per user sweep: `[10, 20, 30, 40, 50]`
  - Fig.7 digit FL rounds: `120`
  - Fig.8/Fig.9 digit FL rounds: `12`
  - plot format: 원 논문처럼 MATLAB-style 흰 배경, 검은 box 축, 회색 grid, 내부 legend, 원 논문 caption 텍스트를 사용합니다.
- 결과 규모
  - Hard 복원 실험: `raw_results.csv` 550 rows, `summary.csv` 110 rows.
  - Continuous soft 실험: `raw_results.csv` 275 rows, `summary.csv` 55 rows.
  - Paper-style 재현 실험: `paper_raw_results.csv` 840 rows, `paper_summary.csv` 280 rows.
- Paper-style 재현 범위
  - 논문과 같은 figure 축을 사용합니다: sample 수, 사용자 수, RB 수, FL iteration.
  - Linear regression은 논문식 `y = -2x + 1 + 0.4n` 데이터를 사용합니다.
  - 원 논문은 Matlab FNN과 MNIST를 사용하지만, 현재 repo는 재현 가능성을 위해 NumPy softmax FL과 scikit-learn digits 데이터를 사용합니다.
  - 따라서 이 결과는 논문 수치의 완전 복제가 아니라, 논문 실험 구조에 relaxation baseline을 추가한 paper-style reproduction입니다.

### 실험 결과

- Hard assignment 복원 실험 핵심 결과

| 방법 | 평균 objective gap | 최악 평균 gap | 평균 runtime | 평균 FL quality |
|---|---:|---:|---:|---:|
| `Hungarian` | 0.00% | 0.00% | 0.093 ms | 0.7288 |
| `LP-Relax+Projection` | 0.00% | 0.00% | 6.580 ms | 0.7288 |
| `Hybrid-Score-Greedy(alpha=0.25)` | 10.30% | 39.79% | 2.788 ms | 0.7238 |
| `Greedy-Cost` | 16.41% | 89.39% | 0.225 ms | 0.7213 |
| `Entropy-Relax(tau=0.8)+Projection` | 30.20% | 121.58% | 5.735 ms | 0.6710 |
| `Entropy-Relax(tau=0.3)+Greedy` | 34.25% | 90.19% | 5.367 ms | 0.6720 |

- Continuous soft allocation 실험 핵심 결과

| 방법 | 평균 linear gap | 최악 평균 gap | fractional mass ratio | 평균 KKT residual | 평균 runtime | 평균 FL quality |
|---|---:|---:|---:|---:|---:|---:|
| `Hungarian(reference)` | 0.00% | 0.00% | 0.0000 | 0.000e+00 | 0.051 ms | 0.7281 |
| `Continuous-LP(HiGHS)` | 0.00% | 0.00% | 0.0000 | 0.000e+00 | 2.887 ms | 0.7281 |
| `Soft-Entropy-KKT(tau=1)` | 10.28% | 52.48% | 1.0000 | 1.918e-04 | 3.257 ms | 0.7254 |
| `Soft-Entropy-KKT(tau=5)` | 48.83% | 211.22% | 1.0000 | 3.417e-11 | 0.992 ms | 0.7125 |
| `Soft-Entropy-KKT(tau=25)` | 250.57% | 975.23% | 1.0000 | 6.293e-12 | 0.261 ms | 0.6343 |

- Paper-style 재현 실험 핵심 결과

| 실험 축 | 최고/주요 방법 | 핵심 수치 | Paper-Hungarian 기준 |
|---|---|---:|---:|
| Fig.4-style linear MSE 평균 | `Soft-Entropy-KKT(tau=1)` | 0.00143 | `Paper-Hungarian` 0.00192 |
| Fig.4-style linear MSE 평균 | `Hybrid-Score-Greedy(alpha=0.25)` | 0.00154 | `Paper-Hungarian` 0.00192 |
| Fig.6-style normalized convergence gap | `Paper-Hungarian` / `LP-Relax+Projection` / `Continuous-LP(HiGHS)` | 0.06227 | 동일 |
| Fig.7-style final digit accuracy, 120 rounds | `Soft-Entropy-KKT(tau=1)` | 0.9401 | `Paper-Hungarian` 0.9375 |
| Fig.8-style digit accuracy by users 평균 | `Soft-Entropy-KKT(tau=1)` | 0.8949 | `Paper-Hungarian` 0.8808 |
| Fig.9-style digit accuracy by RBs 평균 | `Soft-Entropy-KKT(tau=1)` | 0.7472 | `Paper-Hungarian` 0.6151 |

논문 baseline이 강조한 조건도 별도로 확인했습니다.

| 조건 | `Paper-Hungarian` | 논문 baseline 중 최고 | 추가 baseline 중 최고 |
|---|---:|---:|---:|
| Fig.8-style `U=18, R=12` digit accuracy | 0.8198 | `Paper-Wireless-PER` 0.8362 | `Soft-Entropy-KKT(tau=1)` 0.8511 |
| Fig.9-style `U=15, R=9` digit accuracy | 0.7292 | `Paper-RandomUserRB` 0.7329 | `Soft-Entropy-KKT(tau=1)` 0.8299 |

### 결과 평가 및 기존 방법과의 결과 비교

- `LP-Relax+Projection`
  - Hungarian 대비 평균 gap이 `0.00%`입니다.
  - 현재 선형 assignment 구조에서는 LP relaxation이 성능 손실 없이 hard optimum을 복원합니다.
  - 단, 평균 runtime은 `6.580 ms`로 Hungarian의 `0.093 ms`보다 큽니다.
- `Hybrid-Score-Greedy(alpha=0.25)`
  - `Greedy-Cost` 대비 평균 gap을 `16.41% -> 10.30%`로 줄였습니다.
  - 평균 FL quality도 `0.7213 -> 0.7238`로 개선되었습니다.
  - hard 복원이 필요하고 Hungarian/LP보다 가벼운 근사가 필요할 때 가장 실용적인 대안입니다.
- `Entropy-Relax+Projection`
  - RB가 충분한 일부 조건에서는 Hungarian에 매우 근접합니다.
  - 전체 평균 기준으로는 gap이 `30%` 이상이라 안정적이지 않습니다.
  - soft matrix만 믿고 projection하면 원래 cost 구조를 충분히 반영하지 못할 수 있습니다.
- `Continuous-LP(HiGHS)`
  - convex LP이지만 fractional allocation을 만들지 못했습니다.
  - fractional mass ratio가 `0.0000`이므로 결과적으로 Hungarian과 같은 hard extreme point로 수렴했습니다.
- `Soft-Entropy-KKT(tau=1)`
  - fractional mass ratio가 `1.0000`이므로 실제 continuous soft allocation입니다.
  - 평균 linear gap은 `10.28%`이지만 FL quality는 `0.7254`로 Hungarian reference의 `0.7281`에 가깝습니다.
  - soft/time-sharing allocation을 시스템적으로 허용한다면 가장 의미 있는 convex continuous 대안입니다.
- `tau` 영향
  - `tau=1`: 품질과 softness의 균형이 가장 좋습니다.
  - `tau=5`: smoothing 증가로 평균 gap이 `48.83%`까지 증가합니다.
  - `tau=25`: allocation이 과도하게 퍼져 평균 gap이 `250.57%`로 악화됩니다.
- Paper-style 재현 실험
  - `Paper-Hungarian`, `LP-Relax+Projection`, `Continuous-LP(HiGHS)`는 normalized convergence gap에서 같은 값 `0.06227`을 냈습니다. 이는 선형 assignment relaxation이 hard extreme point로 수렴한다는 앞선 결과와 일치합니다.
  - `Soft-Entropy-KKT(tau=1)`은 digit accuracy에서 가장 높았습니다. 평균 accuracy는 사용자 수 sweep에서 `0.8949`, RB 수 sweep에서 `0.7472`입니다.
  - 이 개선은 soft allocation이 여러 사용자의 업데이트를 fractional하게 반영해 gradient 다양성을 높인 효과로 해석됩니다.
  - 단, 실제 RB가 반드시 0/1 hard allocation이어야 한다면 이 이점은 time-sharing 또는 probabilistic scheduling 계층이 있을 때만 실현 가능합니다.
  - `Paper-OptUser-RandomRB`는 대부분의 축에서 가장 낮았습니다. RB를 random으로 제한하면 user selection을 개선해도 feasible/high-quality edge를 놓치는 손실이 큽니다.

### 한계점 및 추후 계획

- 한계점
  - `P*_{i,n}`과 `qhat_{i,n}`을 상수로 고정했기 때문에 전력 제어와 allocation의 joint optimization은 아직 다루지 않았습니다.
  - LP relaxation은 convex이지만 assignment polytope 특성상 soft solution을 보장하지 않습니다.
  - Soft allocation은 실제 무선 시스템이 time-sharing 또는 probabilistic RB allocation을 허용해야 적용 가능합니다.
  - Hard 복원이 필요한 경우 rounding/projection 단계 때문에 전체 파이프라인은 완전한 convex optimization으로 닫히지 않습니다.
  - 현재 실험은 synthetic channel/problem generator 기반이며 실제 wireless trace 검증은 포함하지 않았습니다.
  - Paper-style 재현 실험은 논문 figure 구조를 따르지만, Matlab FNN/MNIST 원본 실험을 bitwise 또는 수치적으로 완전 복제한 것은 아닙니다.
  - Fig.5의 Hungarian iteration은 SciPy가 내부 iteration을 제공하지 않아 원 논문 축/텍스트를 유지한 solver iteration proxy로 대체했습니다.
- 추후 계획
  - 전력 변수까지 포함한 convex surrogate 또는 alternating convex optimization 확장.
  - entropy-KKT의 `tau` adaptive scheduling 및 residual 기반 stopping rule 개선.
  - cost-aware projection을 강화해 `Entropy-Relax+Projection`의 불안정성 완화.
  - soft allocation을 time-sharing 정책으로 해석한 실제 round-level FL simulation 추가.
  - 실제 channel/RB trace 기반 robustness 평가 추가.
  - torchvision/MNIST 또는 원 논문 Matlab 설정을 사용할 수 있는 환경에서 paper-style 실험을 full MNIST/FNN reproduction으로 확장.
