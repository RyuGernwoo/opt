# Relaxed RB Allocation Experiments

## 배경

이 저장소는 `A_Joint_Learning_and_Communications_Framework_for_Federated_Learning_Over_Wireless_Networks.pdf`의 wireless FL 자원 할당 문제를 바탕으로, hard assignment 기반 Hungarian matching과 continuous relaxation 기반 근사 방법을 비교하는 실험 코드입니다.

베이스 논문은 binary user selection과 RB allocation을 포함한 non-convex MINLP를 FL 수렴률 분석으로 단순화합니다. 이후 각 사용자-RB 조합에 대해 최적 송신 전력 `P*_{i,n}`을 먼저 계산하고, 남은 사용자-RB 할당 문제를 Hungarian bipartite matching으로 풉니다.

## 제안 방법

본 구현은 binary RB 할당 변수를 연속 변수로 완화합니다.

```text
r_{i,n} in {0,1}  ->  x_{i,n} in [0,1]
```

각 사용자-RB 조합에 대해 `P*_{i,n}`과 `qhat_{i,n}`을 미리 계산한 뒤 상수로 취급하면 목적함수는 다음과 같은 선형 assignment 형태가 됩니다.

```text
min_X sum_i K_i (1 - sum_n x_{i,n} + sum_n x_{i,n} qhat_{i,n})
```

비교한 방법은 다음과 같습니다.

- `Hungarian`: 논문식 hard matching 기준선.
- `LP-Relax+Projection`: LP relaxation을 푼 뒤 matching projection으로 0/1 할당 복원.
- `Greedy-Cost`: 원래 cost 기준 greedy rounding.
- `Entropy-Relax(tau=...)+Greedy`: entropy soft assignment 후 greedy rounding.
- `Entropy-Relax(tau=...)+Projection`: entropy soft assignment 후 matching projection.
- `Hybrid-Score-Greedy(alpha=0.25)`: soft score와 원래 cost utility를 혼합한 greedy rounding.

## 실행 방법

필요 패키지는 Python, `numpy`, `scipy`입니다.

```powershell
python -m unittest discover -v
python run_experiments.py --output results --seeds 5
```

빠른 smoke run은 다음과 같습니다.

```powershell
python run_experiments.py --output results_quick --seeds 2 --quick
```

생성 파일:

- `results/raw_results.csv`
- `results/summary.csv`
- `results/plots/objective_gap_by_users.svg`
- `results/plots/runtime_by_users.svg`
- `results/plots/convergence_gap_by_rbs.svg`

continuous soft allocation만 별도로 평가하려면 다음을 실행합니다.

```powershell
python run_continuous_experiments.py --output results/continuous --seeds 5
```

이 실험은 rounding/projection으로 0/1 assignment를 복원하지 않고, 최종 자원 할당 자체를 `X in [0, 1]`인 time-sharing 또는 soft allocation으로 해석합니다.

생성 파일:

- `results/continuous/raw_results.csv`
- `results/continuous/summary.csv`
- `results/continuous/plots/soft_objective_gap_by_users.svg`
- `results/continuous/plots/fractional_mass_by_users.svg`
- `results/continuous/plots/soft_runtime_by_users.svg`

## 결과

개선 방법을 포함해 `--seeds 5` 조건으로 전체 실험을 다시 실행했습니다. 결과는 `raw_results.csv` 550개 row, `summary.csv` 110개 row로 생성되었습니다.

시각적 plot:

- [사용자 수에 따른 objective gap](results/plots/objective_gap_by_users.svg)
- [사용자 수에 따른 runtime](results/plots/runtime_by_users.svg)
- [RB 수에 따른 convergence-gap objective](results/plots/convergence_gap_by_rbs.svg)

### 전체 평균

| 방법 | Hungarian 대비 평균 objective gap | 최악 gap | 평균 runtime | 평균 FL quality |
|---|---:|---:|---:|---:|
| `Hungarian` | 0.00% | 0.00% | 0.093 ms | 0.7288 |
| `LP-Relax+Projection` | 0.00% | 0.00% | 6.580 ms | 0.7288 |
| `Hybrid-Score-Greedy(alpha=0.25)` | 10.30% | 39.79% | 2.788 ms | 0.7238 |
| `Greedy-Cost` | 16.41% | 89.39% | 0.225 ms | 0.7213 |
| `Entropy-Relax(tau=0.8)+Projection` | 30.20% | 121.58% | 5.735 ms | 0.6710 |
| `Entropy-Relax(tau=0.3)+Projection` | 30.50% | 121.58% | 5.533 ms | 0.6708 |
| `Entropy-Relax(tau=0.1)+Projection` | 32.98% | 126.44% | 5.051 ms | 0.6651 |
| `Entropy-Relax(tau=0.3)+Greedy` | 34.25% | 90.19% | 5.367 ms | 0.6720 |
| `Entropy-Relax(tau=0.8)+Greedy` | 36.47% | 94.26% | 5.899 ms | 0.6716 |
| `Entropy-Relax(tau=0.1)+Greedy` | 37.16% | 110.07% | 5.115 ms | 0.6642 |

### 사용자 수 sweep

`R=12`로 고정했을 때, `LP-Relax+Projection`은 모든 사용자 수에서 Hungarian과 동일한 objective를 냈습니다. 이는 단순화된 assignment 제약의 LP relaxation이 integral extreme point를 갖는다는 예상과 일치합니다.

| 사용자 수 | Hungarian quality | Greedy gap | Hybrid gap | best entropy projection gap | LP gap |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.9855 | 35.31% | 30.60% | 0.02% | 0.00% |
| 15 | 0.9144 | 15.81% | 7.64% | 121.58% | 0.00% |
| 20 | 0.7793 | 5.37% | 3.44% | 59.73% | 0.00% |
| 30 | 0.5831 | 1.49% | 0.73% | 17.94% | 0.00% |
| 50 | 0.3904 | 0.57% | 0.33% | 13.10% | 0.00% |
| 80 | 0.2462 | 0.34% | 0.08% | 5.79% | 0.00% |

`Hybrid-Score-Greedy`는 사용자 수가 15 이상인 대부분의 구간에서 기존 `Greedy-Cost`보다 gap을 줄였습니다. 특히 `U=80`에서는 gap이 `0.34%`에서 `0.08%`로 감소했습니다.

### RB 수 sweep

`U=15`로 고정했을 때 RB 수가 증가하면 Hungarian의 FL quality는 `R=5`의 0.4810에서 `R=20`의 0.9852로 증가했습니다. convergence-gap objective도 875.75에서 23.11로 감소했습니다.

| RB 수 | Hungarian quality | Hungarian convergence gap | Greedy gap | Hybrid gap | best entropy projection gap | LP gap |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.4810 | 875.75 | 1.37% | 0.84% | 11.09% | 0.00% |
| 9 | 0.7791 | 334.68 | 4.59% | 4.68% | 34.98% | 0.00% |
| 12 | 0.8911 | 177.10 | 7.41% | 8.28% | 54.87% | 0.00% |
| 15 | 0.9821 | 28.22 | 89.39% | 39.79% | 0.08% | 0.00% |
| 20 | 0.9852 | 23.11 | 18.91% | 16.91% | 0.12% | 0.00% |

`Entropy-Relax+Projection`은 RB가 충분한 `R=15,20`에서는 매우 좋은 성능을 보였습니다. 예를 들어 `R=15`에서 기존 greedy gap은 89.39%였지만, `Entropy-Relax(tau=0.3)+Projection`은 0.08%까지 낮아졌습니다. 반면 RB가 부족하거나 중간 규모인 `R=9,12`에서는 projection이 여전히 큰 gap을 보였습니다.

## 평가

첫째, `LP-Relax+Projection`은 모든 sweep point에서 Hungarian과 동일한 objective를 냈습니다. 따라서 현재처럼 `P*_{i,n}`과 `qhat_{i,n}`을 고정한 선형 assignment 문제에서는 LP relaxation이 성능상 손실 없는 대안입니다. 다만 평균 runtime은 4.588 ms로 Hungarian의 0.125 ms보다 느려서, 속도 개선 방법으로 보기는 어렵습니다.

둘째, 추가한 방법 중 가장 안정적으로 개선된 것은 `Hybrid-Score-Greedy(alpha=0.25)`입니다. 평균 objective gap이 기존 `Greedy-Cost`의 16.41%에서 10.30%로 감소했고, 평균 FL quality도 0.7213에서 0.7238로 상승했습니다. runtime은 2.788 ms로 greedy보다 느리지만 LP보다 빠릅니다.

셋째, `Entropy-Relax+Projection`은 특정 조건에서 강력하지만 안정적이지 않습니다. RB가 충분한 경우에는 Hungarian에 거의 근접하지만, 사용자 수 sweep의 `U=15`와 같은 일부 설정에서는 gap이 121.58%까지 커졌습니다. 즉 projection만 붙인다고 항상 성능이 보장되지는 않으며, soft matrix와 원래 cost를 함께 고려하는 projection이 필요합니다.

결론적으로, 현재 실험 기준에서 추천 순서는 다음과 같습니다.

1. 정확도가 가장 중요하면 `Hungarian` 또는 `LP-Relax+Projection`.
2. hard matching 없이 빠른 근사와 품질 절충이 필요하면 `Hybrid-Score-Greedy(alpha=0.25)`.
3. differentiable soft allocation이 연구 목적이라면 `Entropy-Relax+Projection`을 쓰되, cost-aware projection으로 추가 개선해야 합니다.

## Continuous Soft Allocation 실험

위 결과는 모두 최종적으로 0/1 hard assignment를 복원하는 실험입니다. 추가 실험에서는 기존 결과를 유지한 채, soft allocation 자체를 허용하는 별도 convex pipeline을 구성했습니다. 비교 대상은 다음과 같습니다.

- `Hungarian(reference)`: hard matching 기준선. soft 실험의 objective gap 기준으로만 사용합니다.
- `Continuous-LP(HiGHS)`: 선형 objective와 선형 assignment 제약을 그대로 `0 <= X <= 1`에서 풉니다. 이는 convex LP입니다.
- `Soft-Entropy-KKT(tau=1,5,25)`: linear objective에 convex entropy regularizer를 더한 문제를 풉니다. 최종 `X`를 rounding하지 않으므로 실제 soft allocation입니다.

entropy-KKT 문제는 다음 형태입니다.

```text
min_X <C, X> + tau * sum_{i,n} x_{i,n}(log x_{i,n} - 1)
s.t.  sum_n x_{i,n} <= 1,  sum_i x_{i,n} <= 1,  x_{i,n} >= 0
```

부등식 제약은 dummy user/RB를 추가해 balanced equality scaling 문제로 바꾸었습니다. KKT stationarity는 다음 Gibbs/Sinkhorn 형태를 갖습니다.

```text
x_{i,n} = u_i * exp(-C_{i,n} / tau) * v_n
```

따라서 `Soft-Entropy-KKT`는 일반 nonlinear solver를 호출하지 않고, KKT 조건에서 유도한 Sinkhorn scaling으로 풉니다. `Continuous-LP(HiGHS)`만 `scipy.optimize.linprog(method="highs")`를 사용합니다.

시각적 plot:

- [continuous objective gap](results/continuous/plots/soft_objective_gap_by_users.svg)
- [fractional mass ratio](results/continuous/plots/fractional_mass_by_users.svg)
- [continuous runtime](results/continuous/plots/soft_runtime_by_users.svg)

### Continuous 전체 평균

`--seeds 5` 실행 결과 `raw_results.csv` 275개 row, `summary.csv` 55개 row가 생성되었습니다.

| 방법 | Hungarian 대비 평균 linear gap | 최악 평균 gap | 평균 fractional mass ratio | 평균 KKT residual | 평균 runtime | 평균 FL quality |
|---|---:|---:|---:|---:|---:|---:|
| `Hungarian(reference)` | 0.00% | 0.00% | 0.0000 | 0.000e+00 | 0.051 ms | 0.7281 |
| `Continuous-LP(HiGHS)` | 0.00% | 0.00% | 0.0000 | 0.000e+00 | 2.887 ms | 0.7281 |
| `Soft-Entropy-KKT(tau=1)` | 10.28% | 52.48% | 1.0000 | 1.918e-04 | 3.257 ms | 0.7254 |
| `Soft-Entropy-KKT(tau=5)` | 48.83% | 211.22% | 1.0000 | 3.417e-11 | 0.992 ms | 0.7125 |
| `Soft-Entropy-KKT(tau=25)` | 250.57% | 975.23% | 1.0000 | 6.293e-12 | 0.261 ms | 0.6343 |

### Continuous sweep 해석

`Continuous-LP(HiGHS)`는 모든 사용자 수/RB 수 sweep에서 Hungarian과 같은 objective를 냈습니다. 이는 현재처럼 `P*_{i,n}`과 `qhat_{i,n}`을 상수로 고정한 선형 assignment polytope가 integral extreme point를 갖기 때문입니다. 따라서 LP relaxation은 convex이지만, 결과적으로는 soft allocation이 아니라 Hungarian과 동등한 hard allocation으로 수렴합니다.

`Soft-Entropy-KKT(tau=1)`은 fractional mass ratio가 평균 1.0000으로, 최종 allocation이 거의 전부 fractional입니다. 평균 linear gap은 10.28%였지만 FL quality는 0.7254로 Hungarian의 0.7281에 가깝게 유지되었습니다. 사용자 수가 많아져 RB 경쟁이 완화되는 구간에서는 gap이 크게 줄었습니다. 예를 들어 `U=80, R=12`에서는 gap이 0.09%, FL quality가 0.2426으로 Hungarian의 0.2433과 거의 동일했습니다.

반대로 RB가 매우 충분한 구간에서는 entropy smoothing의 비용이 커집니다. `U=15, R=20`에서 `tau=1`의 gap은 52.48%였고, `tau=5`와 `tau=25`는 각각 211.22%, 975.23%까지 증가했습니다. 이는 optimum이 소수의 매우 좋은 RB에 집중되는 상황에서 entropy가 mass를 넓게 분산시키기 때문입니다.

결론적으로, 전체 파이프라인을 convex continuous optimization으로 유지하려면 최종 soft allocation을 time-sharing 또는 probabilistic allocation으로 인정해야 합니다. 이 경우 `Soft-Entropy-KKT(tau=1)`이 가장 합리적인 절충점입니다. 반면 실제 시스템이 반드시 0/1 RB 할당을 요구하면 rounding/projection 단계가 다시 필요하며, 그 순간 전체 파이프라인은 완전한 convex optimization만으로 닫히지 않습니다.
