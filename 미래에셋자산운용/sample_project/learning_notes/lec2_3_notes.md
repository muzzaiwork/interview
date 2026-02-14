# Lec 2-3: 성과 평가 지표 (Performance Indicators)

이 장에서는 투자 전략의 우수성을 정량적으로 평가하기 위한 핵심 지표들을 학습합니다. 연율화된 수익률과 변동성, CAGR, 샤프 지수, MDD 등 퀀트 실무에서 가장 빈번하게 사용되는 성과 측정 도구들을 다룹니다.

## 1. 연율화 (Annualization)

데이터의 관측 주기(일간, 주간 등)와 관계없이 동일한 기준(연 단위)으로 비교하기 위해 연율화 과정이 필요합니다.

### 1.1. 연율화 수익률 (Annualized Return)
로그 수익률을 사용하는 경우, 일평균 수익률에 연간 영업일수(보통 252일)를 곱하여 계산합니다.

```python
import pandas as pd
import numpy as np
import FinanceDataReader as fdr

df = fdr.DataReader("069500", '2019-01-02', '2020-10-30')
log_rtn = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)

# 연율화 수익률 = 일평균 로그 수익률 * 252
ann_rtn = log_rtn.mean() * 252
print(f"연율화 수익률: {ann_rtn:.4f}")
# 출력: 연율화 수익률: 0.1005
```

### 1.2. 연율화 변동성 (Annualized Volatility)
수익률의 표준편차에 영업일수의 제곱근($\sqrt{252}$)을 곱하여 계산합니다.

```python
# 연율화 표준편차 = 일간 표준편차 * sqrt(252)
ann_std = log_rtn.std() * np.sqrt(252)
print(f"연율화 변동성: {ann_std:.4f}")
# 출력: 연율화 변동성: 0.2312
```

---

## 2. CAGR (Compound Annual Growth Rate)

연 복리 수익률을 의미하며, 투자 기간 전체의 성과를 연 단위로 환산한 지표입니다.

```python
# CAGR = (최종가치 / 최초가치) ^ (252 / 총 영업일수) - 1
cum_rtn = (1 + df['Close'].pct_change().fillna(0)).cumprod()
total_days = len(cum_rtn)
cagr = cum_rtn.iloc[-1] ** (252 / total_days) - 1

print(f"CAGR: {cagr:.4f}")
# 출력: CAGR: 0.1057
```

---

## 3. 위험 조정 수익률: 샤프 지수 (Sharpe Ratio)

단순 수익률이 아닌, 위험(변동성) 한 단위당 초과 수익이 얼마인지를 나타냅니다.

```python
rfr = 0.025 # 무위험 수익률 (예: 국고채 금리 2.5%)
sharpe = (ann_rtn - rfr) / ann_std

print(f"샤프 지수: {sharpe:.4f}")
# 출력: 샤프 지수: 0.3264
```

---

## 4. 낙폭 분석: MDD (Maximum Drawdown)

특정 기간 동안 전고점 대비 발생한 최대 손실 폭을 의미하며, 리스크 관리에서 가장 중요한 지표 중 하나입니다.

```python
# 1. 전고점(CumMax) 계산
cummax = cum_rtn.cummax()

# 2. 전고점 대비 하락률(Drawdown) 계산
drawdown = cum_rtn / cummax - 1

# 3. 최대 낙폭(MDD) 추출
mdd = drawdown.min()

print(f"MDD: {mdd:.4f}")
# 출력: MDD: -0.3465
```

---

## 5. 학습 포인트 (퀀트 투자 관점)

1.  **로그 수익률의 통계적 우수성**: 로그 수익률은 합산이 가능하기 때문에 연율화 수익률을 구할 때 단순히 `mean * 252`로 계산할 수 있어 매우 편리합니다.
2.  **위험의 연율화**: 변동성을 연율화할 때 영업일수 자체가 아닌 **제곱근($\sqrt{n}$)**을 곱하는 이유는 분산(Variance)이 시간에 비례하여 증가한다는 통계적 가정 때문입니다.
3.  **Drawdown의 의미**: 수익률이 좋아도 MDD가 너무 깊으면(예: -50% 이상) 투자자가 전략을 유지하기 어렵습니다. 샤프 지수와 함께 MDD를 반드시 확인해야 합니다.

---

## 6. 주요 라이브러리 함수 정리

| 라이브러리 | 클래스/모듈 | 함수명 | 설명 |
| :--- | :--- | :--- | :--- |
| **Pandas** | `Series/DataFrame` | `cummax()` | 누적 최대값 계산 (Drawdown 산출 시 필수) |
| **Pandas** | `Series/DataFrame` | `mean()` | 산술 평균 계산 |
| **Pandas** | `Series/DataFrame` | `std()` | 표준편차 계산 |
| **Pandas** | `Series/DataFrame` | `min() / max()` | 최소값 / 최대값 추출 |
| **NumPy** | `np` | `sqrt()` | 제곱근 계산 |
| **NumPy** | `np` | `exp()` | 지수 함수 연산 |
