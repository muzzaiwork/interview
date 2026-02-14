# Lec 2-5: 시그널 기반 백테스팅 (Signal-based Backtesting)

이 장에서는 기술적 지표를 활용하여 매수/매도 시그널을 생성하고, 이를 바탕으로 실제 투자 전략의 성과를 시뮬레이션하는 **백테스팅(Backtesting)** 과정을 학습합니다. 이동평균 교차, 모멘텀, 평균 회귀 등 대표적인 퀀트 전략들을 구현합니다.

## 1. 이동평균 교차 전략 (SMA Crossover)

단기 이동평균선이 장기 이동평균선을 상향 돌파할 때 매수하고, 하향 돌파할 때 매도(또는 청산)하는 전략입니다.

```python
import pandas as pd
import numpy as np
import FinanceDataReader as fdr

# 1. 데이터 로드 및 지표 계산
df = fdr.DataReader("005930", '2015-01-01', '2020-12-31')[['Close']]
df['SMA_short'] = df['Close'].rolling(20).mean()
df['SMA_long'] = df['Close'].rolling(60).mean()
df = df.dropna()

# 2. 포지션 결정: 단기 > 장기 이면 1(매수), 아니면 0(현금)
df['position'] = np.where(df['SMA_short'] >= df['SMA_long'], 1, 0)

# 3. 전략 수익률 계산
# 주의: '오늘' 결정된 포지션은 '내일'의 수익률에 영향을 미침 (shift(1))
df['rtn'] = np.log(df['Close'] / df['Close'].shift(1))
df['strategy_rtn'] = (df['position'].shift(1) * df['rtn']).fillna(0)

# 4. 누적 수익률 계산
df['cum_rtn'] = np.exp(df['rtn'].fillna(0).cumsum())
df['cum_strategy_rtn'] = np.exp(df['strategy_rtn'].cumsum())

print(df[['Close', 'position', 'cum_rtn', 'cum_strategy_rtn']].tail())
# 출력:
#             Close  position  cum_rtn  cum_strategy_rtn
# Date
# 2020-12-24  77800         1   2.6995            2.1923
# 2020-12-28  78700         1   2.7307            2.2176
# 2020-12-29  78300         1   2.7169            2.2064
# 2020-12-30  81000         1   2.8105            2.2824
```

---

## 2. 모멘텀 전략 (Momentum)

과거 일정 기간 동안 수익률이 좋았던 자산이 미래에도 좋을 것이라는 가정을 바탕으로 하는 전략입니다.

```python
# 1. 시계열 모멘텀 시그널 (최근 60일 수익률이 양수이면 매수)
df['rtn_60'] = df['Close'].pct_change(60)
df['position'] = np.where(df['rtn_60'] > 0, 1, 0)

# 2. 전략 수익률 계산
df['strategy_rtn'] = (df['position'].shift(1) * df['rtn']).fillna(0)
df['cum_strategy_rtn'] = np.exp(df['strategy_rtn'].cumsum())

print(f"모멘텀 전략 최종 누적 수익률: {df['cum_strategy_rtn'].iloc[-1]:.4f}")
# 출력: 모멘텀 전략 최종 누적 수익률: 2.1827
```

---

## 3. 평균 회귀 전략 (Mean-reversion)

가격이 정상 범위를 벗어났을 때 다시 평균으로 돌아올 것을 기대하는 전략입니다. 볼린저 밴드 하단 돌파 시 매수, 상단 돌파 시 매도(또는 공매도)를 수행합니다.

```python
# 1. 볼린저 밴드 생성
df['SMA'] = df['Close'].rolling(20).mean()
df['std'] = df['Close'].rolling(20).std()
df['Upper'] = df['SMA'] + 2 * df['std']
df['Lower'] = df['SMA'] - 2 * df['std']

# 2. 포지션: 하단 이하 매수(1), 상단 이상 매도(-1)
df['position'] = 0
df.loc[df['Close'] <= df['Lower'], 'position'] = 1
df.loc[df['Close'] >= df['Upper'], 'position'] = -1

# 3. 전략 수익률 계산
df['strategy_rtn'] = (df['position'].shift(1) * df['rtn']).fillna(0)
df['cum_strategy_rtn'] = np.exp(df['strategy_rtn'].cumsum())

print(f"평균 회귀 전략 최종 누적 수익률: {df['cum_strategy_rtn'].iloc[-1]:.4f}")
# 출력: 평균 회귀 전략 최종 누적 수익률: 0.9263
```

---

## 4. 학습 포인트 (퀀트 투자 관점)

1.  **포지션 시프트(`shift(1)`)**: 백테스팅에서 가장 흔히 하는 실수 중 하나가 **Cheating(미래 참조)**입니다. 당일 종가로 계산된 시그널은 당일 매매가 불가능하므로, 반드시 익일 수익률에 적용해야 합니다.
2.  **로그 수익률의 편의성**: 여러 기간의 전략 수익률을 계산할 때 로그 수익률을 사용하면 단순히 더하기(`cumsum`)만으로 누적 성과를 파악할 수 있어 연산이 매우 간결해집니다.
3.  **Drawdown의 시각화**: 전략의 누적 수익률 곡선과 함께 MDD를 시각화하여, 수익률이 좋은 구간에서도 견뎌야 하는 고통(낙폭)의 크기를 확인하는 습관이 중요합니다.

---

## 5. 주요 라이브러리 함수 정리

| 라이브러리 | 클래스/모듈 | 함수명 | 설명 |
| :--- | :--- | :--- | :--- |
| **NumPy** | `np` | `where()` | 조건에 따라 값을 선택하여 배열 생성 (Vectorized If-else) |
| **NumPy** | `np` | `sign()` | 값의 부호를 반환 (1, 0, -1) |
| **Pandas** | `Series/DataFrame` | `pct_change(N)` | N일 전 대비 변화율 계산 (모멘텀 산출 시 유용) |
| **Pandas** | `Series/DataFrame` | `shift(1)` | 데이터를 한 칸 뒤로 밀어 미래 참조 방지 |
| **Pandas** | `Series/DataFrame` | `cumsum() / cumprod()` | 누적 합 / 누적 곱 계산 |
| **Pandas** | `DataFrame` | `filter(like=...)` | 특정 문구를 포함한 컬럼만 선택 |
