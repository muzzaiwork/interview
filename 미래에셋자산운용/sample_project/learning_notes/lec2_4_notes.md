# Lec 2-4: 가격 기반 지표 (Price-based Indicators)

이 장에서는 주가 데이터를 활용하여 다양한 기술적 지표를 산출하는 방법을 학습합니다. 이동평균(SMA, EWMA), 볼린저 밴드, 그리고 시계열 상관계수 등 퀀트 전략의 신호(Signal)를 생성하는 핵심 기법들을 다룹니다.

## 1. 이동평균 (Moving Average)

이동평균은 가격의 노이즈를 제거하고 추세를 파악하는 데 사용됩니다.

### 1.1. 단순 이동평균 (SMA, Simple Moving Average)
일정 기간(window) 동안의 가격 산술 평균입니다.

```python
import pandas as pd
import FinanceDataReader as fdr

df = fdr.DataReader("005930", '2020-01-01', '2020-01-31')[['Close']]
df.columns = ["삼성전자"]

# 5일 이동평균 (기본적으로 앞의 4개는 NaN 발생)
df['5SMA'] = df['삼성전자'].rolling(window=5).mean()

# min_periods 활용 (데이터가 1개만 있어도 계산 시작)
df['5SMA_min1'] = df['삼성전자'].rolling(window=5, min_periods=1).mean()

print(df.head(7))
# 출력:
#              삼성전자      5SMA  5SMA_min1
# Date
# 2020-01-02  55200       NaN  55200.000
# 2020-01-03  55500       NaN  55350.000
# ...
# 2020-01-08  56800 55760.000  55760.000
```

### 1.2. 지수 가중 이동평균 (EWMA, Exponentially-weighted Moving Average)
최근 데이터에 더 높은 가중치를 부여하여 추세 변화를 더 빠르게 반영합니다.

```python
# EWMA (평활계수 alpha=0.2 설정)
df['EWMA_0.2'] = df['삼성전자'].ewm(alpha=0.2).mean()

print(df[['삼성전자', 'EWMA_0.2']].head(3))
# 출력:
#              삼성전자  EWMA_0.2
# Date
# 2020-01-02  55200 55200.000
# 2020-01-03  55500 55366.667
```

---

## 2. 볼린저 밴드 (Bollinger Bands)

이동평균선을 중심으로 표준편차 범위를 설정하여 가격의 변동성을 시각화합니다.

```python
# 20일 이동평균 및 표준편차 기반 상하단 밴드
df_bb = df.copy()
df_bb['20SMA'] = df_bb['삼성전자'].rolling(window=20, min_periods=1).mean()
df_bb['std'] = df_bb['삼성전자'].rolling(window=20, min_periods=1).std()

df_bb['Upper'] = df_bb['20SMA'] + 2 * df_bb['std']
df_bb['Lower'] = df_bb['20SMA'] - 2 * df_bb['std']

print(df_bb[['삼성전자', 'Upper', 'Lower']].tail(3))
# 출력:
#              삼성전자     Upper     Lower
# Date
# 2020-01-31  56400 63526.140 54103.860
```

---

## 3. 시계열 상관관계 (Rolling Correlation)

두 자산 간의 관계가 시간에 따라 어떻게 변하는지 분석합니다. (예: 주식과 채권의 역상관 관계 확인)

```python
df1 = fdr.DataReader("005930", '2020-01-01', '2020-06-30')[['Close']] # 삼성전자
df2 = fdr.DataReader("069500", '2020-01-01', '2020-06-30')[['Close']] # KODEX 200
df_corr = pd.concat([df1, df2], axis=1)
df_corr.columns = ["Samsung", "KODEX200"]

# 20일 롤링 상관계수
rolling_corr = df_corr['Samsung'].rolling(20).corr(df_corr['KODEX200'])

print(rolling_corr.dropna().head(3))
# 출력:
# Date
# 2020-01-31   0.927
# 2020-02-03   0.923
```

---

## 4. 학습 포인트 (퀀트 투자 관점)

1.  **Look-ahead Bias 주의**: 이동평균 계산 시 `rolling` 윈도우가 미래 데이터를 포함하지 않도록 Pandas의 기본 설정을 준수해야 합니다.
2.  **SMA vs EWMA**: 
    *   **SMA**는 단순 추세를 보는 데 좋으나 후행성(Lagging)이 강합니다.
    *   **EWMA**는 최근 변화를 민감하게 반영하므로 단기 트레이딩 신호 생성에 유리합니다.
3.  **동적 자산 배분**: 롤링 상관계수를 통해 자산 간 상관관계가 높아지는 시기(위기 상황 등)를 포착하고 포트폴리오 비중을 조절하는 전략을 세울 수 있습니다.

---

## 5. 주요 라이브러리 함수 정리

| 라이브러리 | 클래스/모듈 | 함수명 | 설명 |
| :--- | :--- | :--- | :--- |
| **Pandas** | `Series/DataFrame` | `rolling()` | 시계열 데이터에 이동 윈도우(Window) 적용 |
| **Pandas** | `Series/DataFrame` | `expanding()` | 윈도우 크기를 누적해서 늘려가며 연산 적용 |
| **Pandas** | `Series/DataFrame` | `ewm()` | 지수 가중(Exponential Weighted) 연산 적용 |
| **Pandas** | `Rolling` | `mean() / std() / corr()` | 윈도우 내 평균 / 표준편차 / 상관계수 계산 |
| **Pandas** | `Rolling` | `apply()` | 윈도우 내 데이터에 사용자 정의 함수 적용 |
| **Pandas** | `DataFrame` | `unstack()` | 멀티인덱스의 특정 레벨을 컬럼으로 변환 (Pivot 효과) |
