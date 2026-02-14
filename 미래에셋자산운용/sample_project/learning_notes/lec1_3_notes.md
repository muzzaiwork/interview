# Lec 1-3: 데이터 범주화와 그룹 분석 (Categorizing & Groupby)

이 장에서는 데이터를 특정 기준에 따라 나누고(Categorizing), 그룹별로 통계량을 산출하는 **Groupby** 기법을 학습합니다. 이는 퀀트 투자에서 포트폴리오를 구성하고 각 그룹(예: 저PBR 그룹 vs 고PBR 그룹)의 수익률을 비교 분석할 때 필수적인 도구입니다.

## 1. 데이터 준비 및 수익률 산출

분석을 위해 2016년 12월 재무제표 데이터를 로드하고, 이후 1년간의 수익률(`rtn`)을 계산합니다.

```python
import numpy as np
import pandas as pd

# 데이터 로드 (2016년 12월 재무제표)
df = pd.read_csv("미래에셋자산운용/sample_project/my_data/naver_finance/2016_12.csv")

# 수익률 구하기 (16.12 대비 17.12 가격 변동률)
df['rtn'] = df['price2'] / df['price'] - 1
```

---

## 2. 데이터 범주화 (Categorizing)

특정 지표(예: PER)를 기준으로 종목들을 여러 그룹으로 나누는 방법입니다.

### 2.1. Boolean Selection과 Loc을 활용한 수동 그룹화
조건문을 직접 작성하여 그룹 번호를 부여합니다.

```python
# 조건 설정
bound1 = df['PER(배)'] >= 10
bound2 = (5 <= df['PER(배)']) & (df['PER(배)'] < 10)
bound3 = (0 <= df['PER(배)']) & (df['PER(배)'] < 5)
bound4 = df['PER(배)'] < 0

# 그룹 번호 부여
df.loc[bound1, 'PER_Score'] = 1
df.loc[bound2, 'PER_Score'] = 2
df.loc[bound3, 'PER_Score'] = 3
df.loc[bound4, 'PER_Score'] = -1

# 결측치 처리 (NaN을 0으로 채움)
df.loc[df['PER_Score'].isna(), "PER_Score"] = 0

print(df['PER_Score'].value_counts())
# 출력:
# PER_Score
#  1.0    378
#  2.0    148
# -1.0    120
#  3.0     23
#  0.0     12
# Name: count, dtype: int64
```

### 2.2. pd.cut()을 활용한 절대값 기준 분할
값의 범위를 지정하여 데이터를 나눕니다. `labels`를 사용하여 그룹에 이름을 붙일 수 있습니다.

```python
# 절대 구간 기준 분할
bins = [-np.inf, 10, 20, np.inf]
labels = ['저평가주', '보통주', '고평가주']
df['PER_Label'] = pd.cut(df['PER(배)'], bins=bins, labels=labels)

print(df['PER_Label'].head())
# 출력:
# 0     보통주
# 1    고평가주
# 2    저평가주
# 3     보통주
# 4    고평가주
# Name: PER_Label, dtype: category
```

### 2.3. pd.qcut()을 활용한 상대적 비율(분위수) 분할
데이터를 동일한 개수가 들어가도록 등분합니다. 퀀트 포트폴리오 구성 시 가장 많이 사용되는 방식입니다.

```python
# PER 기준 10등분 (데실 분석용)
df['PER_Score_Q'] = pd.qcut(df['PER(배)'], 10, labels=range(1, 11))

print(df['PER_Score_Q'].value_counts().head(3))
# 출력:
# PER_Score_Q
# 1    67
# 2    67
# 3    67
# Name: count, dtype: int64
```

---

## 3. Groupby: 분할-적용-결합 (Split-Apply-Combine)

데이터를 그룹으로 묶고, 각 그룹에 함수를 적용하여 결과를 요약하는 과정입니다.

### 3.1. Groupby 객체와 Aggregation
`groupby()`는 그룹화 가능 여부만 검증하며, 실제 연산은 `agg()` 또는 집계 함수(`mean`, `sum` 등)를 호출할 때 발생합니다.

```python
# 결측치 제거 후 복사본 생성
g_df = df.dropna().copy()
g_df.set_index('ticker', inplace=True)

# PBR 기준 그룹별 평균 수익률 계산
pbr_rtn = g_df.groupby("PBR_score").agg({'rtn': 'mean'})
print(pbr_rtn.head(2))
# 출력:
#                rtn
# PBR_score
# 1         -0.001363
# 2          0.020453
```

### 3.2. 다중 인덱스 및 다중 집계
여러 컬럼을 기준으로 그룹화하거나, 여러 개의 통계량을 한 번에 산출할 수 있습니다.

```python
# PBR과 PER 점수를 기준으로 수익률의 평균과 표준편차 계산
g_results = g_df.groupby(["PBR_score", "PER_score"]).agg({
    'rtn': ['mean', 'std'],
    'ROE(%)': ['mean', 'size']
})

print(g_results.head(2))
# 출력:
#                      rtn                 ROE(%)
#                     mean       std         mean size
# PBR_score PER_score
# 1         1        -0.099839  0.071890   -1.401800    5
#           2        -0.093158  0.266421  154.966727   11
```

---

## 4. Multi-index 처리 및 컬럼 병합

`agg()` 사용 시 생성되는 계층적 컬럼 구조(Multi-index)를 다루기 쉽게 병합하는 실무 팁입니다.

```python
# 계층 구조 확인
level0 = g_results.columns.get_level_values(0)
level1 = g_results.columns.get_level_values(1)

# 컬럼명 병합 (예: rtn_mean, rtn_std)
g_results.columns = level0 + '_' + level1
g_results = g_results.reset_index()

print(g_results.columns)
# 출력: Index(['PBR_score', 'PER_score', 'rtn_mean', 'rtn_std', 'ROE(%)_mean', 'ROE(%)_size'], dtype='object')
```

---

## 5. 학습 포인트 (퀀트 투자 관점)

1.  **상대평가의 중요성**: `cut()`은 절대적인 수치(예: PER 10 이하)를 기준으로 나누지만, `qcut()`은 시장 내 순위를 기준으로 나눕니다. 퀀트 전략에서는 시장 상황 변화에 유연하게 대응하기 위해 `qcut()`을 통한 상대평가를 자주 사용합니다.
2.  **포트폴리오 성과 분석**: `groupby`는 포트폴리오를 여러 버킷으로 나누고 각 버킷의 성과(수익률, 리스크 등)를 비교하는 데 핵심적입니다. 예를 들어 PBR이 가장 낮은 1분위 그룹이 10분위 그룹보다 높은 수익률을 내는지 검증할 때 사용합니다.
3.  **데이터 편향 주의**: `groupby` 수행 시 `NaN` 값은 자동으로 그룹에서 제외됩니다. 만약 `NaN`이 유의미한 정보라면(예: 상장 폐지 등으로 인한 데이터 유실), 이를 미리 전처리하여 분석 결과가 왜곡되지 않도록 해야 합니다.

---

## 6. 주요 라이브러리 함수 정리

| 라이브러리 | 클래스/모듈 | 함수명 | 설명 |
| :--- | :--- | :--- | :--- |
| **Pandas** | `pd` | `cut()` | 데이터를 동일한 간격 또는 지정한 경계값 기준으로 범주화 |
| **Pandas** | `pd` | `qcut()` | 데이터를 표본 변분위수(Quantile) 기준으로 동일한 개수가 포함되게 범주화 |
| **Pandas** | `DataFrame` | `groupby()` | 지정한 열을 기준으로 데이터를 그룹별로 분할 |
| **Pandas** | `DataFrameGroupBy` | `agg()` | 그룹별로 다양한 집계 함수(sum, mean 등)를 적용 |
| **Pandas** | `DataFrameGroupBy` | `size()` | 각 그룹의 데이터 개수 반환 |
| **Pandas** | `DataFrameGroupBy` | `get_group()` | 특정 그룹의 데이터만 추출 |
| **Pandas** | `Index` | `get_level_values()` | 멀티 인덱스에서 특정 레벨의 인덱스 값을 반환 |
| **Pandas** | `Series/DataFrame` | `fillna()` | 결측치(NaN)를 특정 값이나 방식으로 채움 |
| **Pandas** | `Series/DataFrame` | `astype()` | 데이터 타입을 변경 (예: float -> int) |
| **Pandas** | `Series/DataFrame` | `equals()` | 두 객체의 내용과 타입이 동일한지 비교 |
| **Pandas** | `DataFrame` | `reset_index()` | 인덱스를 일반 컬럼으로 변환하고 기본 정수 인덱스로 초기화 |
| **Pandas** | `Series/DataFrame` | `plot()` | 데이터를 그래프(Kind 지정 가능)로 시각화 |
| **NumPy** | `np` | `inf` | 무한대(Infinity)를 나타내는 상수 |
| **NumPy** | `np` | `nan` | 결측치(NaN)를 나타내는 상수 |
