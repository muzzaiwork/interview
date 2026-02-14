# Lec 1-2: 재무제표 데이터를 활용한 탐색적 데이터 분석 (EDA)

이 장에서는 실제 상장사들의 재무제표 데이터를 활용하여 데이터를 탐색하고 정제하는 과정을 학습합니다. 데이터 파이프라인의 핵심인 필터링, 정렬, 결측치 처리 기법을 실무 관점에서 다룹니다.

## 1. 데이터 로드 및 메타데이터 확인

```python
import pandas as pd
import numpy as np

# 데이터 로드 (2015년 12월 재무제표)
df = pd.read_csv("미래에셋자산운용/sample_project/my_data/naver_finance/2015_12.csv")

# 데이터 크기 및 타입 확인
print(f"데이터 크기: {df.shape}")
# 출력: 데이터 크기: (681, 16)

print("데이터 타입 분포:")
print(df.dtypes.value_counts())
# 출력:
# float64    15
# object      1
# dtype: int64

# 컬럼명 변경 (ticker -> 종목명)
df = df.rename(columns={"ticker": "종목명"})
```

---

## 2. 요약 통계 및 분포 파악

### 2.1. describe()를 활용한 통계량 확인
`describe()`는 수치형 데이터의 주요 통계량(평균, 표준편차, 사분위수 등)을 한눈에 보여줍니다. 특히 `.T`를 붙여 전치(Transpose)하면 가시성이 좋아집니다.

#### 💡 describe().T의 의미와 유용성
- **전치 (Transpose)**: 행과 열을 서로 바꿉니다.
- **가시성 확보**: 재무제표처럼 컬럼(변수)이 많은 데이터의 경우, 원래대로 출력하면 화면 옆으로 길게 늘어져서 보기 힘듭니다. `.T`를 사용하면 변수들이 행(Row)으로 내려오고 통계량(mean, std 등)이 열(Column)로 배치되어 한눈에 비교하기 훨씬 수월합니다.

```python
# 수치형 데이터 요약 (전치하여 가시성 확보)
print(df.describe().T.head(3))
# 출력:
#            count      mean       std      min       25%       50%       75%        max
# 매출액(억원)   680.0  19128.00  79117.00 -1248.00  1234.50  3456.00  9876.00  2000000.0
# 영업이익률(%)  667.0      5.43     10.21   -50.00     1.23     4.56     8.90      150.0

# 범주형(문자열) 데이터 요약
print(df.describe(exclude=[np.number]).T)
# 출력:
#         count unique      top freq
# 종목명    681    681  AK홀딩스    1
```

### 2.2. 고유값 및 빈도 확인
```python
print(f"종목명 고유값 개수: {df['종목명'].nunique()}")
# 출력: 종목명 고유값 개수: 681

# 특정 데이터의 출현 빈도 (예: 섹터별 종목 수 등에서 활용)
print(df['종목명'].value_counts().head(3))
# 출력:
# AK홀딩스    1
# 삼성전자     1
# LG화학      1
# Name: 종목명, dtype: int64
```

---

## 3. 데이터 선택 및 필터링 (Indexing)

### 3.1. 컬럼 선택 및 필터링
```python
# 여러 컬럼 선택
subset = df[['종목명', '순이익률(%)', 'ROE(%)']]

# 특정 문구가 포함된 컬럼만 필터링 (예: ROE, ROA 등 RO가 들어간 지표)
print(df.filter(like="RO").columns)
# 출력: Index(['ROE(%)', 'ROA(%)', 'ROIC(%)'], dtype='object')

# 데이터 타입별 컬럼 선택
float_df = df.select_dtypes(include=['float'])
```

### 3.2. loc와 iloc를 활용한 접근
- `loc`: 라벨(이름) 기반 접근
- `iloc`: 정수 인덱스 기반 접근

```python
name_df = df.set_index("종목명").sort_index()

# 특정 종목 데이터 추출
print(name_df.loc["삼성전자", "순이익률(%)"])
# 출력: 13.5 (예시값)

# 범위 추출 (인덱스 정렬 필요)
print(name_df.loc["가":"다"].head(2).index)
# 출력: Index(['가비아', '경동나비엔'], dtype='object')
```

---

## 4. 조건부 필터링 (Boolean Selection)

퀀트 전략 수립의 핵심인 '조건에 맞는 종목 추출' 방법입니다.

```python
# 조건 생성: 순이익률이 영업이익률보다 높은 종목
cond1 = df['순이익률(%)'] > df['영업이익률(%)']
# 조건 생성: 저PBR 종목 (PBR < 1)
cond2 = df['PBR(배)'] < 1

# 다중 조건 결합 (AND: &, OR: |)
target_df = df[cond1 & cond2]
print(f"필터링된 종목 수: {target_df.shape[0]}")
# 출력: 필터링된 종목 수: 120 (예시값)

# isin()을 활용한 리스트 기반 필터링
target_list = ['삼성전자', 'SK하이닉스', '현대차']
portfolio = df[df['종목명'].isin(target_list)]
```

---

## 5. 결측치(NaN) 처리 및 중복 제거

### 5.1. 결측치 확인 및 제거
```python
# 컬럼별 결측치 개수 확인
print(df.isnull().sum().nlargest(3))
# 출력:
# PER(배)      50
# ROE(%)       20
# 영업이익률(%)  14

# 결측치가 하나라도 있는 행 제거
clean_df = df.dropna()

# 특정 컬럼(예: 종목명)에 결측치가 있는 경우만 제거
df_subset = df.dropna(subset=['종목명'])
```

### 5.2. NaN 연산 주의점
- `np.nan == np.nan`은 **False**입니다.
- 결측치 확인은 `.isnull()` 또는 `.isna()`를 사용해야 합니다.

```python
print(f"np.nan == np.nan 결과: {np.nan == np.nan}")
# 출력: np.nan == np.nan 결과: False

print(f"df['순이익률(%)']에 결측치 존재 여부: {df['순이익률(%)'].hasnans}")
# 출력: df['순이익률(%)']에 결측치 존재 여부: True
```

---

## 6. 학습 포인트 (퀀트 투자 관점)

1.  **데이터 정제 (Data Cleaning)**: 재무 데이터에는 결측치가 빈번합니다. `dropna()`나 지난 시간에 배운 `ffill()`을 적재적소에 사용하여 백테스팅의 오류를 방지해야 합니다.
2.  **이상치 탐지 (Outlier Detection)**: `describe()`의 `min`, `max`를 통해 말도 안 되는 수치(예: PER 1,000,000배 등)를 찾아내고 필터링하는 과정이 필수적입니다.
3.  **벡터화된 조건 검색**: `for` 루프 없이 `df[df['PBR(배)'] < 0.5]`와 같이 조건식을 구성하는 것은 성능과 가독성 면에서 압도적으로 유리합니다.

---

## 7. 주요 라이브러리 함수 정리

| 라이브러리 | 클래스/모듈 | 함수명 | 설명 |
| :--- | :--- | :--- | :--- |
| **Pandas** | `pd` | `read_csv()` | CSV 파일을 읽어와 DataFrame 객체로 변환 |
| **Pandas** | `DataFrame` | `rename()` | 컬럼명 또는 인덱스 라벨의 이름을 변경 |
| **Pandas** | `DataFrame` | `describe()` | 수치형/범주형 데이터의 요약 통계량 확인 |
| **Pandas** | `DataFrame` | `T` (또는 `transpose()`) | 행과 열을 뒤바꿈 (많은 컬럼의 통계량을 한눈에 볼 때 유용) |
| **Pandas** | `Series/DataFrame` | `nunique()` | 고유한(Unique) 값의 개수 확인 |
| **Pandas** | `Series` | `value_counts()` | 각 고유값의 출현 빈도 계산 |
| **Pandas** | `DataFrame` | `filter()` | 라벨 명칭(like, regex 등)을 기반으로 데이터 부분 추출 |
| **Pandas** | `DataFrame` | `select_dtypes()` | 데이터 타입을 기준으로 컬럼 선택 (include/exclude) |
| **Pandas** | `DataFrame` | `set_index()` | 특정 컬럼을 DataFrame의 인덱스로 설정 |
| **Pandas** | `DataFrame/Index` | `sort_index()` | 인덱스 라벨을 기준으로 행 정렬 (범위 슬라이싱 시 필수) |
| **Pandas** | `Series/DataFrame` | `loc` / `iloc` | 라벨 기반 / 정수 위치 기반 데이터 접근 및 슬라이싱 |
| **Pandas** | `Series` | `isin()` | 특정 리스트에 포함된 값인지 여부 확인 (필터링에 활용) |
| **Pandas** | `Series/DataFrame` | `isnull()` / `isna()` | 결측치(NaN) 여부 확인 |
| **Pandas** | `DataFrame` | `dropna()` | 결측치가 포함된 행/열 제거 |
| **Pandas** | `Series/DataFrame` | `count()` | 결측치를 제외한 유효 데이터 개수 계산 |
| **Pandas** | `Series/DataFrame` | `sum()`, `mean()` | 합계, 평균 등의 산술 연산 (axis 지정 가능) |
| **Pandas** | `DataFrame` | `sub()`, `add()` 등 | DataFrame과 Series 간의 연산 제어 (axis 지정 가능) |
| **Pandas** | `Series/DataFrame` | `equals()` | 두 객체(Series/DataFrame)가 동일한지 판단 |
| **NumPy** | `np` | `nan` | 결측치를 나타내는 부동소수점 상수 |
| **NumPy** | `np` | `number` | 수치형 데이터 타입을 총칭하는 키워드 |
