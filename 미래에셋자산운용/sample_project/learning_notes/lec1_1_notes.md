# Lec 1-1: Pandas 기초 - Series와 DataFrame 이해

이 장에서는 파이썬 데이터 분석의 핵심 라이브러리인 **Pandas**의 기본 자료구조인 `Series`와 `DataFrame`을 학습합니다. 특히 퀀트 투자에서 시계열 데이터를 다룰 때 가장 중요한 '인덱스(Index) 기반 연산'과 '재색인(Reindexing)'의 개념을 정립합니다.

## 1. 코드와 실행 결과 (Step-by-Step)

### 1.1. Series 생성과 주요 함수
Pandas의 `Series`는 인덱스를 가진 1차원 데이터입니다. 다양한 생성 방법과 결측치(`NaN`) 처리 함수를 익힙니다.

```python
import pandas as pd
import numpy as np

# 1. 리스트와 인덱스를 지정하여 생성
s2 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print("인덱스 지정 Series (s2):")
print(s2.head(2))
# 출력:
# a    1
# b    2
# dtype: int64

# 2. Dictionary를 이용하여 생성
s2_dict = pd.Series({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})
print("Dict 기반 Series:")
print(s2_dict.head(2))
# 출력:
# a    1
# b    2
# dtype: int64

# 3. 주요 속성 및 함수
s = pd.Series([10, 0, 1, 1, 2, 3, 4, 5, 6, np.nan])
print(f"전체 길이 (len): {len(s)}")      # 출력: 전체 길이 (len): 10
print(f"데이터 형태 (shape): {s.shape}")  # 출력: 데이터 형태 (shape): (10,)
print(f"데이터 개수 (count, NaN 제외): {s.count()}") # 출력: 데이터 개수 (count, NaN 제외): 9
print(f"고유값 (unique): {s.unique()}")   # 출력: 고유값 (unique): [10.  0.  1.  2.  3.  4.  5.  6. nan]

print("값별 개수 (value_counts):")
print(s.value_counts().head(2))
# 출력:
# 1.0     2
# 10.0    1
# dtype: int64
```

---

### 1.2. 인덱스 기반 자동 정렬 연산
Pandas는 데이터의 순서가 아니라 **인덱스 라벨**을 기준으로 자동으로 매칭하여 연산합니다.

```python
s3 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
s4 = pd.Series([4, 3, 2, 1], index=['d', 'c', 'b', 'a'])

print("s3 + s4 (자동 정렬 연산 결과):")
print(s3 + s4)
# 출력:
# a    2
# b    4
# c    6
# d    8
# dtype: int64
```

---

### 1.3. DataFrame 생성 방법
`DataFrame`은 여러 개의 `Series`가 모여서 만들어진 2차원 표 형태의 자료구조입니다.

```python
# 1. Dictionary를 이용한 생성 (가장 흔한 방법)
s1 = np.arange(1, 6, 1)
s2 = np.arange(6, 11, 1)
df = pd.DataFrame({'c1': s1, 'c2': s2})
print("Dict 기반 DataFrame:")
print(df.head(2))
# 출력:
#    c1  c2
# 0   1   6
# 1   2   7

# 2. 리스트와 인덱스/컬럼명을 지정하여 생성
df2 = pd.DataFrame([[10, 11], [10, 12]], columns=['a', 'b'], index=['r1', 'r2'])
print("리스트 기반 DataFrame:")
print(df2)
# 출력:
#      a   b
# r1  10  11
# r2  10  12

# 3. NumPy 배열로부터 생성
df3 = pd.DataFrame(np.array([[10, 11], [20, 21]]), columns=['x', 'y'])
print("NumPy 배열 기반 DataFrame:")
print(df3)
# 출력:
#     x   y
# 0  10  11
# 1  20  21

# 4. 여러 Series(리스트)를 행으로 쌓아서 생성
df4 = pd.DataFrame([np.arange(10, 13), np.arange(15, 18)])
print("행 쌓기 방식 DataFrame:")
print(df4)
# 출력:
#     0   1   2
# 0  10  11  12
# 1  15  16  17
```

---

### 1.4. Series 간 인덱스 자동 정렬 (DataFrame 생성 시)
DataFrame을 생성할 때 포함되는 Series들의 인덱스가 서로 달라도, Pandas는 이를 자동으로 합치고 정렬합니다.

```python
s1 = pd.Series(np.arange(1, 6, 1), index=['a', 'b', 'c', 'd', 'e'])
s2 = pd.Series(np.arange(6, 11, 1), index=['b', 'c', 'd', 'f', 'g'])

df_aligned = pd.DataFrame({'c1': s1, 'c2': s2})
print(df_aligned)
# 출력:
#     c1    c2
# a  1.0   NaN
# b  2.0   6.0
# c  3.0   7.0
# d  4.0   8.0
# e  5.0   NaN
# f  NaN   9.0
# g  NaN  10.0

# 새로운 컬럼 추가 (기존 인덱스에 맞춰 자동 배치)
df_aligned['c3'] = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'f'])
print(df_aligned)
# 출력:
#     c1    c2   c3
# a  1.0   NaN  1.0
# b  2.0   6.0  2.0
# c  3.0   7.0  3.0
# d  4.0   8.0  NaN
# e  5.0   NaN  NaN
# f  NaN   9.0  4.0
# g  NaN  10.0  NaN
```

---

### 1.5. Reindexing 및 결측치 채우기 (ffill)
새로운 인덱스에 맞춰 데이터를 재배치하고, 빈 공간을 이전 값으로 채우는(`ffill`) 기법은 퀀트 실무에서 매우 자주 쓰입니다.

```python
s3 = pd.Series(['red', 'green', 'blue'], index=[0, 3, 5])
# 0~6까지의 인덱스로 재구성하면서 ffill 적용
s_reindexed = s3.reindex(np.arange(0, 7), method='ffill')
print(s_reindexed)
# 출력:
# 0      red
# 1      red
# 2      red
# 3    green
# 4    green
# 5     blue
# 6     blue
```

---

### 1.6. 실무 예제: FinanceDataReader를 활용한 데이터 정합성 맞추기
서로 다른 상장일이나 휴장일을 가진 종목 데이터를 합칠 때 `reindex`와 `ffill`을 활용합니다.

```python
import FinanceDataReader as fdr

# 삼성전자(df1)와 KODEX 200(df2) 데이터 로드
df1 = fdr.DataReader("005930", '2025-01-02', '2025-01-10')
df2 = fdr.DataReader("069500", '2025-01-02', '2025-01-10')

# df2에서 특정 날짜가 누락된 상황 가정 (데이터 불일치)
df2_dropped = df2.drop(df2.index[1]) 

# 삼성전자(df1)의 인덱스를 기준으로 KODEX 200 데이터를 재정렬
new_df2 = df2_dropped.reindex(df1.index)
print(new_df2[['Close']].head(3))
# 출력 (NaN 발생):
#               Close
# Date
# 2025-01-02  31252.0
# 2025-01-03      NaN
# 2025-01-06  32562.0

# ffill로 누락된 데이터 채우기
print(new_df2[['Close']].ffill().head(3))
# 출력:
#               Close
# Date
# 2025-01-02  31252.0
# 2025-01-03  31252.0
# 2025-01-06  32562.0
```

---

## 2. 주요 개념 및 학습 포인트

### 2.1. Series: Index가 있는 1차원 배열
`Series`는 NumPy의 `ndarray`와 비슷하지만, 숫자가 아닌 **라벨(Label)**을 인덱스로 사용할 수 있다는 점이 다릅니다.
- **데이터 자동 정렬**: 두 Series를 더할 때, 데이터의 순서가 아니라 **인덱스 라벨을 기준으로 매칭**하여 연산합니다.
- **NaN (Not a Number)**: 데이터가 비어 있음을 의미하며, Pandas의 많은 함수들(`count()` 등)은 이를 자동으로 제외하고 계산합니다.

### 2.2. DataFrame: 데이터 분석의 표준
`DataFrame`은 여러 개의 `Series`가 모여서 만들어진 2차원 표 형태의 자료구조입니다.
- 각 열(Column)은 서로 다른 데이터 타입을 가질 수 있습니다.
- 행(Row) 인덱스를 공유하므로, 엑셀 시트와 유사한 구조를 가집니다.

### 2.3. Reindexing (재색인)
기존 데이터를 새로운 인덱스에 맞춰 재배열하는 과정입니다.
- **금융 데이터에서의 중요성**: 서로 다른 종목의 데이터를 합칠 때, 상장일이나 휴장일 차이로 인해 인덱스(날짜)가 일치하지 않는 경우가 많습니다. 이때 `reindex`를 통해 기준이 되는 날짜 인덱스로 통일할 수 있습니다.
- **ffill (Forward Fill)**: 재색인 과정에서 발생하는 빈 공간(NaN)을 이전의 유효한 값으로 채우는 방식입니다. 주식 시장에서 '어제의 가격을 오늘의 가격으로 간주'하는 로직 등에 자주 쓰입니다.

---

## 3. 퀀트 투자 관점에서의 의미

### 💡 데이터 파이프라인의 정합성
퀀트 백테스팅을 수행할 때 가장 흔히 발생하는 오류 중 하나가 **'날짜 불일치'**입니다. 
- 예를 들어, KOSPI 지수와 삼성전자 주가를 더하려고 하는데 삼성전자가 거래 정지된 날이 있다면 인덱스가 어긋나게 됩니다.
- Pandas는 인덱스를 기준으로 연산을 수행하므로, 어긋난 날짜에 대해서는 자동으로 `NaN` 처리를 하여 잘못된 계산이 일어나는 것을 방지합니다.

### 💡 생존 편향(Survivorship Bias) 방지
`reindex`를 적절히 활용하면 특정 시점에 상장되어 있던 모든 종목의 리스트를 기준으로 데이터를 재구성할 수 있어, 현재 상장된 종목들로만 테스트하는 '생존 편향'을 방지하는 기초를 마련할 수 있습니다.

---

## 4. 시각적 이해

### Series 연산 원리 (Index Alignment)
```text
s3 (Index: a, b, c, d)    s4 (Index: d, c, b, a)
[ a: 1 ]                [ d: 4 ]          ==>  [ a: 1 + 1 = 2 ]
[ b: 2 ]                [ c: 3 ]          ==>  [ b: 2 + 2 = 4 ]
[ c: 3 ]       +        [ b: 2 ]          ==>  [ c: 3 + 3 = 6 ]
[ d: 4 ]                [ a: 1 ]          ==>  [ d: 4 + 4 = 8 ]

* 데이터 위치가 달라도 '라벨'을 찾아가서 더합니다.
```

---

## 5. 학습 결론
Pandas의 가장 큰 강력함은 **"인덱스를 통한 데이터 관리"**에 있습니다. 단순히 데이터를 담는 그릇을 넘어, 데이터 간의 관계를 인덱스로 정의함으로써 복잡한 금융 시계열 데이터를 안전하고 빠르게 처리할 수 있게 해줍니다.

다음 장에서는 실제 재무제표 데이터를 활용하여 **탐색적 데이터 분석(EDA)**을 수행하는 방법을 알아보겠습니다.

---

## 6. 주요 라이브러리 함수 정리

| 라이브러리 | 클래스/모듈 | 함수명 | 설명 |
| :--- | :--- | :--- | :--- |
| **Pandas** | `pd` | `Series()` | 1차원 데이터 구조 생성 (인덱스 라벨 포함 가능) |
| **Pandas** | `pd` | `DataFrame()` | 2차원 표 형태의 데이터 구조 생성 |
| **Pandas** | `Series/DataFrame` | `head()` / `tail()` | 데이터의 앞부분 또는 뒷부분 일부를 확인 |
| **Pandas** | `Series/DataFrame` | `count()` | 결측치를 제외한 데이터 개수 확인 |
| **Pandas** | `Series` | `unique()` | 중복을 제거한 고유값 목록 반환 |
| **Pandas** | `Series` | `value_counts()` | 고유값별 출현 빈도 계산 |
| **Pandas** | `Series/DataFrame` | `reindex()` | 새로운 인덱스를 기반으로 데이터를 재배열 |
| **Pandas** | `DataFrame` | `drop()` | 특정 행이나 열을 삭제 |
| **Pandas** | `Series/DataFrame` | `ffill()` | 앞의 유효한 값으로 결측치를 채움 (Forward Fill) |
| **Pandas** | `pd` | `to_datetime()` | 문자열 등을 판다스 Timestamp/Datetime 객체로 변환 |
| **Pandas** | `pd` | `set_option()` | 출력 형식(float 포맷, 최대 컬럼 수 등) 설정 |
| **NumPy** | `np` | `arange()` | 일정한 간격의 숫자를 가진 배열 생성 |
| **NumPy** | `np` | `nan` | 결측치(NaN)를 나타내는 상수 |
| **FinanceDataReader** | `fdr` | `DataReader()` | 종목 코드와 기간을 입력받아 주가/지수 데이터 로드 |
