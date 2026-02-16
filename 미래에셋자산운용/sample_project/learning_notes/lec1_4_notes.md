# Lec 1-4: 데이터 병합 및 결합 (Combining Pandas Objects)

이 장에서는 여러 개의 Pandas 객체(Series, DataFrame)를 하나로 합치는 다양한 기법들을 학습합니다. 데이터 파이프라인 구축 시 흩어져 있는 재무 데이터나 주가 데이터를 통합할 때 필수적인 `concat`, `join`, `merge`의 차이점과 활용법을 다룹니다.

## 1. DataFrame에 데이터 추가

### 1.1. loc[]를 이용한 행 추가 (In-place)
`loc[]`를 사용하면 기존 DataFrame에 새로운 행을 즉시 추가하거나 수정할 수 있습니다.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame(columns=['a', 'b'])

# 리스트로 추가
df.loc[0] = [1, 2]

# 문자열 인덱스로 추가
df.loc['ㅋㅋ'] = [1, 2]

# 딕셔너리로 추가 (인덱스 계산 포함)
df.loc[len(df)] = {'b' : 'ㅎ', 'a': 'ㅋ'}

# Series로 추가
df.loc["yay"] = pd.Series({'a': 'ㅋ', 'b' : 'ㅎ'})

print(df)
# 출력:
#      a  b
# 0    1  2
# ㅋㅋ   1  2
# 2    ㅋ  ㅎ
# yay  ㅋ  ㅎ
```

### 1.2. concat을 이용한 행 추가
`append()` 함수가 Pandas 최신 버전에서 삭제됨에 따라, 행을 추가할 때도 `pd.concat()`을 사용합니다. 이는 원본을 수정하지 않고 새로운 객체를 반환합니다.

```python
names_df = pd.DataFrame(
    {'Name':['철수', '영희', '영수', '영미'], 'Age':[12, 13, 14, 15]},
    index = ['Canada', 'Canada', 'USA', 'USA']
)

# 새로운 행(DataFrame) 연결
new_row = pd.DataFrame([{'Name':'명수', 'Age':1}])
print(pd.concat([names_df, new_row]))
# 출력:
#        Name  Age
# Canada   철수   12
# Canada   영희   13
# USA      영수   14
# USA      영미   15
# 0        명수    1

# 인덱스 무시하고 새로 부여 (ignore_index=True)
print(pd.concat([names_df, pd.DataFrame([{'Name':'명수', 'Age':100}])], ignore_index=True))
# 출력:
#   Name  Age
# 0   철수   12
# 1   영희   13
# 2   영수   14
# 3   영미   15
# 4   명수  100
```

---

## 2. concat(): 수직/수평 연결

`concat()`은 여러 객체를 축(axis)을 기준으로 단순 연결합니다. 인덱스나 컬럼명이 일치하지 않으면 `NaN`으로 채워집니다 (기본값: Outer Join).

```python
import FinanceDataReader as fdr

# 데이터 로드 (삼성전자, KODEX 200)
samsung_df = fdr.DataReader('005930', '2017-01-01', '2017-01-05')
kodex_df = fdr.DataReader('069500', '2017-01-01', '2017-01-05')

# 1. 수직 연결 (axis=0) - 멀티인덱스 활용
result_v = pd.concat([samsung_df, kodex_df], keys=['삼성', 'KODEX200'])
print(result_v.head(3))
# 출력:
#                        Open   High    Low  Close   Volume  Change
#          Date
# 삼성       2017-01-02  35980  36240  35880  36100    93012   0.002
#          2017-01-03  36280  36620  36020  36480   147153   0.011
#          2017-01-04  36500  36520  36100  36160   159435  -0.009

# 2. 수평 연결 (axis=1)
result_h = pd.concat([samsung_df, kodex_df], keys=['삼성', 'KODEX200'], axis=1)
print(result_h.head(2))
# 출력:
#                 삼성                                     KODEX200
#              Open   High    Low  Close  Volume Change     Open   High    Low   Close   Volume Change
# Date
# 2017-01-02  35980  36240  35880  36100   93012  0.002    21727  21861  21652   21804  2052815  0.003
# 2017-01-03  36280  36620  36020  36480  147153  0.011    21861  21976  21835   21976  4421696  0.008
```

**[코드 상세 설명]**
*   **`axis=0` (수직 연결)**: 데이터를 위아래로 쌓습니다. 기본값이며, 주로 날짜가 다른 데이터를 이어 붙이거나 여러 종목 데이터를 하나로 합칠 때 사용합니다.
*   **`axis=1` (수평 연결)**: 데이터를 좌우로 옆으로 붙입니다. 날짜(Index)가 같은 두 종목의 가격 데이터를 나란히 배치하여 비교 분석할 때 유용합니다.
*   **`keys=['삼성', 'KODEX200']`**: 연결된 데이터에 그룹 이름을 부여하여 **계층적 인덱스(Multi-index)**를 만듭니다. 수직 연결 시에는 행 인덱스에, 수평 연결 시에는 컬럼 인덱스에 이름표가 붙어 데이터를 구분하기 쉬워집니다.
*   **자동 정렬 및 Outer Join**: `concat`은 기본적으로 인덱스나 컬럼명을 기준으로 데이터를 맞춥니다. 만약 한쪽에만 있는 데이터가 있다면, 없는 쪽은 `NaN`(결측치)으로 채워집니다.

---

## 3. join()과 merge(): 관계형 결합

### 3.1. join(): 인덱스 중심 결합
`join()`은 기본적으로 호출하는 DataFrame의 인덱스를 기준으로 다른 DataFrame을 합칩니다.

```python
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'], 'B': ['B0', 'B1', 'B2']}, index=['K0', 'K1', 'K2'])
right = pd.DataFrame({'C': ['C0', 'C2', 'C3'], 'D': ['D0', 'D2', 'D3']}, index=['K0', 'K2', 'K3'])

# Left Join (기본값)
print(left.join(right))
# 출력:
#      A   B    C    D
# K0  A0  B0   C0   D0
# K1  A1  B1  NaN  NaN
# K2  A2  B2   C2   D2

# Outer Join
print(left.join(right, how='outer'))
# 출력:
#       A    B    C    D
# K0   A0   B0   C0   D0
# K1   A1   B1  NaN  NaN
# K2   A2   B2   C2   D2
# K3  NaN  NaN   C3   D3
```

### 3.2. merge(): 컬럼(값) 중심 결합
`merge()`는 인덱스 대신 특정 컬럼의 **값**을 기준으로 결합합니다 (SQL의 JOIN과 유사).

```python
left_m = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'], 'key2': ['K0', 'K1', 'K0', 'K1'], 'A': ['A0', 'A1', 'A2', 'A3']})
right_m = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'], 'key2': ['K0', 'K0', 'K0', 'K0'], 'C': ['C0', 'C1', 'C2', 'C3']})

# 여러 키를 기준으로 Inner Join
print(pd.merge(left_m, right_m, on=['key1', 'key2']))
# 출력:
#    key1 key2   A   C
# 0   K0   K0  A0  C0
# 1   K1   K0  A2  C1
# 2   K1   K0  A2  C2
```

---

## 4. 데이터 재구조화 (Pivot)
Long-form 데이터를 Wide-form으로 변환할 때 `pivot()`을 사용합니다. 퀀트 분석에서 종목별 시계열 가격 테이블을 만들 때 매우 유용합니다.

```python
sample_data = pd.DataFrame({
    "종목명":["삼성", "현대", "하이닉스", "삼성", "현대", "하이닉스"],
    "datetime":["2019-01-01", "2019-01-01", "2019-01-01", "2019-01-02", "2019-01-02", "2019-01-02"],
    "price":[1,2,3, 4,5,6]
})

print(sample_data.pivot(index="datetime", columns="종목명", values="price"))
# 출력:
# 종목명         삼성  하이닉스  현대
# datetime
# 2019-01-01   1     3   2
# 2019-01-02   4     6   5
```

---

## 5. 학습 포인트 (퀀트 투자 관점)

1.  **데이터 통합의 중요성**: 개별 종목 파일로 흩어진 데이터를 `concat`이나 `merge`를 통해 하나의 판으로 모아야 전 종목 백테스팅이 가능해집니다.
2.  **Join 방식의 선택**:
    *   **Inner Join**: 두 데이터셋 모두에 존재하는 날짜만 추출 (데이터 누락 방지).
    *   **Left Join**: 벤치마크 지수(예: KOSPI)를 기준으로 종목 데이터를 붙일 때 유용.
3.  **Cartesian Product 주의**: 키 값이 중복된 데이터끼리 `merge`하면 데이터가 기하급수적으로 늘어날 수 있으므로 전처리 시 중복 제거가 필수입니다.

---

## 6. 주요 라이브러리 함수 정리

| 라이브러리 | 클래스/모듈 | 함수명 | 설명 |
| :--- | :--- | :--- | :--- |
| **Pandas** | `pd` | `concat()` | 여러 Pandas 객체를 축(axis)을 기준으로 연결 |
| **Pandas** | `pd` | `merge()` | 공통 컬럼의 값을 기준으로 두 DataFrame을 결합 (SQL Style) |
| **Pandas** | `DataFrame` | `join()` | 인덱스를 기준으로 다른 DataFrame과 결합 |
| **Pandas** | `DataFrame` | `pivot()` | 인덱스, 컬럼, 값을 지정하여 데이터 구조 변경 (Reshaping) |
| **Pandas** | `DataFrame` | `reset_index()` | 인덱스를 일반 컬럼으로 되돌림 |
| **Pandas** | `DataFrame` | `set_index()` | 특정 컬럼을 인덱스로 설정 |
| **Pandas** | `DataFrame` | `drop()` | 특정 행이나 열을 삭제 (axis 지정 필요) |
| **Pandas** | `DataFrame` | `resample()` | 시계열 데이터의 빈도(Frequency)를 변경하여 집계 |
| **Pandas** | `Series` | `to_frame()` | Series를 DataFrame으로 변환 |
