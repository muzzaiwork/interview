# Lec 2-6: 빈도 변환 및 리샘플링 (Frequency Conversion)

이 장에서는 시계열 데이터의 시간 단위를 변경하는 **리샘플링(Resampling)** 기법을 학습합니다. 일간 데이터를 월간/연간 데이터로 통합하거나, 낮은 빈도의 데이터를 높은 빈도로 확장하고 결측치를 채우는 방법을 다룹니다.

## 1. 리샘플링 기초: resample() vs asfreq()

### 1.1. asfreq()
특정 빈도의 '정확한 시점'에 위치한 데이터를 추출합니다. 해당 시점에 데이터가 없으면 `NaN`을 반환합니다.

```python
import pandas as pd
import FinanceDataReader as fdr

df = fdr.DataReader("005930", '2020-01-01', '2020-03-31')[['Close']]

# 매 월말(Month End) 시점의 데이터 추출
# 주의: 최신 Pandas 버전에서는 'M' 대신 'ME' 사용 권장
print(df.asfreq("ME"))
# 출력:
#                 Close
# Date
# 2020-01-31 56400.0000
# 2020-02-29        NaN  (29일이 휴장일인 경우)
# 2020-03-31 47750.0000
```

### 1.2. resample()
시간 단위로 데이터를 그룹화하여 집계(Aggregation)합니다. SQL의 `GROUP BY`와 유사하게 동작합니다.

```python
# 월 단위로 그룹화하여 해당 월의 마지막 영업일 데이터 선택
print(df['Close'].resample("ME").last())
# 출력:
# Date
# 2020-01-31    56400
# 2020-02-29    54200 (실제 데이터상 2월의 마지막 영업일 값)
# 2020-03-31    47750
```

---

## 2. 업샘플링과 보간법 (Upsampling & Interpolation)

낮은 빈도의 데이터를 높은 빈도(예: 일간 -> 시간)로 늘릴 때 발생하는 빈 공간을 채우는 기법입니다.

```python
# 일간 데이터를 시간(h) 단위로 확장
upsampled = df.iloc[:2].resample("h").mean()

# 선형 보간법(Linear Interpolation)으로 빈 값 채우기
interpolated = upsampled.interpolate()

print(interpolated.head())
# 출력:
#                           Close
# Date
# 2020-01-02 00:00:00 55200.0000
# 2020-01-02 01:00:00 55212.5000
# 2020-01-02 02:00:00 55225.0000
```

---

## 3. 실전 활용: 월별 수익률 및 OHLC 생성

### 3.1. 월별 수익률 계산
일간 수익률 데이터를 월 단위로 합산하여 월별 성과를 산출합니다.

```python
import numpy as np

log_rtn = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)

# 월별 로그 수익률 합산 후 단순 수익률로 변환
monthly_rtn = np.exp(log_rtn.resample("ME").sum()) - 1
print(monthly_rtn)
# 출력:
# Date
# 2020-01-31    0.0217
# 2020-02-29   -0.0390
```

### 3.2. OHLC 리샘플링
일간 종가 데이터를 주간 단위의 시가, 고가, 저가, 종가로 변환합니다.

```python
# 주간(W) 단위 OHLC 생성
weekly_ohlc = df['Close'].resample("W").ohlc()
print(weekly_ohlc.head())
# 출력:
#               open   high    low  close
# Date
# 2020-01-05  55200  55500  55200  55500
# 2020-01-12  55500  59500  55500  59500
```

---

## 4. 학습 포인트 (퀀트 투자 관점)

1.  **데이터 정합성**: `asfreq()`는 휴장일 데이터를 `NaN`으로 처리하므로 데이터 유실에 주의해야 합니다. 반면 `resample().last()`는 실제 존재하는 마지막 데이터를 가져오므로 백테스팅 시 더 안전합니다.
2.  **리밸런싱 주기 설정**: 월간 또는 분기별 리밸런싱 전략을 테스트할 때 `resample`을 통해 가격 데이터를 해당 주기에 맞춰 변환하는 과정이 선행되어야 합니다.
3.  **데이터 편향 방지**: 보간법(`interpolate`) 사용 시 미래 데이터를 참조하여 과거 값을 채우는 'Look-ahead Bias'가 발생하지 않도록 주의해야 합니다.

---

## 5. 주요 라이브러리 함수 정리

| 라이브러리 | 클래스/모듈 | 함수명 | 설명 |
| :--- | :--- | :--- | :--- |
| **Pandas** | `DataFrame` | `resample()` | 시간 단위로 데이터를 그룹화 (Time-based GroupBy) |
| **Pandas** | `DataFrame` | `asfreq()` | 특정 빈도의 시점에 해당하는 데이터를 선택 |
| **Pandas** | `Resampler` | `last() / first()` | 그룹 내 마지막 / 첫 번째 데이터 선택 |
| **Pandas** | `Resampler` | `ohlc()` | 그룹 내 시가, 고가, 저가, 종가 산출 |
| **Pandas** | `DataFrame` | `interpolate()` | 결측치를 주변 값과의 관계(선형 등)를 통해 채움 |
| **Pandas** | `DataFrame` | `drop_duplicates()` | 중복된 행 제거 (특정 주기 마지막 데이터 추출 시 활용 가능) |
