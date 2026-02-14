# Lec 2-1: 시계열 인덱스 마스터 (Timestamp & DatetimeIndex)

이 장에서는 퀀트 투자 데이터 분석의 핵심인 시계열(Time-series) 데이터를 다루기 위한 Pandas의 강력한 도구들을 학습합니다. 특정 시점을 나타내는 `Timestamp`와 기간을 나타내는 `Period`의 차이를 명확히 이해하고 활용하는 법을 다룹니다.

## 1. 시점 데이터 (Timestamp & DatetimeIndex)

### 1.1. Timestamp와 DatetimeIndex 생성
`Timestamp`는 특정 시점을 나타내는 객체이며, 이를 모아둔 것이 `DatetimeIndex`입니다.

```python
import pandas as pd
import numpy as np
from datetime import datetime

# 1. Timestamp 생성
dt = datetime(2021, 1, 1)
ts = pd.Timestamp(dt)
print(f"Pandas Timestamp: {ts}")
# 출력: Pandas Timestamp: 2021-01-01 00:00:00

# 2. DatetimeIndex 생성 (pd.to_datetime)
dates = [datetime(2014, 8, 1), datetime(2014, 8, 5)]
dti = pd.to_datetime(dates)
print(dti)
# 출력: DatetimeIndex(['2014-08-01', '2014-08-05'], dtype='datetime64[ns]', freq=None)
```

### 1.2. 시계열 인덱싱 (Time-series Indexing)
시계열 인덱스를 가진 데이터는 문자열만으로도 매우 편리하게 인덱싱과 슬라이싱이 가능합니다.

```python
ts_series = pd.Series(np.random.randn(2), index=dti)

# 문자열을 이용한 접근
print(ts_series.loc["2014-08-01"])
# 출력: -0.425 (예시값)

# 월 단위 슬라이싱
# ts_series.loc["2014-08"] 와 같이 월 정보만 입력해도 해당 월의 모든 데이터 추출 가능
```

### 1.3. 규칙적인 날짜 생성 (pd.date_range)
특정 주기(`freq`)를 가진 날짜 범위를 생성할 때 사용합니다.

```python
# 일간 데이터 (Daily)
dr = pd.date_range('2014-08-01', periods=3, freq="D")
print(dr)
# 출력: DatetimeIndex(['2014-08-01', '2014-08-02', '2014-08-03'], dtype='datetime64[ns]', freq='D')

# 평일 데이터 (Business Day) - 주말 제외
dr_biz = pd.date_range('2014-08-01', periods=3, freq="B")
print(dr_biz)
# 출력: DatetimeIndex(['2014-08-01', '2014-08-04', '2014-08-05'], dtype='datetime64[ns]', freq='B')
```

---

## 2. 기간 데이터 (Period & PeriodIndex)

### 2.1. Period의 개념
`Timestamp`가 **점(Point)**이라면, `Period`는 **구간(Interval)**을 의미합니다 (예: 분기, 월 등).

```python
# 분기(Quarter) 단위 기간 생성
period = pd.Period('2014-08', freq='Q')
print(f"Period: {period}, 시작일: {period.start_time}, 종료일: {period.end_time}")
# 출력: Period: 2014Q3, 시작일: 2014-07-01 00:00:00, 종료일: 2014-09-30 23:59:59.999999999

# 기간 연산 (다음 분기)
print(f"다음 분기: {period + 1}")
# 출력: 다음 분기: 2014Q4
```

### 2.2. PeriodIndex 활용
연간/월간 리밸런싱이나 분기별 재무 데이터 처리 시 유용합니다.

```python
pr = pd.period_range('2013-01-01', periods=3, freq='M')
print(pr)
# 출력: PeriodIndex(['2013-01', '2013-02', '2013-03'], dtype='period[M]')
```

---

## 3. 학습 포인트 (퀀트 투자 관점)

1.  **시계열 정렬의 중요성**: 백테스팅 전 반드시 `sort_index()`를 수행해야 합니다. 날짜가 섞여 있으면 슬라이싱 시 오류가 발생하거나 예상치 못한 결과가 나올 수 있습니다.
2.  **데이터 소스별 정밀도**: `FinanceDataReader`나 로컬 CSV 파일 로드 시 날짜 컬럼을 반드시 `pd.to_datetime()`으로 변환하여 `DatetimeIndex`로 설정해야 Pandas의 강력한 시계열 기능을 온전히 활용할 수 있습니다.
3.  **B (Business Day) 빈도 활용**: 주식 시장은 주말에 열리지 않으므로, 주가 데이터를 생성하거나 인덱스를 맞출 때 `freq='B'`를 적절히 활용해야 데이터 불일치를 줄일 수 있습니다.

---

## 4. 주요 라이브러리 함수 정리

| 라이브러리 | 클래스/모듈 | 함수명 | 설명 |
| :--- | :--- | :--- | :--- |
| **Pandas** | `pd` | `Timestamp()` | 특정 날짜/시간 시점 객체 생성 |
| **Pandas** | `pd` | `to_datetime()` | 다양한 형태의 날짜 데이터를 DatetimeIndex로 변환 |
| **Pandas** | `pd` | `date_range()` | 일정한 주기(`freq`)를 가진 날짜 배열(DatetimeIndex) 생성 |
| **Pandas** | `pd` | `Period()` | 특정 기간(Interval) 단위 객체 생성 (Q, M, D 등) |
| **Pandas** | `pd` | `period_range()` | 일정한 주기를 가진 기간 배열(PeriodIndex) 생성 |
| **Pandas** | `Timestamp` | `to_period()` | 시점(Timestamp)을 기간(Period)으로 변환 |
| **Pandas** | `Period` | `to_timestamp()` | 기간(Period)을 시점(Timestamp)으로 변환 |
| **datetime** | `datetime` | `datetime()` | Python 표준 라이브러리의 날짜/시간 객체 생성 |
