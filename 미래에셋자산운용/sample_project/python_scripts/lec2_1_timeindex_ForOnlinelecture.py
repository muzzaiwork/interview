"""
[학습용] lec2_1_timeindex_ForOnlinelecture.py
이 스크립트는 실습 내용을 가독성 좋게 정리한 코드입니다.
"""

#!/usr/bin/env python
# coding: utf-8

# [중요] 수강 전 필독

# 1. 아래 내용들은 수업 전 반드시 참고하시길 바랍니다.
#     - 강의 상세페이지
#     - '오리엔테이션' 시청
#     - jupyter 파일 내 목차가 보이도록 하는 extension 설치하는 법: (https://www.inflearn.com/course/생초보-입문-파이썬/lecture/73183)
#     - 수업자료에 있는 `requirements.txt`에 대해 `pip install -r requirements.txt`를 통해 설치해주세요
#         - 잘모르시겠다면 https://www.inflearn.com/course/파이썬-웹크롤링-업무자동화/lecture/73447 를 참고해주세요
# 2. 본 수업과 관련이 있는 다른 수업들
#     1. **[선행수업]** 문과생도, 비전공자도, 누구나 배울 수 있는 파이썬 (무료): https://www.inflearn.com/course/생초보-입문-파이썬?inst=659a82bb
#     2. **[선행수업]** 파이썬(Python)으로 데이터 기반 주식 퀀트 투자하기 part1: https://www.inflearn.com/course/파이썬-판다스-퀀트-투자?inst=b50adcaa
#     3. [선택수업] 내 업무를 대신 할 파이썬(Python) 웹크롤링 & 자동화: https://www.inflearn.com/course/파이썬-웹크롤링-업무자동화?inst=631f2f8e
# 3. **[중요]** 데이터 관련
#     - 본 수업에서 제공하는 price 데이터(`us_etf_1.csv`, `us_etf2.csv`)는 '수정종가(adjusted close)이기 때문에 1) 언제, 2) 어느 데이터 소스에서 데이터를 가져오느냐에 따라, 수업 영상에서 보이는 값과 서로 상이할 수 있습니다. 또한 `FinanceDataReader` 역시 언제 해당 api를 호출하느냐에따라 값이 다르게 나올 수 있습니다(추가 '수정(adjustment)' 사항이 발생했을 수도 있기 때문)
#     - 데이터는 크롤링 등의 방법(e.g. 내 업무를 대신 할 파이썬(Python) 웹크롤링 & 자동화 에서 배운 내용) 등을 사용하여 수강생분들의 재량에 따라 자유롭게 대체 가능합니다. 본 수업에서는 시간관계상, 그리고 수업 진행의 일관성을 위해 csv 데이터 제공 혹은 `FinanceDataReader`을 사용했으니 참고바랍니다.
# 4. **[중요]** 모든 수업 영상 및 모든 jupyter파일에서는 아래 cell에 있는 코드(import 관련 코드)는 항상 실행이 되었다고 가정합니다.

# 앞으로 수업에서 실행이 되었다고 가정하는 코드

# 그래프 시각화 시, jupyter에서 바로 표현되게 만드는 magic command
# get_ipython().run_line_magic('matplotlib', 'inline')

# jupyter의 code cell에서 print() 함수 호출 없이, (복수의)변수명만 입력하면 해당 변수들의 값들이 한번에 출력이 되도록 해주는 설정입니다.
# 구글에 "multiple output in jupyter withtout print()" 로 검색하면 나오는 코드 중에 하나이니, 딱히 외우거나 하시지 않아도 됩니다.
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Pandas 라이브러리를 pd라는 별칭으로 임포트합니다.
import pandas as pd
# NumPy 라이브러리를 np라는 별칭으로 임포트합니다.
import numpy as np
pd.set_option('display.float_format', lambda x: '%.3f' % x)    # DataFrame 숫자데이터가 소수점 셋째자리까지 나오도록 설정
pd.set_option('display.max_columns', None)  # DataFrame에서 모든 컬럼을 다 볼 수 있게 설정

from datetime import datetime

# `Timestamp` & `DatetimeIndex`

# [학습 포인트] Python's `datetime` vs Pandas's `TimeStamp`
#     - `TimeStamp`는 Numpy의 datetime64 based
#     - `TimeStamp` 클래스의 object가 Python의 `datetime` object보다 더 높은 정밀도를 갖는다

# [학습 포인트] `DatetimeIndex`: 다수의 `TimeStamp` object를 하나의 변수로 관리해주는 클래스
#     - Python `list`에 `TimeStamp` object를 저장하는 것보다 더 최적화가 잘되어있음

# `TimeStamp` object

from datetime import datetime

datetime(2021, 1, 1)
type(datetime(2021, 1, 1))

a = datetime(2014, 8, 1)
b = pd.Timestamp(a)
b

pd.Timestamp("2021-01-02")

# `DatetimeIndex`  object

dates = [datetime(2014, 8, 1), datetime(2014, 8, 5)]
type(dates)

# ### `DatetimeIndex()`

dti = pd.DatetimeIndex(dates)
dti

# ### `pd.to_datetime()`

# 2nd method
pd.to_datetime(dates)

pd.to_datetime(dates)[0]

type(pd.to_datetime(dates)[0])

# ### `Series`'s index

# NumPy 라이브러리를 np라는 별칭으로 임포트합니다.
import numpy as np

dates = [datetime(2014, 8, 1), datetime(2014, 8, 5)]
ts = pd.Series(np.random.randn(2), index=dates)
ts

ts.index

type(ts.index)

# 인덱싱

ts

ts.loc[pd.Timestamp("2014-08-01")]

ts.loc[datetime(2014, 8, 1)]
ts.loc["2014-08-01"]

# True
pd.Timestamp(dates[0])  == datetime(2014, 8, 1)
pd.to_datetime(dates)[0] == datetime(2014, 8, 1)

# False
pd.to_datetime(dates)[0] == "2014-08-01"

ts = ts.sort_index()
ts

ts.loc["2014-08-01"]

ts.loc["2014-08"]

ts.loc["2014-08-01":]

# 주의! list 인덱싱과는 다르게 양끝 포함
ts.loc["2014-08-01":"2014-08-05"]

# `pd.date_range()`

dates = pd.date_range('2014-08-01', periods=10, freq="D")
dates

dates = pd.date_range('2014-08-01', periods=10, freq="B")
dates

dates = pd.date_range('2014-08-01', "2014-08-14", freq="D")
dates

# `Period` & `PeriodIndex`

# [학습 포인트] `Period`= interval of datetime

# [학습 포인트] `PeriodIndex` = 다수의 `Period` object를 하나의 변수로 관리해주는 클래스
#     - `DateTimeIndex`= sequence of `Timestamp`

# `Period` object

period = pd.Period('2014-08', freq='Q')  # freq= "D", "M", .. etc
period

period.start_time
period.end_time

# +1 ==> `freq`에 해당하는 단위가 더해짐 (여기서는 1Q)
period2 = period + 1
period2

period2.start_time
period2.end_time

# `PeriodIndex` object

p2013 = pd.period_range('2013-01-01', '2013-12-31', freq='M')
p2013

p2013[0]

for p in p2013:
    print("{0} {1} {2} {3}".format(p, p.freq, p.start_time, p.end_time))

# ### `DatetimeIndex`와 차이

# DateTimeIndex : collections of `Timestamp` objects
a = pd.date_range('1/1/2013', '12/31/2013', freq='M')
a
a[0]

# PeriodIndex : collections of `Period` objects
b = pd.period_range('1/1/2013', '12/31/2013', freq='M')
b
b[0]

# ### As a `Series`'s index

# 해당 index는 이제 특정 date(time) 시점을 의미하는 것이 아닌, date 범위(range)를 의미
ps = pd.Series(np.random.randn(12), p2013)
ps

ps.loc["2013-11"]

ps.loc["2013-11":]
