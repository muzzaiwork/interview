"""
[학습용] lec2_6_freq_conversion_ForOnlineLecture.py
이 스크립트는 실습 내용을 가독성 좋게 정리한 코드입니다.
"""

#!/usr/bin/env python
# coding: utf-8

# **수업을 수강하시기 전, lec2_1.ipynb의 "수강 전 필독"을 반드시 확인해주세요**

import FinanceDataReader as fdr

df = fdr.DataReader("005930", '2009-09-01', '2010-12-31').drop("Change", axis=1)
df.tail()

df.index

# [학습 포인트] 참고: index는 DateTimeIndex이어야함 & sorting 되어있어야함

df.index = pd.to_datetime(df.index)
df = df.sort_index()

sample = df.iloc[:3]
sample

# `asfreq()`

# [학습 포인트] Sampling하는 index에 mapping되어 있는 value는 그대로 유지

sample.asfreq("H").head()

sample.asfreq("H")['Close'].head()

sample.mean()

# DATAFRAME.mean()
sample.asfreq("H").mean()

sample.asfreq("H", method="ffill").head()

sample.asfreq("M")

df.loc["2009-10-27":]

df.asfreq("M")

# `resample()`

# [학습 포인트] **(date)time-based groupby**

# [학습 포인트] aggregation하는 funciton과 같이 쓰인다

close_df = df[['Close']]
close_df.head()

log_rtn_df = np.log(close_df / close_df.shift(1)).fillna(0)
log_rtn_df.head()

log_rtn_df.resample("M")

log_rtn_df.head()
log_rtn_df.tail()

# resample의 단점: '데이터 내'의 년 말의 값을 대표값으로 설정 (대신 날짜는 12-31로 표기 )
log_rtn_df.resample("A").last()

log_rtn_df.loc["2010-12-30"]
# log_rtn_df.loc["2010-12-31"]  # 존재하지 않음

df.resample("H").mean()

df.resample("H").agg({"Close": "mean"})
df.resample("H")["Close"].mean()

# Upsampling & Downsampling

by_hour = sample.resample("H").mean()
by_hour.head()

# downsampling
sample.resample("A").mean()

sample.mean()

# ### Filling `NaN`
# 1. `fillna` : bfill, ffill
# 2. `interpolate()` : linear interploation

by_hour.fillna(method="ffill").head()

by_hour.interpolate().head()

# = (df - df.shift(1))
by_hour.interpolate().diff()

# `kind` arg

print_header("df.head()"); print(df.head())

df.resample("M", kind="period").mean()
df.resample("M", kind="timestamp").mean()

df.resample("M", kind="period").mean().index
df.resample("M", kind="timestamp").mean().index

# OHLC

print_header("df.head()"); print(df.head())

# df.resample("W").ohlc().head()
df['Close'].resample("M").ohlc().head()

# `asfreq()`와 차이

print_header("df.head()"); print(df.head())

df.asfreq("M")   # 얘는 DataFrame을 뱉어냄
df.resample("M")

# 얘는 딱 월 말마다 데이터가 있어야함 (현재 index값 유지, 데이터가 없으면 NaN...)
df.asfreq("M").head(5)

# 월 말에 데이터가 없으면 해당 월의 가장 마지막 날 데이터를 끌어와서 월의 가장마지막 날 데이터로 설정함
df.resample("M").last().head(5)

# ### 비교 문제

# [학습 포인트] `resample("M").mean()` : Month별 mean()

# [학습 포인트] `asfreq("M").mean()` : Month별 마지막 날의 데이터를 가져오고, 그 값들 전체에 대한 mean()

df.asfreq("M").head(10)
# vs
df.asfreq("M").mean().head()

df.resample("M").head(10)
# vs
df.resample("M").mean().head(10)

# 예제: 월별 수익률 구하기

# [학습 포인트] 월마다 발생되는 총수익을 월별로 산출

pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Using `resample()`

log_rtn_df.head()

np.exp(log_rtn_df.resample("M").sum()) - 1

# Monthly 수익률 구하기
month_cum_rtn_df = (np.exp(log_rtn_df.resample("M").sum())-1)

month_cum_rtn_df.head()
month_cum_rtn_df.loc['2010-01-31']

print_header("df.head()"); print(df.head())

(df.loc["2009-10-30", 'Close'] / df.loc["2009-09-30", 'Close']) - 1

df.loc["2009-09-28":].head(3)
df.loc["2009-10-28":].head(3)

# Using `drop_duplicates()`

# [학습 포인트] 정확하게 '월말 데이터'를 내가 가지고 있는 데이터의 '월 말 날짜'에 가져오기

print_header("df.head()"); print(df.head())

df.index.year
df.index.month

df['year'] = df.index.year
df['month'] = df.index.month
print_header("df.head()"); print(df.head())

monthly_df = df.drop_duplicates(subset=['year', 'month'], keep='last')
monthly_df

monthly_df = monthly_df.drop(['year', 'month'], axis=1)
monthly_df.head()

# 첫 달은 0으로 나오는 한계;
monthly_df[['Close']].pct_change().fillna(0).head()
