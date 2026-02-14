"""
[학습용] lec2_4_price_based_indicator_ForOnlineLecture.py
이 스크립트는 실습 내용을 가독성 좋게 정리한 코드입니다.
"""

#!/usr/bin/env python
# coding: utf-8

# **수업을 수강하시기 전, lec2_1.ipynb의 "수강 전 필독"을 반드시 확인해주세요**

# 이동평균(SMA, Simple Moving Average)

# [학습 포인트] $d_{k} = {1 \over k} (p_1 + p_2 + ... + p_k) $

# [학습 포인트] $d_{k+1} = {1 \over k} (p_2 + p_3 + ... + p_{k+1}) $

# [학습 포인트] 주의할점: cheating

import FinanceDataReader as fdr

df = fdr.DataReader("005930", '2014-01-02', '2020-10-30')
df = df[['Close']]
df.columns = ["삼성전자"]
df.head(10)

# `rolling()`

df.rolling(window=5)

df.rolling(window=5).mean().head(10)

df.loc[:, "삼성전자_20_SMA"] = df['삼성전자'].rolling(window=5).mean()

df.head(10)

# ### `min_period` argument

# window = 20이면 맨처음 20개 데이터가 nan이 되는데, min_period를 사용함으로써
# 최소 min_period 갯수 이상이면 window 보다 크기가 작아도 그놈들로 operation 진행
df['삼성전자'].rolling(window=20, min_periods=2).mean().head()

# `expanding()`

# [학습 포인트] window를 1부터 시작해서 +1씩 점점 늘려가면서 roling

df['삼성전자'].expanding(min_periods=1).mean().head()

df['삼성전자'].expanding(min_periods=2).mean().head()

df.loc[:, "삼성전자_60_SMA_min_period"] = df['삼성전자'].rolling(window=60, min_periods=1).mean()
df.loc[:, "삼성전자_expanding"] = df['삼성전자'].expanding(min_periods=1).mean()

df.head()

import matplotlib
import matplotlib.font_manager as fm

matplotlib.rcParams['axes.unicode_minus'] = False

font_location = "/Library/Fonts/NanumBarunGothic.ttf"
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name, size=15)

# 데이터를 시각화합니다.
df.iloc[-100:].plot(figsize=(15, 7))

# 볼린져밴드(Bollinger band)

import FinanceDataReader as fdr

df = fdr.DataReader("005930", '2014-01-02', '2020-10-30')
df = df[['Close']]
df.columns = ["삼성전자"]
df.head(10)

df.loc[:, "삼성전자_60_SMA_min_period"] = df['삼성전자'].rolling(window=60, min_periods=1).mean()

# 데이터를 시각화합니다.
df['삼성전자_60_SMA_min_period'].plot(figsize=(15, 10))

df['Upper'] = df['삼성전자_60_SMA_min_period'] + 2*df['삼성전자'].rolling(window=60).std()
df['Lower'] = df['삼성전자_60_SMA_min_period'] - 2*df['삼성전자'].rolling(window=60).std()

import platform
import matplotlib.font_manager as fm
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False

from pathlib import Path
home = str(Path.home())
font_location = home + "/Library/Fonts/NanumBarunGothic.ttf"
font_name = fm.FontProperties(fname=font_location).get_name()
mpl.rc('font', family=font_name)

# 데이터를 시각화합니다.
df[['삼성전자', '삼성전자_60_SMA_min_period', 'Upper', 'Lower']].iloc[-400:].plot(figsize=(15, 7))

# rolling function 커스터마이징

#  - Mean absolute deviation

def mean_abs_dev(x):
    return np.abs(x - x.mean()).mean()

mean_abs_dev(df['삼성전자'].iloc[:5])

simple_rtn_df = df['삼성전자'].pct_change().fillna(0)
simple_rtn_df.head()

simple_rtn_df.rolling(10).apply(mean_abs_dev)

# Rolling correlation of returns

# ### 다수 종목(3종목 이상)

df1 = fdr.DataReader("005930", '2018-01-02', '2020-10-30')
df2 = fdr.DataReader("148070", '2018-01-02', '2020-10-30')
df3 = fdr.DataReader("035420", '2018-01-02', '2020-10-30')
df = pd.concat([df1[['Close']], df2[['Close']], df3[['Close']]], axis=1)
df.columns = ["삼성전자", "KOSEF 국고채10년", "네이버"]

daily_rtn_df = np.log(df.pct_change() + 1).fillna(0)
daily_rtn_df.head()

daily_rtn_df.corr()

total_corr_df = daily_rtn_df.rolling(window=250).corr().dropna()
total_corr_df

total_corr_df.index.get_level_values(0)
total_corr_df.index.get_level_values(1)

# unstack(level=1): index level=1을 columns로 옮긴다
unstacked_total_corr_df = total_corr_df.unstack(level=1)
unstacked_total_corr_df.head()

# 데이터를 시각화합니다.
unstacked_total_corr_df['삼성전자'].drop("삼성전자", axis=1).plot(figsize=(15, 8))

# ### Just 두 종목

import FinanceDataReader as fdr

df1 = fdr.DataReader("005930", '2018-01-02', '2020-10-30')
df2 = fdr.DataReader("148070", '2018-01-02', '2020-10-30')
df = pd.concat([df1[['Close']], df2[['Close']]], axis=1)
df.columns = ["삼성전자", "KOSEF 국고채10년"]

daily_rtn_df = np.log(df.pct_change() + 1).fillna(0)
daily_rtn_df.head()

corr_pair_df = daily_rtn_df['삼성전자'].rolling(window=60).corr(
    daily_rtn_df['KOSEF 국고채10년']
)
corr_pair_df.head()
corr_pair_df.tail()

# 데이터를 시각화합니다.
ax = df.iloc[60:].plot(figsize=(15, 5));
# 데이터를 시각화합니다.
corr_pair_df.iloc[60:].plot(ax=ax, secondary_y=True)

# Exponentially-weighted moving average

# [학습 포인트] SMA의 단점
#     - Smaller window size --> signal보단 여전히 noise 일 수 있음
#     - lagging (`min_periods`로 어느정도 커버가능)
#     - window내 가장 최근 데이터나 가장 과거데이터나 가중치가 같게 평가됨
#     - 극단적인 값(outlier)의 출현은 SMA를 왜곡시킬 수 있음
#     - trend 정도만 보여줌

# [학습 포인트] EWMA (Exponentially-weighted moving average)
#     - 최근 발생한 값들에 대해 더 가중치를 줌으로써 time에 대한 정보가 값에 반영이 됨

# $ (0 \lt \alpha \le 1) $

# $ EMA_{t} = 1p_t + (1-\alpha)p_{t-1} + (1-\alpha)^2p_{t-2} + ... + (1-\alpha)^tp_0$

# $EMA_{t} = \frac{ 1p_t + (1-\alpha)p_{t-1} + (1-\alpha)^2p_{t-2} + ... + (1-\alpha)^tp_0 }{1 + (1 - \alpha) + (1 - \alpha)^2 + ... + (1 - \alpha)^t}$

# if $ t \rightarrow \infty  $

# $EMA_{t} = \frac{ 1p_t + (1-\alpha)p_{t-1} + (1-\alpha)^2p_{t-2} + ... }{1 + (1 - \alpha) + (1 - \alpha)^2 + ... }$

# Since, $ {{1 + (1 - \alpha) + (1 - \alpha)^2 + ... }} = {1 \over { 1 - (1 - \alpha)}} = {1 \over {\alpha}} $

# $EMA_{t} = \alpha[{ p_t + (1-\alpha)p_{t-1} + (1-\alpha)^2p_{t-2} + (1-\alpha)^3p_{t-3} ... }] $

# $EMA_{t} = \alpha{p_t} + {\alpha} [{   (1-\alpha)p_{t-1} + (1-\alpha)^2p_{t-2} + (1-\alpha)^3p_{t-3} ... ]} $

# $EMA_{t} = \alpha{p_t} + {(1-\alpha)} [{ {\alpha}p_{t-1} + {\alpha}(1-\alpha)p_{t-2}  + {\alpha}(1-\alpha)^2p_{t-3} ... ]    } $

# $EMA_{t} = \alpha{p_t} + {(1-\alpha)} [{ {\alpha}[p_{t-1} + (1-\alpha)p_{t-2}  + (1-\alpha)^2p_{t-3} ... ]    }] $

# $EMA_{t} = \alpha{p_t} + {(1-\alpha)} EMA_{t-1} $

# $ \begin{split}
#     y_0 &= x_0\\
#     y_t &= (1 - \alpha) y_{t-1} + \alpha x_t,
# \end{split} $

df = df[['삼성전자']]

df.head(3)

# 51500 = 0.8 * 51620 + (1 - 0.8) * 51020
# 51164 = 0.8 * 51080 + (1 - 0.8) * 51500
df.ewm(alpha=0.8, adjust=False).mean().head(3)

# if $ t \ne \infty  $

# $        y_t = \frac{x_t + (1 - \alpha)x_{t-1} + (1 - \alpha)^2 x_{t-2} + ... + (1 -
#         \alpha)^t x_0}{1 + (1 - \alpha) + (1 - \alpha)^2 + ... + (1 - \alpha)^t}$

df.head(3)

# 51520 = (51620 + (1 - 0.8) * 51020) / (1 + (1 - 0.8))
# 51165.161 = (51080 + (1 - 0.8) * 51620 + ((1 - 0.8)**2) * 51020 ) / (1 + (1 - 0.8) + (1 - 0.8)**2)
df.ewm(alpha=0.8, adjust=True).mean().head(3)

tmp_df = df[['삼성전자']].copy()
tmp_df['20SMA'] = tmp_df["삼성전자"].rolling(20).mean()
tmp_df['EWMA_02'] = tmp_df['삼성전자'].ewm(alpha=0.2).mean()

# 데이터를 시각화합니다.
tmp_df.iloc[-100:].plot(figsize=(15, 8))
