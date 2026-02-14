"""
[학습용] lec2_5_signal_based_backtesting_ForOnlineLecture.py
이 스크립트는 실습 내용을 가독성 좋게 정리한 코드입니다.
"""

#!/usr/bin/env python
# coding: utf-8

# **수업을 수강하시기 전, lec2_1.ipynb의 "수강 전 필독"을 반드시 확인해주세요**

def get_returns_df(df, N=1, log=False):
    if log:
        return np.log(df / df.shift(N)).iloc[N-1:].fillna(0)
    else:
        return df.pct_change(N, fill_method=None).iloc[N-1:].fillna(0)

def get_cum_returns_df(return_df, log=False):
    if log:
        return np.exp(return_df.cumsum())
    else:
        return (1 + return_df).cumprod()    # same with (return_df.cumsum() + 1)

def get_CAGR_series(cum_rtn_df, num_day_in_year=250):
    cagr_series = cum_rtn_df.iloc[-1]**(num_day_in_year/(len(cum_rtn_df))) - 1
    return cagr_series

def get_sharpe_ratio(log_rtn_df, yearly_rfr = 0.025):
    excess_rtns = log_rtn_df.mean()*252 - yearly_rfr
    return excess_rtns / (log_rtn_df.std() * np.sqrt(252))

def get_drawdown_infos(cum_returns_df):
    # 1. Drawdown
    cummax_df = cum_returns_df.cummax()
    dd_df = cum_returns_df / cummax_df - 1

    # 2. Maximum drawdown
    mdd_series = dd_df.min()

    # 3. longest_dd_period
    dd_duration_info_list = list()
    max_point_df = dd_df[dd_df == 0]
    for col in max_point_df:
        _df = max_point_df[col]
        _df.loc[dd_df[col].last_valid_index()] = 0
        _df = _df.dropna()

        periods = _df.index[1:] - _df.index[:-1]

        days = periods.days
        max_idx = days.argmax()

        longest_dd_period = days.max()
        dd_mean = int(np.mean(days))
        dd_std = int(np.std(days))

        dd_duration_info_list.append(
            [
                dd_mean,
                dd_std,
                longest_dd_period,
                "{} ~ {}".format(_df.index[:-1][max_idx].date(), _df.index[1:][max_idx].date())
            ]
        )

    dd_duration_info_df = pd.DataFrame(
        dd_duration_info_list,
        index=dd_df.columns,
        columns=['drawdown mean', 'drawdown std', 'longest days', 'longest period']
    )
    return dd_df, mdd_series, dd_duration_info_df

# Moving Avg를 이용한 전략

# [학습 포인트] 'Buy' whenever the shorter SMA start to be above the longer one

# [학습 포인트] 'Sell' whenever the shorter SMA start to be below the longer one

import FinanceDataReader as fdr

df = fdr.DataReader("005930", '2010-01-02', '2020-10-30')
df = df[['Close']]
df.head()

df['Close'].rolling(21*6).mean()

df['SMA_short'] = df['Close'].rolling(21*6).mean()
df['SMA_long'] = df['Close'].rolling(21*12).mean()
df = df.dropna()
df.head()

# 데이터를 시각화합니다.
df.plot(figsize=(10, 5))

# 데이터를 시각화합니다.
df.iloc[-200:].plot(figsize=(10, 5))

# position 구하기

# else에 대한 position

# [학습 포인트] 0으로 설정하면: exit / -1로 설정하면: short
df.loc[:, 'position'] = np.where(df['SMA_short'] >= df['SMA_long'], 1, 0)

# 데이터를 시각화합니다.
df['position'].plot(
    ylim=[-1.1, 1.1], title='Market Positioning', figsize=(10, 5), marker='.', markersize=5, linestyle="none"
)

_df = df.iloc[-400:]
_df.head()

exit_index = _df[(_df['position'] - _df['position'].shift()) == -1].index
long_index = _df[(_df['position'] - _df['position'].shift()) == 1].index
exit_index
long_index

# 데이터를 시각화합니다.
ax = _df.drop(['position'], axis=1).plot(figsize=(10, 5))
# 데이터를 시각화합니다.
_df.loc[exit_index, "SMA_short"].plot(ax=ax, marker="o", markersize=8, linestyle="none", color="red");
# 데이터를 시각화합니다.
_df.loc[long_index, "SMA_short"].plot(ax=ax, marker="o", markersize=8, linestyle="none", color="green");

# return 구하기

df.loc[:, 'rtn'] = get_returns_df(df['Close'], log=True)

# position에 대해 shift(1)을 해줘야되는 이유

# [학습 포인트] 해당 일에 position은 당일 종가를 받아서 이미 moving avg등 모든 계산이 끝난 후에 결정되는 position임

# [학습 포인트] 따라 오늘 position을 1로 설정했으면 => (다음날 얻는 수익 * 1) 만큼 먹게됨
df.loc[:, 'strategy_rtn'] = (df['position'].shift(1) * df['rtn']).fillna(0)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10));
# 데이터를 시각화합니다.
df['position'].loc["2012-01"].plot(
    ax=axes[0],
    ylim=[-1.1, 1.1], title='Market Positioning', figsize=(10, 5), marker='.', markersize=5, linestyle="none"
);
# 데이터를 시각화합니다.
df['position'].loc["2013-09"].plot(
    ax=axes[1],
    ylim=[-1.1, 1.1], title='Market Positioning', figsize=(10, 5), marker='.', markersize=5, linestyle="none"
);

df.loc[:, 'cum_rtn'] = get_cum_returns_df(df['rtn'], log=True)
df.loc[:, 'cum_strategy_rtn'] = get_cum_returns_df(df['strategy_rtn'], log=True)

exit_index = df[(df['position'] - df['position'].shift()) == -1].index
long_index = df[(df['position'] - df['position'].shift()) == 1].index

# 데이터를 시각화합니다.
ax = df[['cum_rtn', 'cum_strategy_rtn']].plot(figsize=(10, 5));
# 데이터를 시각화합니다.
df.loc[exit_index, "cum_rtn"].plot(ax=ax, marker="o", markersize=8, linestyle="none", color="red");
# 데이터를 시각화합니다.
df.loc[long_index, "cum_rtn"].plot(ax=ax, marker="o", markersize=8, linestyle="none", color="orange");

# performance 구하기

get_sharpe_ratio(df[['rtn', 'strategy_rtn']]).to_frame("Sharpe Ratio")

dd_df, mdd_series, longest_dd_period_df = get_drawdown_infos(df.filter(like="cum_"))
# 데이터를 시각화합니다.
dd_df.plot(figsize=(10, 5))

mdd_series.to_frame("MDD")

longest_dd_period_df

get_CAGR_series(df.filter(like="cum_")).to_frame("CAGR")

# Momentum을 이용한 전략

# [학습 포인트] 2 types
#     1. Cross-sectional momentum (Relative momentum)
#     2. Time-series momentum

import FinanceDataReader as fdr

df = fdr.DataReader("005930", '2010-01-02', '2020-10-30')
df = df[['Close']]
df.head()

# position 구하기

momentum_window = 20 * 6

# -1 or 1 or 0
position = np.sign(get_returns_df(df['Close'], N=momentum_window))
position.head()   # 20 X 6만큼 날라감

position.value_counts()

position.loc[position == -1] = 0  # short는 안 한다고 가정
# 데이터를 시각화합니다.
position.plot(
    ylim=[-1.1, 1.1], title='Market Positioning', figsize=(10, 5), marker='.', markersize=5, linestyle="none"
)

df.shape
position.shape

df.loc[:, 'position'] = position

df = df.dropna()
df.head()

_df = df.iloc[-200:]

exit_index = _df[(_df['position'] - _df['position'].shift()) == -1].index
long_index = _df[(_df['position'] - _df['position'].shift()) == 1].index

# 데이터를 시각화합니다.
ax = _df.drop(['position'], axis=1).plot(figsize=(10, 5));
# 데이터를 시각화합니다.
_df.loc[exit_index, "Close"].plot(ax=ax, marker="o", markersize=8, linestyle="none", color="red");
# 데이터를 시각화합니다.
_df.loc[long_index, "Close"].plot(ax=ax, marker="o", markersize=8, linestyle="none", color="green");

ax.axvline(df.index[df.index.get_loc("2020-09-02") - 120]);

long_index

df.index.get_loc("2020-09-02") - 120

df.index[df.index.get_loc("2020-09-02") - 120]

# return 구하기

df.loc[:, 'rtn'] = get_returns_df(df['Close'], log=True)
df.loc[:, 'strategy_rtn'] = (df['position'].shift(1) * df['rtn']).fillna(0)

df.loc[:, 'cum_rtn'] = get_cum_returns_df(df['rtn'], log=True)
df.loc[:, 'cum_strategy_rtn'] = get_cum_returns_df(df['strategy_rtn'], log=True)

exit_index = df[(df['position'] - df['position'].shift()) == -1].index
long_index = df[(df['position'] - df['position'].shift()) == 1].index

# 데이터를 시각화합니다.
ax = df[['cum_rtn', 'cum_strategy_rtn']].plot(figsize=(10, 5));
# 데이터를 시각화합니다.
df.loc[exit_index, "cum_rtn"].plot(ax=ax, marker="o", markersize=3, linestyle="none", color="red");
# 데이터를 시각화합니다.
df.loc[long_index, "cum_rtn"].plot(ax=ax, marker="o", markersize=3, linestyle="none", color="green");

_df = df.iloc[-150:]

exit_index = _df[(_df['position'] - _df['position'].shift()) == -1].index
long_index = _df[(_df['position'] - _df['position'].shift()) == 1].index

# 데이터를 시각화합니다.
ax = _df[['cum_rtn', 'cum_strategy_rtn']].plot(figsize=(10, 5));
# 데이터를 시각화합니다.
_df.loc[exit_index, "cum_rtn"].plot(ax=ax, marker="o", markersize=8, linestyle="none", color="red");
# 데이터를 시각화합니다.
_df.loc[long_index, "cum_rtn"].plot(ax=ax, marker="o", markersize=8, linestyle="none", color="green");

# [학습 포인트] Multiple momentum window test

import FinanceDataReader as fdr

df = fdr.DataReader("005930", '2010-01-02', '2020-10-30')
df = df[['Close']]
df.loc[:, 'rtn'] = get_returns_df(df['Close'], log=True)

rtn_column_list = ["rtn"]
momentum_list = [1*20, 3*20, 5*20]
for momentum_window in momentum_list:
    position = np.sign(get_returns_df(df['Close'], N=momentum_window))
    position.loc[position == -1] = 0

    df.loc[:, 'position_{}'.format(momentum_window)] = position
    df = df.dropna()
    df.loc[:, 'strategy_{}_rtn'.format(momentum_window)] = (
        df['position_{}'.format(momentum_window)].shift(1) * df['rtn']
    ).fillna(0)
    rtn_column_list.append('strategy_{}_rtn'.format(momentum_window))

cum_rtn_df = get_cum_returns_df(df[rtn_column_list], log=True)
# 데이터를 시각화합니다.
cum_rtn_df.plot(figsize=(10, 5))

# performance 구하기

get_sharpe_ratio(df[rtn_column_list]).to_frame("Sharpe Ratio")

dd_df, mdd_series, longest_dd_period_df = get_drawdown_infos(cum_rtn_df)
# 데이터를 시각화합니다.
dd_df.plot(figsize=(10, 5))

mdd_series.to_frame("MDD")

longest_dd_period_df

get_CAGR_series(cum_rtn_df).to_frame("CAGR")

# [학습 포인트] 결국은 'position' series를 어떻게 만드느냐가 중요

position = np.sign(get_returns_df(df['Close'], N=momentum_window) - 0.1)
position

# Mean-reversion을 이용한 전략

import FinanceDataReader as fdr

df = fdr.DataReader("005930", '2010-01-02', '2020-10-30')
df = df[['Close']]
df.head()

window = 60
df.loc[:, 'SMA'] = df['Close'].rolling(window).mean()

rolling_std = df['Close'].rolling(window).std()

df = df.dropna()

df['+threshold'] = df['SMA'] + 1.5*rolling_std
df['-threshold'] = df['SMA'] - 1.5*rolling_std

# 데이터를 시각화합니다.
df[['Close', 'SMA', '+threshold', '-threshold']].plot(figsize=(10, 5))

# 데이터를 시각화합니다.
df[['Close', 'SMA', '+threshold', '-threshold']].iloc[-200:].plot(figsize=(10, 5))

# position 구하기

pos1 = np.where(df['Close'] <= df['-threshold'], 1, 0)
pos2 = np.where(df['Close'] >= df['+threshold'], -1, 0)
df.loc[:, 'position'] = pos1 + pos2

# 데이터를 시각화합니다.
df['position'].plot(
    ylim=[-1.1, 1.1], title='Market Positioning', figsize=(10, 5), marker='.', markersize=5, linestyle="none"
);

_df = df.iloc[-500:]

short_index = _df[
    ((_df['position'] - _df['position'].shift()) == -1) & (_df['position'] == -1)
].index
long_index = _df[
    ((_df['position'] - _df['position'].shift()) == 1) & (_df['position'] == 1)
].index

# 데이터를 시각화합니다.
ax = _df[['Close', 'SMA', '+threshold', '-threshold']].plot(figsize=(10, 5));
# 데이터를 시각화합니다.
_df.loc[short_index, "Close"].plot(ax=ax, marker="o", markersize=5, linestyle="none", color="red");
# 데이터를 시각화합니다.
_df.loc[long_index, "Close"].plot(ax=ax, marker="o", markersize=5, linestyle="none", color="green");

# return 구하기

df['rtn'] = get_returns_df(df['Close'], log=True)
df['strategy_rtn'] = (df['position'].shift(1) * df['rtn']).fillna(0)

df.loc[:, 'cum_rtn'] = get_cum_returns_df(df['rtn'], log=True)
df.loc[:, 'cum_strategy_rtn'] = get_cum_returns_df(df['strategy_rtn'], log=True)

short_index = df[
    ((df['position'] - df['position'].shift()) == -1) & (df['position'] == -1)
].index
long_index = df[
    ((df['position'] - df['position'].shift()) == 1) & (df['position'] == 1)
].index

# 데이터를 시각화합니다.
ax = df[['cum_rtn', 'cum_strategy_rtn']].plot(figsize=(10, 5));
# 데이터를 시각화합니다.
df.loc[short_index, "cum_rtn"].plot(ax=ax, marker="o", markersize=3, linestyle="none", color="red");
# 데이터를 시각화합니다.
df.loc[long_index, "cum_rtn"].plot(ax=ax, marker="o", markersize=3, linestyle="none", color="green");

_df = df.iloc[:200]

short_index = _df[
    ((_df['position'] - _df['position'].shift()) == -1) & (_df['position'] == -1)
].index
long_index = _df[
    ((_df['position'] - _df['position'].shift()) == 1) & (_df['position'] == 1)
].index

short_exit_index = _df[
    ((_df['position'] - _df['position'].shift()) == 1) & (_df['position'] == 0)
].index
long_exit_index = _df[
    ((_df['position'] - _df['position'].shift()) == -1) & (_df['position'] == 0)
].index

# 데이터를 시각화합니다.
ax = _df[['cum_rtn', 'cum_strategy_rtn']].plot(figsize=(10, 5));

# 데이터를 시각화합니다.
_df.loc[short_index, "cum_rtn"].plot(ax=ax, marker="o", markersize=5, linestyle="none", color="red");
# 데이터를 시각화합니다.
_df.loc[short_exit_index, "cum_rtn"].plot(ax=ax, marker="x", markersize=5, linestyle="none", color="red");

# 데이터를 시각화합니다.
_df.loc[long_index, "cum_rtn"].plot(ax=ax, marker="o", markersize=5, linestyle="none", color="green");
# 데이터를 시각화합니다.
_df.loc[long_exit_index, "cum_rtn"].plot(ax=ax, marker="x", markersize=5, linestyle="none", color="green");

# performance 구하기

get_sharpe_ratio(df[['rtn', 'strategy_rtn']]).to_frame("Sharpe Ratio")

dd_df, mdd_series, longest_dd_period_df = get_drawdown_infos(df.filter(like="cum_"))
# 데이터를 시각화합니다.
dd_df.plot(figsize=(10, 5))

mdd_series.to_frame("MDD")

longest_dd_period_df

get_CAGR_series(df.filter(like="cum_")).to_frame("CAGR")

# 변동성 돌파전략

# [학습 포인트] 참고 링크: https://ldgeao99.tistory.com/441

import FinanceDataReader as fdr

df = fdr.DataReader("005930", '2010-01-02', '2020-10-30')
df = df[['Open', 'High', 'Close', 'Low']]
df.head()

# 데이터를 시각화합니다.
df['Open'].plot()

(df == 0).sum()

df[df['Open'] == 0]

df = df[df['Open'] != 0]

(df == 0).sum()

# poisition 구하기

df['range'] = df['High'] - df['Low']

df['threshold'] = df['Open'] + df['range'].shift() * 0.6

cond = df['threshold'] <= df['High']
df['position'] = cond.astype(int)

df.head()

# return 구하기

p_current = df['Open']
p_prev = df['threshold'].shift()
df['rtn'] = p_current / p_prev - 1

df.head()

df = df.dropna()

df.head(8)

df['strategy_rtn'] = df['position'].shift() * df['rtn']
df['strategy_rtn'] = df['strategy_rtn'].fillna(0)

df['cum_rtn'] = df['Close'] / df['Close'].iloc[0]
df['strategy_cum_rtn'] = (df['strategy_rtn'] + 1).cumprod()

# 데이터를 시각화합니다.
df.filter(like="cum").plot(figsize=(10, 5))

# performance 구하기

cum_rtn_df = df.filter(like="cum")

rtn_df = get_returns_df(cum_rtn_df, log=True)

get_sharpe_ratio(rtn_df).to_frame("Sharpe Ratio")

dd_df, mdd_series, longest_dd_period_df = get_drawdown_infos(cum_rtn_df)
# 데이터를 시각화합니다.
dd_df.plot(figsize=(10, 5))

mdd_series.to_frame("MDD")

longest_dd_period_df

get_CAGR_series(cum_rtn_df).to_frame("CAGR")

# 주의사항

# [학습 포인트] 거래비용

# [학습 포인트] 정확히 threshold에 진입하지 못하는 경우

# [학습 포인트] 내가 실제 개입함으로써 지정가에 체결이 발생하지 않는 경우

# [학습 포인트] 전략이 많이 알려지면?

# [학습 포인트] 정확히 시초가 매도가 가능할까?

# [학습 포인트] 거래량이 작아 층이 얇은 경우

# [학습 포인트] 변동성이 너무 커서 open + k*threshold가 생각보다 너무 빨리 도달하는 경우(Covid-19)

# [학습 포인트] position이 연일에 걸쳐 1이 두번 나오는 경우

# [학습 포인트] 하루에 threshold를 2번이상 찍는 경우?
