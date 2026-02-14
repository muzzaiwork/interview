"""
[학습용] lec2_7_rebalancing_based_backtesting_ForOnlineLecture.py
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

# 기본 개념 & 컨셉

# Reminder

# [학습 포인트] cumulative return을 구하는 방법

# Weight

# [학습 포인트] 각 자산 비중(ratio)

# [학습 포인트] 특정 시점에서, 항상 sum이 1이어야함(진입기준)

# [학습 포인트] e.g. Equal weight rebalancing

# Portfolio Return

# [학습 포인트] $\text{Return of portfolio on day1} = w_ar_a + w_br_b + w_cr_c + ...$
#     - 1일차에서의 $\sum{(각 자산의 비중 * 각 자산의 return)}$

# 중요한 포인트

# [학습 포인트] A 주가 : 10 -> 12 -> 6 (daily return: +0.2, -0.5)

# [학습 포인트] B 주가 : 10 -> 5 -> 6 (daily return: -0.5, +0.2)

# [학습 포인트] 각 10주씩 매수를 하고 **buy & hold**

# [학습 포인트] 위의 1.3의 방식으로 수익률 구해보기
#     - 첫째날: -0.15 (by 0.5 \* 0.2 + 0.5 \*-0.5)
#     - 둘째날: -0.15 (by 0.5 \* -0.5 + 0.5 \* 0.2)
#     - 따라서: 200 -> 200\*(1-0.15) -> 200\*(1-0.15)^2
#         - 200 -> 170 -> 144.49999999999997

# [학습 포인트] 포트폴리오 전체 value관점에서 계산해보면: 200 -> 170 -> 120
#     - 수익률은: 0 -> -0.15 -> -0.294

# ### 60:40 or 올웨더 전략 등 비중 기반 전략 진입시, 비중만 유지하면 언제든 진입해도 된다. O/X ?

# Buy & hold

# 구현 방법 1

# ### data 준비

df = pd.DataFrame(
    {
        "A": [10, 15, 12, 13, 10, 11, 12],
        "B": [10, 10, 8, 13, 12, 12, 12],
        "C": [10, 12, 14, 16, 14, 14, 16],
    },
    index=pd.to_datetime(["2018-01-31", "2018-02-10", "2018-02-20", "2018-02-28", "2018-03-20", "2018-03-29", "2018-04-30",])
)
df

# ### shifted return 구하기

# log return을 쓰면 안됨 --> 일마다 종목끼리 sum aggregation을 할 것이므로
rtn_df = get_returns_df(df, log=False)
rtn_df.head()

# => shift를 해줘야, 해당 date에서 가지고 있을 때 발생하는 수익률을
# 그 date에 mapping 가능
shifted_rtn_df = rtn_df.shift(-1)
shifted_rtn_df

shifted_rtn_df = shifted_rtn_df.fillna(0)
shifted_rtn_df

# ### asset flow 구하기

cum_rtn_df = df / df.iloc[0]
cum_rtn_df

asset_flow_df = cum_rtn_df * [0.3, 0.5, 0.2]
asset_flow_df

asset_flow_df = cum_rtn_df * [3000000, 5000000, 2000000]
asset_flow_df

# ### weight df 구하기

# 항상 일별로 sum이 1이어야 함
port_weight_df = asset_flow_df.divide(asset_flow_df.sum(axis=1), axis=0)
port_weight_df

port_weight_df.sum(axis=1)

# ### 최종 portfolio return 구하기

shifted_rtn_df.head()

port_weight_df.head()

# w*r 을 나타낸 것이 net_rtn_df
net_rtn_df = port_weight_df * shifted_rtn_df
net_rtn_df

# 다시 원래 위치로 돌린다
net_rtn_df = net_rtn_df.shift(1).fillna(0)
net_rtn_df

# 일별 sum
# (total_return_1 = w_A1*r_A1 + w_B1*r_B1 + ...)
rtn_series = net_rtn_df.sum(axis=1)
rtn_series

(rtn_series + 1).cumprod()

# 데이터를 시각화합니다.
(rtn_series + 1).cumprod().plot()

# ### 개별종목별 portval 구하기 (feat. 구현방법 2)

# [학습 포인트] 참고: buy and hold 인 경우, **`asset_flow_df`가 개별종목 portval**

# #### `asset_flow_df`가 개별 portval df인 이유 1

# [학습 포인트] A(1 + $r_{a1}$) -> A(1 + $r_{a1}$)(1 + $r_{a2}$) -> A(1 + $r_{a1}$)(1 + $r_{a2}$)(1 + $r_{a3}$) + ...

# [학습 포인트] 0.1A(1 + $r_{a1}$) -> 0.1A(1 + $r_{a1}$)(1 + $r_{a2}$) -> 0.1A(1 + $r_{a1}$)(1 + $r_{a2}$)(1 + $r_{a3}$) + ...
#     - 첫 설정된 자산인 '0.1A'에 A의 수익률이 그대로 복리계산이 되기 때문에, 계산을 망가뜨리지 않음

cum_rtn_df = df / df.iloc[0]
cum_rtn_df

individual_cum_rtn_df = cum_rtn_df * [0.3, 0.5, 0.2]
individual_cum_rtn_df

# portval
individual_cum_rtn_df.sum(axis=1)   # * 100000000

# #### `asset_flow_df`가 개별 portval df인 이유 2

# [학습 포인트] 직접 식으로 표현해보고 비교해보기(e.g. A:B = 40:60 전략)
#     - individual 관점
#         - day1
#             - A: 0.4
#             - B: 0.6
#             - +: 1
#         - day2
#             - A: 0.4(1 + $r_{a1}$)
#             - B: 0.6(1 + $r_{b1}$)
#             - +: 1 + 0.4$r_{a1}$ + 0.6$r_{b1}$
#         - day3
#             - A: 0.4(1 + $r_{a1}$)(1 + $r_{a2}$) = 0.4(1 + $r_{a1}$ + $r_{a2}$ + $r_{a1}r_{a2}$)
#             - B: 0.6(1 + $r_{b1}$)(1 + $r_{b2}$) = 0.6(1 + $r_{b1}$ + $r_{b2}$ + $r_{b1}r_{b2}$)
#             - +: 1 + 0.4($r_{a1}$ + $r_{a2}$ + $r_{a1}r_{a2}$) + 0.6($r_{b1}$ + $r_{b2}$ + $r_{b1}r_{b2}$)
#                 - (1 + 0.4$r_{a1}$ + 0.6$r_{b1}$) + 0.4($r_{a2}$ + $r_{a1}r_{a2}$) + 0.6($r_{b2}$ + $r_{b1}r_{b2}$)
#     - portfolio 관점
#         - day1
#             - 1
#         - day2
#             - 1 * $(1 + (0.4r_{a1} + 0.6r_{b1}))$
#         - day3
#             - 1 * $(1 + (0.4r_{a1} + 0.6r_{b1}))(1 + ...)$

# 실수할만한 내용

df.head()

rtn_df = get_returns_df(df, log=False)
cum_rtn_df = df / df.iloc[0]

(cum_rtn_df * 1/3).sum(axis=1)

a_1 = (cum_rtn_df * 1/3).sum(axis=1)
a_1

rtn_df * 1/3

a_2 = ((rtn_df * 1/3).sum(axis=1) + 1).cumprod()
a_2

# 주의사항

# [학습 포인트] 2018/1/1부터 0.5/0.5씩 들고 있는 거랑 2018/10/1부터 0.5/0.5씩 들고있는것이랑 결과가 다름

# CSV 파일을 읽어와 데이터프레임으로 저장합니다.
df = pd.read_csv("data/us_etf_1.csv", index_col=0)
df.index = pd.to_datetime(df.index)
df.head()

df1 = df.loc["2017-01-01":]
cum_rtn_df = df1 / df1.iloc[0]

allocation = [0.2, 0.2, 0.2, 0.2, 0.2]

allocation_df = cum_rtn_df * allocation
# 데이터를 시각화합니다.
allocation_df.sum(axis=1).plot(figsize=(7, 3))

df2 = cum_rtn_df.loc["2020-01-02":]
cum_rtn_df2 = df2 / df2.iloc[0]

allocation_df2 = cum_rtn_df2 * allocation

# 데이터를 시각화합니다.
ax = allocation_df.sum(axis=1).plot();
# 데이터를 시각화합니다.
allocation_df2.sum(axis=1).plot(ax=ax);

allocation_df2.head(2)

allocation_df.loc["2020-01-02"]

allocation_df.loc["2020-01-02"].sum()

# 데이터를 시각화합니다.
ax = allocation_df.sum(axis=1).plot();
# 데이터를 시각화합니다.
(allocation_df2.sum(axis=1) * 1.3588167271181084).plot(ax=ax);

# Periodic weight rebalancing

# simple rtn x weight를 이용한 방법

# [학습 포인트] recap
#     - `shifted_rtn_df`은 내 전략과는 상관없이, 시장에서 각 종목들의 수익률을 담은 raw data
#     - `weight_df`가 전략의 로직이 들어있는 데이터
#         - 중요한 점은 `weight_df`를 항상 `asset_flow_df`를 가지고 만들어야한다는 점!
#             - 예를 들어,[0.3, 0.5, 0.2] 상태에서, 며칠 후 해당 자산들 비율이 [0.29, 0.49, 0.19]가 되고 할 것인데, 이런 자산의 흐름을 계산하는 `asset_flow_df`로 먼저구하고, 이 dataframe을 axis=1 방향으로 normalizing 하면 `weight_df`임
#     - 이 둘의 조합을 가지고 어떻게 포트폴리오를 표현할 것인가가 관건

# ### 구현

df = pd.DataFrame(
    {
        "A": [10, 15, 12, 13, 10, 11, 12],
        "B": [10, 10, 8, 13, 12, 12, 12],
        "C": [10, 12, 14, 16, 14, 14, 16],
    },
    index=pd.to_datetime(["2018-01-31", "2018-02-10", "2018-02-20", "2018-02-28", "2018-03-20", "2018-03-29", "2018-04-30",])
)
df

rtn_df = get_returns_df(df, log=False)
shifted_rtn_df = rtn_df.shift(-1).fillna(0)
shifted_rtn_df

df['year'] = df.index.year
df['month'] = df.index.month

rebal_index = df.drop_duplicates(['year','month'], keep="last").index
df.drop(['year', 'month'], axis=1, inplace=True)

rebal_index

month_cum_rtn_df_list = []
for start, end in zip(rebal_index[:-1], rebal_index[1:]):
    month_price_df = df.loc[start:end]
    month_cum_rtn_df = month_price_df / month_price_df.iloc[0]
    month_cum_rtn_df_list.append(month_cum_rtn_df)

month_cum_rtn_df_list[0]
month_cum_rtn_df_list[1]

monthly_asset_flow_df = pd.concat(month_cum_rtn_df_list)
monthly_asset_flow_df

# [학습 포인트] 월말에 해당하는 row: 월말 종가 기준 새롭게 조율된 자산의 비중

# [학습 포인트] 그 이외의 row: 월말로부터 시간이 흘렀을 때 변한 자산의 비중
monthly_asset_flow_df = monthly_asset_flow_df.loc[~monthly_asset_flow_df.index.duplicated(keep="last")]
monthly_asset_flow_df

monthly_asset_flow_df = monthly_asset_flow_df * [0.3, 0.5, 0.2]
monthly_asset_flow_df

# 자산 비중의 흐름을 일별로 sum=1이 되게 만들면 결국 weight_df
weight_df = monthly_asset_flow_df.divide(monthly_asset_flow_df.sum(axis=1), axis=0)
weight_df

net_rtn_df = shifted_rtn_df * weight_df
net_rtn_df

rtn_series = net_rtn_df.sum(axis=1).shift(1).fillna(0)
rtn_series

(rtn_series + 1).cumprod()

# cumulative rtn을 이용하는 방법

# [학습 포인트] event-based

# ### 구현

df = pd.DataFrame(
    {
        "A": [10, 15, 12, 13, 10, 11, 12],
        "B": [10, 10, 8, 13, 12, 12, 12],
        "C": [10, 12, 14, 16, 14, 14, 16],
    },
    index=pd.to_datetime(["2018-01-31", "2018-02-10", "2018-02-20", "2018-02-28", "2018-03-20", "2018-03-29", "2018-04-30",])
)
df['year'] = df.index.year
df['month'] = df.index.month

rebal_index = df.drop_duplicates(['year','month'], keep="last").index
df.drop(['year', 'month'], axis=1, inplace=True)

target_weight_df = pd.DataFrame(
    [[0.3, 0.5, 0.2]]* len(rebal_index),
    index=rebal_index,
    columns=df.columns
)
target_weight_df

individual_port_val_df_list = []
cum_rtn_at_last_month_end = 1

prev_end_day = rebal_index[0]
for end_day in rebal_index[1:]:
    sub_price_df = df.loc[prev_end_day:end_day]
    sub_cum_rtn_df = sub_price_df / sub_price_df.iloc[0]

    weight_series = target_weight_df.loc[prev_end_day]
    # (sub_cum_rtn_df * weight_series): 첫 설정한 weight이 asset 크기의 흐름에 따라 어떻게 변화하는지를 계산함
    indi_port_cum_rtn_df = (sub_cum_rtn_df * weight_series) * cum_rtn_at_last_month_end

    individual_port_val_df_list.append(indi_port_cum_rtn_df)

    total_port_cum_rtn_series = indi_port_cum_rtn_df.sum(axis=1)
    cum_rtn_at_last_month_end = total_port_cum_rtn_series.iloc[-1]

    prev_end_day = end_day

individual_port_val_df_list

# ### 결과비교

# #### 리밸런싱 날, 리밸런싱 바로 직전의 자산 flow 선택

result_port_df_list = [individual_port_val_df_list[0]]
for _df in individual_port_val_df_list[1:]:
    result_port_df_list.append(_df.iloc[1:])

result1_ind = pd.concat(result_port_df_list)
result1_port = result1_ind.sum(axis=1)

# #### 리밸런싱 날, 리밸런싱 직후의 자산 flow 선택

result_port_df_list = []
for i, _df in enumerate(individual_port_val_df_list):
    if i == len(individual_port_val_df_list)-1:
        result_port_df_list.append(_df)
    else:
        result_port_df_list.append(_df.iloc[:-1])

result2_ind = pd.concat(result_port_df_list)
result2_port = result1_ind.sum(axis=1)

result1_port.equals(result2_port)

from matplotlib.pyplot import stackplot

result1_ind.plot.area()

result2_ind.plot.area()

# 실전연습

# 동일가중(equal weight)

# CSV 파일을 읽어와 데이터프레임으로 저장합니다.
price_df = pd.read_csv("data/us_etf_1.csv", index_col=[0], parse_dates=True).drop(
    ["SHY", "TLT", "SPY"], axis=1
)
price_df.head()

price_df['year'] =  price_df.index.year
price_df['month'] = price_df.index.month

rebal_index = price_df.drop_duplicates(subset=['year', 'month'], keep='last').index
price_df = price_df.drop(['year', 'month'], axis=1)
rebal_index

target_weight_df = pd.DataFrame(
    [[1/len(price_df.columns)]*len(price_df.columns)]* len(rebal_index),
    index=rebal_index,
    columns=price_df.columns
)
target_weight_df

cum_rtn_at_last_month_end = 1
individual_port_val_df_list = []

prev_end_day = rebal_index[0]
for end_day in rebal_index[1:]:
    sub_price_df = price_df.loc[prev_end_day:end_day]
    sub_cum_rtn_df = sub_price_df / sub_price_df.iloc[0]

    weight_series = target_weight_df.loc[prev_end_day]
    indi_port_cum_rtn_df = (sub_cum_rtn_df * weight_series) * cum_rtn_at_last_month_end

    individual_port_val_df_list.append(indi_port_cum_rtn_df)

    total_port_cum_rtn_series = indi_port_cum_rtn_df.sum(axis=1)
    cum_rtn_at_last_month_end = total_port_cum_rtn_series.iloc[-1]

    prev_end_day = end_day

from functools import reduce
all_ind_portval_df = reduce(lambda x, y: pd.concat([x, y.iloc[1:]]), individual_port_val_df_list)
all_ind_portval_df

price_df = price_df.loc[all_ind_portval_df.index[0]:]
buy_and_hold_series = price_df["QQQ"] / price_df["QQQ"].iloc[0]
buy_and_hold_series

compare_df = pd.concat(
    [all_ind_portval_df.sum(axis=1), buy_and_hold_series],
    keys=["strategy", "buy_and_hold"], axis=1
)
compare_df.head()

# 데이터를 시각화합니다.
compare_df.plot(figsize=(10, 5));

get_sharpe_ratio(get_returns_df(compare_df, log=True)).to_frame("Sharpe Ratio")

get_CAGR_series(compare_df).to_frame("CAGR")

dd_df, mdd_series, longest_dd_period_df = get_drawdown_infos(compare_df)
# 데이터를 시각화합니다.
dd_df.plot(figsize=(12, 5))

mdd_series.to_frame("MDD")

longest_dd_period_df

# 각자 해보기

# [학습 포인트] 비중변경: 60:40 등

# [학습 포인트] rebal period 변경: monthly, yearly, half year, quarterly
#     - 월초, 월말
#     - 그래도 곡선은 daily로 tracking됨

# [학습 포인트] 본인만의 종목 구성(TQQQ:TLT)

# [학습 포인트] 함수화

# 실전투입 관련 1

# [학습 포인트] 슬리피지, 수수료, 세금
#     - 리벨런싱에서 cum_rtn_at_last_month_end의 값을 전파할때 penalty

# [학습 포인트] 배당
#     - 리벨런싱에서 cum_rtn_at_last_month_end의 값을 전파할때 advantage

# [학습 포인트] 추가매입(물타기)
#     - 리벨런싱
#         - cum_rtn_at_last_month_end의 값을 전파할때 advantage
#     - 특정 %하락 or MDD 갱신 시

from functools import reduce

cum_rtn_at_last_month_end = 100000
individual_port_val_df_list = []
# dividiend_rebal_index =

prev_end_day = rebal_index[0]
for end_day in rebal_index[1:]:
    sub_price_df = price_df.loc[prev_end_day:end_day]
    sub_cum_rtn_df = sub_price_df / sub_price_df.iloc[0]

    weight_series = target_weight_df.loc[prev_end_day]
    indi_port_cum_rtn_df = (sub_cum_rtn_df * weight_series) * cum_rtn_at_last_month_end
    individual_port_val_df_list.append(indi_port_cum_rtn_df)

    total_port_cum_rtn_series = indi_port_cum_rtn_df.sum(axis=1)
    cum_rtn_at_last_month_end = total_port_cum_rtn_series.iloc[-1] * 0.999 * 1.001

    prev_end_day = end_day

all_ind_portval_df1 = reduce(lambda x, y: pd.concat([x, y.iloc[1:]]), individual_port_val_df_list)
all_portval_df1 = all_ind_portval_df1.sum(axis=1)

all_ind_portval_df2 = reduce(lambda x, y: pd.concat([x, y.iloc[1:]]), individual_port_val_df_list)
all_portval_df2 = all_ind_portval_df2.sum(axis=1)

# 데이터를 시각화합니다.
pd.concat([all_portval_df1, all_portval_df2], axis=1).plot()

# [학습 포인트] 오버피팅일 뿐?

# 실전투입 관련 2

# [학습 포인트] 새로운 데이터 인입 --> 몇 주 사고 팔고?

# CSV 파일을 읽어와 데이터프레임으로 저장합니다.
price_df = pd.read_csv("data/us_etf_1.csv", index_col=[0], parse_dates=True).drop(
    ["SHY", "TLT", "SPY"], axis=1
).loc["2021-06-30":]

price_df['year'] =  price_df.index.year
price_df['month'] = price_df.index.month

rebal_index = price_df.drop_duplicates(subset=['year', 'month'], keep='last').index
price_df = price_df.drop(['year', 'month'], axis=1)

target_weight_df = pd.DataFrame(
    [[1/len(price_df.columns)]*len(price_df.columns)]* len(rebal_index),
    index=rebal_index,
    columns=price_df.columns
)

price_df.tail(2)
target_weight_df.tail(2)

cum_rtn_at_last_month_end = 169324
individual_port_val_df_list = []

prev_end_day = rebal_index[0]
for end_day in rebal_index[1:]:
    sub_price_df = price_df.loc[prev_end_day:end_day]
    sub_cum_rtn_df = sub_price_df / sub_price_df.iloc[0]

    weight_series = target_weight_df.loc[prev_end_day]
    indi_port_cum_rtn_df = (sub_cum_rtn_df * weight_series) * cum_rtn_at_last_month_end

    individual_port_val_df_list.append(indi_port_cum_rtn_df)

    total_port_cum_rtn_series = indi_port_cum_rtn_df.sum(axis=1)
    cum_rtn_at_last_month_end = total_port_cum_rtn_series.iloc[-1]

    prev_end_day = end_day

indi_port_cum_rtn_df.head(1)

indi_port_cum_rtn_df.tail(1)

# 주의: 전략에 따라 scalar value or series가 될 수도 있음
target_portval = cum_rtn_at_last_month_end / sub_price_df.shape[1]
target_portval

diff = target_portval - indi_port_cum_rtn_df.iloc[-1]
diff

sub_price_df.iloc[-1]

buy_or_sell_series = diff // sub_price_df.iloc[-1]
buy_or_sell_series

for ticker, qty in buy_or_sell_series.items():
    if qty > 0:
        print("{}: {} x {}주 매수".format(ticker, sub_price_df.iloc[-1].loc[ticker], qty))
    elif qty < 0:
        print("{}: {} x {}주 매도".format(ticker, sub_price_df.iloc[-1].loc[ticker], qty * -1))

# 데이터를 시각화합니다.
(sub_price_df / sub_price_df.iloc[0]).plot()

# 참고자료: 동일가중 vs 60:40 비교 코드

import matplotlib.pyplot as plt

from itertools import product
from functools import reduce

ticker_set_list = [
    ["SPY", "IEF"],
    ["SPY", "SHY"],
    ["QQQ", "IEF"],
    ["QQQ", "SHY"],
]
weight_set_list = [[0.5, 0.5], [0.6, 0.4]]
rebal_set_list = ["month", "yearly"]

# CSV 파일을 읽어와 데이터프레임으로 저장합니다.
total_price_df = pd.read_csv("data/us_etf_1.csv", index_col=[0], parse_dates=True).loc["2002-12-30":]
correct_answer_series_list = []
for ticker_set, weight_set, rebal_period in product(
    ticker_set_list, weight_set_list, rebal_set_list
):
    price_df = total_price_df[ticker_set]

    weight_df = price_df.copy()
    weight_df.loc[:, :] = weight_set

    if rebal_period == "month":
        weight_df['year'] = weight_df.index.year
        weight_df['month'] = weight_df.index.month

        rebal_weight_df = weight_df.drop_duplicates(subset=['year', 'month'], keep='last')
        rebal_weight_df = rebal_weight_df.drop(['year', 'month'], axis=1)
    else:
        weight_df['year'] = weight_df.index.year

        rebal_weight_df = weight_df.drop_duplicates(subset=['year'], keep='last')
        rebal_weight_df = rebal_weight_df.drop(['year'], axis=1)

    first_day = rebal_weight_df.index[0]
    cum_rtn_at_last_month_end = 1
    concat_list = []
    for end_day in rebal_weight_df.index[1:]:
        one_month_price_df = price_df.loc[first_day:end_day]
        one_month_cum_df = one_month_price_df / one_month_price_df.iloc[0]

        weight_series = rebal_weight_df.loc[first_day]
        port_rtn_df = one_month_cum_df.multiply(weight_series)

        final_port_cum_rtn_series = port_rtn_df.sum(axis=1) * cum_rtn_at_last_month_end
        concat_list.append(final_port_cum_rtn_series)

        cum_rtn_at_last_month_end = final_port_cum_rtn_series.iloc[-1]
        first_day = end_day

    correct_answer_series = reduce(lambda x, y: pd.concat([x, y.iloc[1:]]), concat_list)
    correct_answer_series.name = "_".join(ticker_set) + "_" + "_".join([str(w) for w in weight_set]) + "_" + rebal_period

    correct_answer_series_list.append(correct_answer_series)

portval_df = pd.concat(correct_answer_series_list, axis=1)
# 데이터를 시각화합니다.
portval_df.plot(figsize=(15, 8))

a = get_sharpe_ratio(get_returns_df(portval_df, log=True)).to_frame("Sharpe Ratio")
b = get_CAGR_series(portval_df).to_frame("CAGR")
dd_df, mdd_series, longest_dd_period_df = get_drawdown_infos(portval_df)
c = mdd_series.to_frame("MDD")

a.sort_values("Sharpe Ratio", ascending=False).head(30)

longest_dd_period_df

# 데이터를 시각화합니다.
dd_df.plot(figsize=(20, 5))

pd.concat([a, b, c], axis=1).sort_values(["Sharpe Ratio", "CAGR"], ascending=False)
