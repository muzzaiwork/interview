"""
[학습용] lec1_6_project_ForOnlineLecture.py
이 스크립트는 실습 내용을 가독성 좋게 정리한 코드입니다.
"""

#!/usr/bin/env python
# coding: utf-8

# NumPy 라이브러리를 np라는 별칭으로 임포트합니다.
import numpy as np
# Pandas 라이브러리를 pd라는 별칭으로 임포트합니다.
import pandas as pd

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
def print_header(title):
    print(f"\n{'='*20} {title} {'='*20}")


pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)

# Load data

# [학습 포인트] 데이터 출처: 증권사 API, N사 금융, 금투협, 유료 데이터 벤더

# [학습 포인트] Section2: 파일 읽는 법, EDA

# 코드를 돌릴 때 warning이 안나오게 하기
import warnings
warnings.filterwarnings('ignore')

# 영상에서는 fin_statement_2005_2017.csv이지만(데이터 문제가 있는 파일),
# 해당 데이터에서 문제를 발견하여, fin_statement_new.csv라는 데이터(2006 ~ )로 대체되었습니다
# CSV 파일을 읽어와 데이터프레임으로 저장합니다.
df = pd.read_csv("미래에셋자산운용/sample_project/my_data/fin_statement_new.csv")
print_header("df.head()"); print(df.head())

# "12개월전대비수익률(현금배당포함)" 컬럼은 미리 제거하여 파일을 업로드했습니다
df = df.drop(["상장일"], axis=1)

df = df.rename(columns={
    "DPS(보통주, 현금+주식, 연간)": "DPS",
    "P/E(Adj., FY End)": "PER",
    "P/B(Adj., FY End)": "PBR",
    "P/S(Adj., FY End)": "PSR",
})

# 새로 올린 데이터는 2005가 아닌 2006부터 데이터가 존재합니다.
# 특정 열을 기준으로 데이터를 그룹화합니다.
df.groupby(['year'])['Name'].count()
# 특정 열을 기준으로 데이터를 그룹화합니다.
df.groupby(['Name'])['year'].count()

# code or name의 중복 체킹 방법1
# 특정 열을 기준으로 데이터를 그룹화합니다.
df.groupby(['year'])['Name'].nunique().equals(df.groupby(['year'])['Code'].nunique())

# code or name의 중복 체킹 방법2
# 특정 열을 기준으로 데이터를 그룹화합니다.
df.groupby(['year', 'Name'])['Code'].nunique()

# 특정 열을 기준으로 데이터를 그룹화합니다.
df.groupby(['year', 'Name'])['Code'].nunique().nunique()

# yearly returns

df[df['Name'] == '동화약품']

# [학습 포인트] Section4: `pivot()`

yearly_price_df = df.pivot(index="year", columns="Name", values="수정주가")
yearly_price_df.head()

# [학습 포인트] rtn 구하기
#     - $p_{n+1}\over{p_{n}}$ - 1

# 1. year_price_df.pct_change() == year_price_df / year_price_df.shift() - 1
# 2. `shift(-1)`을 하는 이유?
#    - 데이터를 "xx년도에서 1년동안 들고있었더니, xx만큼 수익이 났다"로 해석하고 싶기 때문
yearly_rtn_df = yearly_price_df.pct_change(fill_method=None).shift(-1)
yearly_rtn_df.head()

# [학습 포인트] 상장폐지 종목은 어떻게 처리가 되나?

yearly_price_df['AD모터스']

yearly_price_df['AD모터스'].pct_change(fill_method=None).shift(-1)

# [학습 포인트] 2011/12에 매수했으면, 1년의 rtn value는은 보장됨.

# [학습 포인트] 2012/12에 매수했으면,
#     - 2013년 1월에 상장폐지 되었을 수도 있고, 2013년 12월(초)에 되었을 수도 있기 때문에 => rtn이 nan처리됨

# Single Indicator(지표) backtesting

# [학습 포인트] Section1: `reset_index()`

# [학습 포인트] Section2: boolean selection, DataFrame arithmetic operation, dtype변환

# [학습 포인트] Section3: `groupby()` & `aggregation`

# [학습 포인트] Section4: `join()`, `pivot()`

# [학습 포인트] Section5: visualization

# DataFrame(matrix) Multiplication 복습

a = pd.DataFrame([[1,2], [3, np.nan,], [5,6]], columns=["a", "b"])
b = pd.DataFrame([[1,2], [3, 4,], [5,6]], columns=["a", "b"])*10
print("a:\n", a)
print("b:\n", b)

a * b

a = pd.DataFrame([[1,2], [3, np.nan,], [5,6]], columns=["a", "b"])
b = pd.DataFrame([[1,2,3], [3, 4,5], [5,6,7]], columns=["c", "b", "d"])*10
print("a:\n", a)
print("b:\n", b)

a * b

return_df = pd.DataFrame(
    [
        [np.nan,  np.nan, 2     ],
        [3,       np.nan, 3     ],
        [5,       6,      np.nan],
    ],
    columns=["삼성", "현대", "SK"]
)
asset_on_df = pd.DataFrame(
    [
        [0, 1],
        [0, 1],
        [1, 0],
    ],
    columns=["삼성", "SK"]
)
return_df
asset_on_df

return_df * asset_on_df

(return_df * asset_on_df).mean(axis=1)

# 해결책
asset_on_df = asset_on_df.replace(0, np.nan)

return_df * asset_on_df

# "동일가중" 방식의 투자인 경우, 포트폴리오 평균수익률 구하는 방법
(return_df * asset_on_df).mean(axis=1)

# top_n

print_header("df.head()"); print(df.head())

indicator = "ROA"

top_n = 10

# 특정 열을 기준으로 데이터를 그룹화합니다.
top_n_indicator_df = df.groupby(['year'])[indicator].nlargest(top_n).reset_index()
top_n_indicator_df.head()
top_n_indicator_df.tail()

# 종목 indexing
top_n_roa_df = df.loc[top_n_indicator_df['level_1']]
top_n_roa_df.head()

indicator_df = top_n_roa_df.pivot(index="year", columns="Name", values="ROA")
indicator_df.head()

# [학습 포인트] 주의: nan 값을 가지고 있는 종목은 아예 고려대상에서 배제됨(물론 agg 함수의 연산특성에 따라 다르기는하나, 대부분의 함수가 nan은 배제시키고 계산함)

# [학습 포인트] 깜짝 퀴즈
#     - 각 row별, nan이 아닌 값이 정확히 top_n개 만큼 인지 확인하는 방법?

# backtest

indicator_df.head()

# ### 포트폴리오 수익률 데이터

asset_on_df = indicator_df.notna().astype(int).replace(0, np.nan)
asset_on_df.head()

# 지난 영상 퀴즈 정답1
yearly_rtn_df.shape
asset_on_df.shape

# 지난 영상 퀴즈 정답2
asset_on_df.notna().sum(axis=1)

selected_return_df = yearly_rtn_df * asset_on_df
selected_return_df.head()

selected_return_df.notna().sum(axis=1)

a = asset_on_df.iloc[0]
a[a.notna()]

b = yearly_rtn_df.iloc[0]
b[a[a.notna()].index]

rtn_series = selected_return_df.mean(axis=1)
rtn_series.head()

# 새로 수정된 데이터(fin_statement_new.csv)에서는 데이터 2006부터 시작하므로, 2005를 0으로 설정한 점에 주의바랍니다.
rtn_series.loc[2005] = 0
rtn_series = rtn_series.sort_index()
rtn_series

# ### 포트폴리오 누적 수익률 데이터

cum_rtn_series = (rtn_series + 1).cumprod().dropna()
cum_rtn_series

pd.Series([1,2,3,4,5]).cumsum()

# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=2, figsize=(15, 6), sharex=True)

# 데이터를 시각화합니다.
axes[0].plot(cum_rtn_series.index, cum_rtn_series, marker='o');
axes[0].set_title("Cum return(line)");

axes[1].bar(rtn_series.index, rtn_series);
axes[1].set_title("Yearly return(bar)");

# 함수화

def get_return_series(selected_return_df):
    rtn_series = selected_return_df.mean(axis=1)
    rtn_series.loc[2005] = 0     # 주의: 영상속의 데이터와는 달리, 새로 업로드 된 데이터는 2006부터 존재하므로
                                 # 2004가 아니라 2005를 0으로 설정한 점에 주의해주세요
    rtn_series = rtn_series.sort_index()

    cum_rtn_series = (rtn_series + 1).cumprod().dropna()
    return rtn_series, cum_rtn_series

def plot_return(cum_rtn_series, rtn_series):
    fig, axes = plt.subplots(nrows=2, figsize=(15, 6), sharex=True)
# 데이터를 시각화합니다.
    axes[0].plot(cum_rtn_series.index, cum_rtn_series, marker='o');
    axes[1].bar(rtn_series.index, rtn_series);
    axes[0].set_title("Cum return(line)");
    axes[1].set_title("Yearly return(bar)");

rtn_series, cum_rtn_series = get_return_series(selected_return_df)

plot_return(cum_rtn_series, rtn_series)

# quantile (e.g. 상위 n% 종목 선정)

# 특정 열을 기준으로 데이터를 그룹화합니다.
quantile_by_year_series = df.groupby(['year'])[indicator].quantile(0.9)
quantile_by_year_series

quantilie_indicator_df = df.join(quantile_by_year_series, how="left", on="year", rsuffix="_quantile")
quantilie_indicator_df.head(2)

quantilie_indicator_df = quantilie_indicator_df[
    quantilie_indicator_df[indicator] >= quantilie_indicator_df["{}_quantile".format(indicator)]
]
quantilie_indicator_df.head()

# 특정 열을 기준으로 데이터를 그룹화합니다.
quantilie_indicator_df.groupby('year')['Code'].count()

indicator_df = quantilie_indicator_df.pivot(index='year', columns="Name", values=indicator)
asset_on_df = indicator_df.notna().astype(int).replace(0, np.nan)

selected_return_df = yearly_rtn_df * asset_on_df
selected_return_df.head()

rtn_series, cum_rtn_series = get_return_series(selected_return_df)
plot_return(cum_rtn_series, rtn_series)

# 강환국님의 "할수있다 퀀트투자" 구현해보기

# ![](http://image.kyobobook.co.kr/images/book/large/392/l9791195887392.jpg)

# quantile + top10

# [학습 포인트] Filter + Selector 구조
#     - Filter
#         - e.g. 부채비율 0.5이상
#         - 최종 포트폴리오 종목 갯수 선정에 직접적으로 영향 X
#     - Selector
#         - 최종적으로 xx개의 종목이 선택의 기준이 되는 indicator
#         - e.g. PBR이 0.2 이상인 회사 중에 가장 낮은 순으로 20~30개 매수

# [학습 포인트] zipline (https://github.com/quantopian/zipline)

# [Chapter 6] 투자전략22. 소형주 + 저PBR 전략(200p)

# [학습 포인트] Filter
#     - 소형주(시가총액 하위 20%)

# [학습 포인트] Select
#     - PBR 0.2 이상
#     - PBR이 가장 낮은 주식순으로 20~30개 매수

# Filter
# 특정 열을 기준으로 데이터를 그룹화합니다.
market_cap_quantile_series = df.groupby("year")['시가총액'].quantile(.2)

filtered_df = df.join(market_cap_quantile_series, on="year", how="left", rsuffix="20%_quantile")
filtered_df = filtered_df[filtered_df['시가총액'] <= filtered_df['시가총액20%_quantile']]
filtered_df.head()

# Selector
filtered_df = filtered_df[filtered_df['PBR'] >= 0.2]

# 특정 열을 기준으로 데이터를 그룹화합니다.
smallest_pbr_series = filtered_df.groupby("year")['PBR'].nsmallest(15)
smallest_pbr_series

selected_index = smallest_pbr_series.index.get_level_values(1)

selector_df = filtered_df.loc[selected_index].pivot(
    index='year', columns="Name", values="PBR"
)
selector_df.head()

asset_on_df = selector_df.notna().astype(int).replace(0, np.nan)
selected_return_df = yearly_rtn_df * asset_on_df

rtn_series, cum_rtn_series = get_return_series(selected_return_df)
plot_return(cum_rtn_series, rtn_series)

# [Chapter 5] 투자전략20. 그레이엄의 마지막선물 업그레이드(188p)

# [학습 포인트] Filter
#     - ROA 5% 이상
#     - 부채비율 50% 이하

# [학습 포인트] Select
#     - (PBR 0.2 이상)
#     - PBR 낮은기업 20~30개 매수

# Filter

# ROA >= 0.05
filtered_df = df[df['ROA'] >= 0.05]

# 부채비율 <= 0.5
filtered_df['부채비율'] = filtered_df['비유동부채'] / filtered_df['자산총계']
filtered_df = filtered_df[filtered_df['부채비율'] <= 0.5]

# Selector(위의 투자전략22 것 그대로)
filtered_df = filtered_df[filtered_df['PBR'] >= 0.2]

# 특정 열을 기준으로 데이터를 그룹화합니다.
smallest_pbr_series = filtered_df.groupby("year")['PBR'].nsmallest(15)
selected_index = smallest_pbr_series.index.get_level_values(1)

selector_df = filtered_df.loc[selected_index].pivot(
    index='year', columns="Name", values="PBR"
)

asset_on_df = selector_df.notna().astype(int).replace(0, np.nan)
selected_return_df = yearly_rtn_df * asset_on_df

rtn_series, cum_rtn_series = get_return_series(selected_return_df)
plot_return(cum_rtn_series, rtn_series)

# [Chapter 8] 투자전략24. 슈퍼가치전략(246p)

# [학습 포인트] Filter
#     - 시가총액 하위 20%

# [학습 포인트] Selector
#     - PBR, PCR, PER, PSR 순위를 매김
#     - 각 순위를 sum을 해서 통합순위를 구함
#     - 통합순위가 가장 높은 종목 50개 매수

# Filter
# 특정 열을 기준으로 데이터를 그룹화합니다.
market_cap_quantile_series = df.groupby("year")['시가총액'].quantile(.2)
filtered_df = df.join(market_cap_quantile_series, on="year", how="left", rsuffix="20%_quantile")
filtered_df = filtered_df[filtered_df['시가총액'] <= filtered_df['시가총액20%_quantile']]

pd.Series([100, 1, 1, 3]).rank(method="max")
pd.Series([100, 1, 1, 3]).rank(method="min")

# 특정 열을 기준으로 데이터를 그룹화합니다.
pbr_rank_series = filtered_df.groupby("year")['PBR'].rank(method="max")
# 특정 열을 기준으로 데이터를 그룹화합니다.
per_rank_series = filtered_df.groupby("year")['PER'].rank(method="max")
# 특정 열을 기준으로 데이터를 그룹화합니다.
psr_rank_series = filtered_df.groupby("year")['PSR'].rank(method="max")

psr_rank_series.head()

psr_rank_series.sort_values().dropna().head()

filtered_df = filtered_df.join(pbr_rank_series, how="left", rsuffix="_rank")
filtered_df = filtered_df.join(per_rank_series, how="left", rsuffix="_rank")
filtered_df = filtered_df.join(psr_rank_series, how="left", rsuffix="_rank")

filtered_df['PBR_rank'].isna().sum()

# [학습 포인트] 어떻게 각 rank column의 nan을 메꿔야할까?

filtered_df.filter(like="rank").columns

# 주의: 종목을 선택하는 로직ㅇ[ 따라, '가장 작은 rank'로 부여하는게 타당할 수도 있고, '가장 큰 rank'로 부여하는 것이 타당할 수도 있습니다.
# 예를들어, PER이 작을수록 종목 선정에 우선 순위가 있도록 할 예정이고, PER이 작을수록 rank값이 작도록 설정했다면,
# PER이 nan인 종목들은 PER rank가 가장 큰 값(혹은 그 값보다 +1인 값)으로 메꿔져야 penalty를 받을 수 있습니다.

# 1. 0으로 메꾸는 법
filtered_df.loc[:, filtered_df.filter(like="rank").columns] = filtered_df.filter(like="rank").fillna(0)

# 2. 각 rank별 max 값 (혹은 그것보다 1 큰 값)으로 메꾸는 법
# filtered_df['PBR_rank'] = filtered_df['PBR_rank'].fillna(filtered_df['PBR_rank'].max() + 1)
# filtered_df['PER_rank'] = filtered_df['PER_rank'].fillna(filtered_df['PER_rank'].max() + 1)
# filtered_df['PSR_rank'] = filtered_df['PSR_rank'].fillna(filtered_df['PSR_rank'].max() + 1)

filtered_df['rank_sum'] = filtered_df.filter(like="_rank").sum(axis=1)

# Selector
# 특정 열을 기준으로 데이터를 그룹화합니다.
max_rank_series = filtered_df.groupby("year")['rank_sum'].nlargest(15)
selected_index = max_rank_series.index.get_level_values(1)

selector_df = filtered_df.loc[selected_index].pivot(
    index='year', columns="Name", values="rank_sum"
)

asset_on_df = selector_df.notna().astype(int).replace(0, np.nan)
selected_return_df = yearly_rtn_df * asset_on_df

rtn_series, cum_rtn_series = get_return_series(selected_return_df)
plot_return(cum_rtn_series, rtn_series)

# 재무제표 기반 실전 프로젝트의 한계

# [학습 포인트] **(중요)Look ahead bias & Survivalship bias**
#     - 특정 년도에 상장이 폐지가 되었다면 -> 바로 이전 년도에서 종목선정에 고려가 안됨
#     - 즉, 이미 상장 폐지 정보를 미래 시점에서 확인하고, 해당 년도의 수익률을 nan으로 미리 메꾸어 버림

# [학습 포인트] Data availability(time alignment)
#     - 각 투자지표의 값들이 공시 되는 시기
#         - 년도별, 분기별
#     - 정확한 상장폐지 날짜?

# [학습 포인트] Data acquisition
#     - 고정된 과거데이터로만 테스트 하면 안됨 -> 계속 새로운 데이터에 대한 갱신 필요
#     - 크롤링, 증권사 API, 유료 데이터 벤더 등

# [학습 포인트] Data의 무결성
#     - 아무리 증권사 API나 유로 벤더를 통해서 받아온 데이터라도, 문제가 있는 경우가 많음
#     - 예를 들어, 일봉 OHLC -> C가 H보다 더 큰 경우 / 배당락, 주식분할 등의 이벤트가 제대로 반영이 안된 경우 등
#     - 데이터의 결함, nunique==1, 비이상적인 값 등에 대한 EDA 필요

# [학습 포인트] 데이터가 년도별로만 존재하기 때문에, 1년에 한번 수익률이 찍혀서 변동성, MDD를 제대로 파악하기 어려움

# [학습 포인트] 거래세, 수수료 반영 X
#     - 정확한 asset turnover 고려가 안됨

# [학습 포인트] 기타 위 실전예제에서의 한계
#     - 데이터의 cleaning, validation 필요
#     - Missing value에 대한 전처리 필요
#     - 주어진 데이터 존재하지 않는 지표(column)은 다른 지표로 대체한 점

# 혼자 진행해보면 좋을 것들

# [학습 포인트] `transform()`, `apply()` 함수 등을 구글링해서 독학해보기

# [학습 포인트] OOP 방식으로 구현해보기(확장성 있는 코드화)

filter_list = [
    ColumnIndicator("부채비율", 0.5, lower_than=False),
    ColumnIndicator("ROE", 0.5, lower_than=True),
]
selector = Selector("PBR", 20, lowest=True)

backtest = Backtest(filter_list, selector, yearly_rtn_df)
backtest.run()

# [학습 포인트] 거래비용 주기

for_positive_df = (yearly_rtn_df > 0).astype(int) * 0.99
for_negative_df = (yearly_rtn_df < 0).astype(int) * 1.01

extra_fee_considered_weight_df = for_positive_df + for_negative_df
yearly_rtn_df = yearly_rtn_df * extra_fee_considered_weight_df

# [학습 포인트] weight
#     - 지금까지는 종목 선택 후, 다 동일가중 투자(`mean()`만으로 평균수익률을 구할 수 있었음)
#     - e.g. ROA의 비중만큼 넣기
#         - Top n개의 종목을 산출 후, 각각의 값을 1,0(nan)으로 변환하기 전에, `selector_df`를 sum(axis=1)로 나눠주기 등

# [학습 포인트] 분기별 데이터 & 리벨런싱
