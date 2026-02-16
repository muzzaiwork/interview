"""
[학습용] lec1_5_visualization_ForOnelineLecture.py
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

# Matplotlib overview

import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
def print_header(title):
    print(f"\n{'='*20} {title} {'='*20}")


# ![](https://matplotlib.org/_images/anatomy.png)

# ![](https://static.packt-cdn.com/products/9781784393878/graphics/9ec8935e-0bdc-46aa-b392-18c5431a2218.png)

# [학습 포인트] 출처
#     - https://matplotlib.org
#     - https://subscription.packtpub.com/book/data/9781784393878/11/ch11lvl1sec112/getting-started-with-matplotlib

# [학습 포인트] 2가지 구성요소
#     - Figure
#         - 틀
#     - Axes
#         - Figure 바로 아래에, 실제 그래프가 그려질 공간
#         - 실제로 가장 많이 사용할 요소
#             - 이 안에 각종 plotting components가 존재

# Matplotlib의 2가지 인터페이스

# [학습 포인트] Matplotlib이 배우고 익히기 어려운 이유 중 하나

import FinanceDataReader as fdr

samsung_df = fdr.DataReader('005390', '2017-01-01', '2017-12-31')
samsung_df.head()

# Stateful

# [학습 포인트] Matplotlib이 암묵적으로 현재 상태를 들고 있음
#     - 내부적으로 현재 타겟이 되는 figure, ax 등을 설정하고, operation이 발생하면 '내부에서' 해당 figure,ax에 적용함

# [학습 포인트] 사용은 비추
#     - matplotlib이 암묵적, 내부적으로 변화를 진행하고 적용하기 때문에, 직관적이지 못함
#     - 다수의 plot을 한번에 그리기 어려움
#     - 그냥 간단히 테스트 해볼 때 정도에만 사용

x = [1,2,3]
y = [4,5,6]

# 데이터를 시각화합니다.
something = plt.plot(x, y)
# 생성된 그래프를 화면에 출력합니다.
# plt.show() -> matplotlib inline magic command를 실행하지 않았으면, 항상 필요!

something

type(something)

type(something[0])

# 딱 그래프만 출력이 되게하고, return이 되는 list는 안보이게 만드는 방법
# 데이터를 시각화합니다.
_ = plt.plot(x, y)
# 데이터를 시각화합니다.
plt.plot(x, y);

# [학습 포인트] 예제1

x = [-3, 5, 7]
y = [10, 2, 5]

plt.figure(figsize=(15, 3));

# 데이터를 시각화합니다.
plt.plot(x, y);
plt.xlim(0, 10);
plt.ylim(-3, 8);
plt.xlabel('X Axis');
plt.ylabel('Y axis');
plt.title('Line Plot');
plt.suptitle('Figure Title', size=10, y=1.03);

# [학습 포인트] 예제2

samsung_df.head()

# 데이터를 시각화합니다.
plt.plot(
    samsung_df.index,
    samsung_df['Close']
)

# Stateless(or object-oriented)

# [학습 포인트] Matplotlib의 각 component를 하나의 object로 받아서, 함수 실행 및 property 설정/변경
#     - figure, ax(es)를 먼저 생성한다음, 하나하나 더하고, 적용하는 식

# [학습 포인트] 적용과정이 명시적으로 코드로 드러나기 때문에 조금 더 직관적임

x = [-3, 5, 7]
y = [10, 2, 5]

fig, ax = plt.subplots(figsize=(15, 3))

type(fig)
type(ax)

# 데이터를 시각화합니다.
ax.plot(x, y);
ax.set_xlim(0, 10);
ax.set_ylim(-3, 8);
ax.set_xlabel('X axis');
ax.set_ylabel('Y axis');
ax.set_title('Line Plot');
fig.suptitle('Figure Title', size=10, y=1.03);

fig

# [학습 포인트] 한번에 시각화 그래프가 나오게 하기

fig, ax = plt.subplots(figsize=(15, 3))
# 데이터를 시각화합니다.
ax.plot(x, y);
ax.set_xlim(0, 10);
ax.set_ylim(-3, 8);
ax.set_xlabel('X axis');
ax.set_ylabel('Y axis');
ax.set_title('Line Plot');
fig.suptitle('Figure Title', size=10, y=1.03);

fig, ax = plt.subplots(figsize=(15, 3))
# 데이터를 시각화합니다.
ax.plot(samsung_df.index, samsung_df['Close'])

# -> OOP 방식으로 익히는 것이 확장성 및 추후 새로운 visualization lib에 대해 익힐 때 더 도움이 많이 됨!

# Matplotlib components에 대해 조금 더 깊게 들여다보기

# figure, axes

fig, ax = plt.subplots(figsize=(5, 5))

ax

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

type(fig)
type(axes)   # 2차원 array (행렬형태)

axes[0][0]   # 2차원 array에 대한 indexing으로 target ax에 접근

# [학습 포인트] nrows or ncols가 1보다 크면, `ax`의 type은 `AxesSubplot`가 아니라 numpy array of `AxesSubplot`

# Children of ax(es)

axes[0][0].get_children()

# [학습 포인트] `spines`: axes를 둘러싸고 있는 border

# [학습 포인트] `axis`: x,y축
#     - `ticks`, `labels` 등을 가지고 있음

# [학습 포인트] `axis`
#     - **Tip: get/set 관련 함수들을 잘 이용하기**

ax = axes[0][0]

ax.xaxis

# get_xaxis() 메소드 내부에는 `return self.xaxis` 와 같이 구현이 되어있습니다.
ax.get_xaxis()

ax.xaxis == ax.get_xaxis()

# 예제

data = fdr.DataReader("005930", start="2019-01-01", end="2020-01-01")
close_series = data['Close']
volume_series = data['Volume']

close_series.head()
volume_series.head()

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14,10), sharex=True)
ax1 = axes[0]
ax2 = axes[1]

# ax1
# 데이터를 시각화합니다.
ax1.plot(close_series.index, close_series, linewidth=2, linestyle='--', label="Close");
_ = ax1.set_title('Samsung price', fontsize=15, family='Arial');
_ = ax1.set_ylabel('price', fontsize=15, family='Arial');
_ = ax1.set_xlabel("date", fontsize=15, family='Arial');
ax1.legend(loc="upper left");

# ax2
ax2.bar(volume_series.index, volume_series, label="volume");
_ = ax2.set_title('Samsung volume', fontsize=15, family='Arial');
_ = ax2.set_ylabel('volume', fontsize=15, family='Arial');
_ = ax2.set_xlabel("date", fontsize=15, family='Arial');
ax2.legend(loc="upper left");

fig.suptitle("<Samsung>", fontsize=15, family='Verdana');

# [학습 포인트] 참고

fig, ax = plt.subplots()
# 데이터를 시각화합니다.
ax.plot([5,6,7,8], marker='x')

# Plotting with Pandas

# [학습 포인트] DataFrame, Series는 `plot()`을 호출하면, 내부적으로 matplotlib api를 호출함

# [학습 포인트] plot을 시행한 후 `ax`를 return함

# [학습 포인트] matplotlib arg는 그대로 전달 가능

# [학습 포인트] plot의 종류(`kind` arg)
#     - `bar, line, scatter`, etc
#     - `hist, box`, etc

import FinanceDataReader as fdr

samsung_series = fdr.DataReader("005930", "2017-01-01", "2018-01-01")['Close']
kodex_series = fdr.DataReader("069500", "2017-01-01", "2018-01-01")['Close']

price_df = pd.concat([samsung_series, kodex_series], axis=1)
price_df.columns = ["삼성전자", "KODEX 200"]
price_df.head()

# 특정 열을 기준으로 데이터를 그룹화합니다.
price_max_df = price_df.groupby(price_df.index.month).max()
price_max_df.head()

# Series 혹은 DataFrame 변수에 대해 plot()함수만 호출하면 기본적인 plotting이 진행됨
# 데이터를 시각화합니다.
price_max_df.plot()

# [학습 포인트] 미리 설정한 fig, ax에 대해 plotting하기

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))

# 데이터를 시각화합니다.
price_max_df.plot(ax=ax1, kind='line');
# 데이터를 시각화합니다.
price_max_df.plot(ax=ax2, kind='bar');
# 데이터를 시각화합니다.
price_max_df.plot(ax=ax3, x='삼성전자', y='KODEX 200', kind='scatter');

price_max_df.hist(figsize=(15, 4), bins=30);

price_df.pct_change() # => p2/p1 - 1

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))

# 데이터를 시각화합니다.
price_df.pct_change().plot(kind='kde', ax=ax1, title='kde');
# 데이터를 시각화합니다.
price_df.pct_change().plot(kind='box', ax=ax2, title='box');
# 데이터를 시각화합니다.
price_df.pct_change().plot(kind='hist', ax=ax3, title='hist', bins=30);

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))

# 데이터를 시각화합니다.
price_df.pct_change().plot(x="삼성전자", kind='kde', ax=ax1, title='kde');
# 데이터를 시각화합니다.
price_df.pct_change().plot(x="삼성전자", kind='box', ax=ax2, title='box');
# 데이터를 시각화합니다.
price_df.pct_change().plot(x="삼성전자", kind='hist', ax=ax3, title='hist', bins=30);

# [학습 포인트] 한글 Font 가능하게
#     - Google에 "matplotlib 한글" or "matplotlib 한글 windows" 이라고 검색
#     - Window: https://financedata.github.io/posts/matplotlib-hangul-for-windows-anaconda.html
#     - Mac OS / Linux : http://corazzon.github.io/matplotlib_font_setting
#     - https://programmers.co.kr/learn/courses/21/lessons/950 등등

import matplotlib.font_manager as fm

for f in fm.fontManager.ttflist:
    if 'Gothic' in f.name:
        print((f.name, f.fname))

import matplotlib as mpl
import platform

# 운영체제별 한글 폰트 설정
if platform.system() == 'Darwin': # Mac
    plt.rcParams["font.family"] = 'AppleGothic'
elif platform.system() == 'Windows': # Windows
    plt.rcParams["font.family"] = 'Malgun Gothic'
else: # Linux
    plt.rcParams["font.family"] = 'NanumGothic'

# 마이너스 기호 깨짐 방지
mpl.rcParams['axes.unicode_minus'] = False

# matplotlib minus problem mac os

# Seaborn

# get_ipython().system('pip install seaborn==0.9.0')

import seaborn as sns

# CSV 파일을 읽어와 데이터프레임으로 저장합니다.
df = pd.read_csv("미래에셋자산운용/sample_project/my_data/Small_and_Big.csv", index_col=0, parse_dates=["date"])
print_header("df.head()"); print(df.head())

# 특정 열을 기준으로 데이터를 그룹화합니다.
median_df = df.groupby(['date']).agg({'시가총액 (보통)(평균)(원)': 'median'})
median_df.columns = ["median_시가총액"]
median_df.head()

df = df.join(median_df, on="date")

df.loc[df['시가총액 (보통)(평균)(원)'] < df['median_시가총액'], "size"] = "small"
df.loc[df['시가총액 (보통)(평균)(원)'] >= df['median_시가총액'], "size"] = "big"

print_header("df.head()"); print(df.head())

# Count plot

# ### matplotlib version

df['size'].value_counts()

# 데이터를 시각화합니다.
df['size'].value_counts().plot(kind='bar');

df['size'].hist()

# ### seaborn version

sns.countplot(x="size", data=df)

# 수익률 bar plot

df.shape

# 데이터 사이즈 줄이기
df = df[df['date'] >= "2017-01-01"]

df.shape

print_header("df.head()"); print(df.head())

# 날짜 x tick label을 조금더 심플하게 나타나도록 만들기: DateTime object -> 문자열 object로 변환
df['date'] = df['date'].dt.strftime("%Y-%m-%d")

# 특정 열을 기준으로 데이터를 그룹화합니다.
df.groupby(['date'])['수익률(%)'].mean()

# 특정 열을 기준으로 데이터를 그룹화합니다.
df.groupby(['date'])['수익률(%)'].mean().plot(kind='bar', figsize=(18, 3))

# datetime
# strftime
# strptime

# 특정 열을 기준으로 데이터를 그룹화합니다.
df.groupby(['date'])['수익률(%)'].mean().plot(kind='bar', figsize=(18, 3))

sns.barplot(data=df, x="date", y="수익률(%)")

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 3))
ax = sns.barplot(data=df, x="date", y="수익률(%)", ax=ax);

# [학습 포인트] x tick label을 45도 돌리기

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 3))
ax = sns.barplot(data=df, x="date", y="수익률(%)", ax=ax);

current_x_tick_label = ax.get_xticklabels()
ax.set_xticklabels(current_x_tick_label, rotation=45);

# [학습 포인트] hue 넣기

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 3))
sns.barplot(data=df, x="date", y="수익률(%)", ax=ax, hue="size")

current_x_tick_label = ax.get_xticklabels()
ax.set_xticklabels(current_x_tick_label, rotation=45);

# relation plot (다차원 그래프)

df.head(2)

sns.relplot(
    x="PBR(IFRS-연결)",
    y="수익률(%)",
    col="size",
    hue="베타 (M,5Yr)",
    data=df,

    palette="coolwarm",
)

with sns.plotting_context("notebook", font_scale=1.2):
    sns.relplot(
        x="PBR(IFRS-연결)",
        y="수익률(%)",
        col="size",
        hue="베타 (M,5Yr)",
        palette="coolwarm",
        data=df
    )

with sns.plotting_context("notebook", font_scale=1.2):
    sns.relplot(
        x="PBR(IFRS-연결)",
        y="수익률(%)",
        size="size",           # `col` 대신 `size`사용
        hue="베타 (M,5Yr)",
        palette="coolwarm",
        data=df
    )

# 실전예제

df_list = []
for i in range(2015, 2018):
    df_list.append(
# CSV 파일을 읽어와 데이터프레임으로 저장합니다.
        pd.read_csv("미래에셋자산운용/sample_project/my_data/naver_finance/{}_12.csv".format(i))
    )

df = pd.concat(df_list)

print_header("df.head()"); print(df.head())

df = df.dropna()

df['rtn'] = df['price2'] / df['price'] - 1

# outlier(이상치) 제거하기
for col in df.columns:
    if col not in ['ticker', 'price2', 'price', 'rtn']:
        mu = df[col].mean()
        std = df[col].std()

        cond1 = mu - 2*std <= df[col]
        cond2 = df[col] <= mu + 2*std

        df = df[cond1 & cond2]

# with sns.plotting_context("notebook", font_scale=1.2):
sns.relplot(
    x="순이익률(%)",
    y="rtn",
    hue="ROA(%)",
    palette="coolwarm",
    data=df
)

# with sns.plotting_context("notebook", font_scale=1.2):
sns.relplot(
    x="PSR(배)",
    y="rtn",
    hue="당기순이익(억원)",
    palette="coolwarm",
    data=df
)

# [학습 포인트] Seaborn plot 종류
#     - https://seaborn.pydata.org/examples/index.html
