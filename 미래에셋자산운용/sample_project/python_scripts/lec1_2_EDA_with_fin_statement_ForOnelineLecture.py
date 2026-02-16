"""
[학습용] lec1_2_EDA_with_fin_statement_ForOnelineLecture.py
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


# Pandas DataFrame의 사이즈가 큰 경우, 어떻게 화면에 출력을 할지를 세팅하는 코드
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
pd.set_option("display.max_columns", None)

# Load data using `read_csv`

# [학습 포인트] 미국시장 재무제표 데이터 크롤링: https://nbviewer.jupyter.org/gist/FinanceData/35a1b0d5248bc9b09513e53be437ac42

# CSV 파일을 읽어와 데이터프레임으로 저장합니다.
df = pd.read_csv("미래에셋자산운용/sample_project/my_data/naver_finance/2015_12.csv")

print_header("df.head()"); print(df.head())

# Exploratory Data Analysis (EDA)

# [학습 포인트] In statistics, exploratory data analysis (EDA) is an approach to analyzing data sets to summarize their main characteristics, often with visual methods(wiki)

# [학습 포인트] Two parts
#     - *Metadata* : data about data.
#         - 데이터 크기
#         - 컬럼명
#         - 데이터 타입
#         - 비어 있는 데이터
#         - etc
#     - *Univariate descriptive statistics*: summary statistics about individual variables(columns)

# Metadata

df.shape

df.dtypes.value_counts()

df.info()

df['ticker'].dtype

# [학습 포인트] Rename columns

df = df.rename(columns={"ticker": "종목명"})

print_header("df.head()"); print(df.head())

# `describe()`

df.shape

df.describe()

# Trnaspose (index <-> columns 뒤집기)
# 함수의 cascading을 통해서 한번에 진행 가능
df.describe().T

# 1. numeric
df.describe(include=[np.number]).T  #  = df.describe()의 기본(default) 작동방식과 같습니다

# dtype을 나타낼 때, string으로 해도 되고, library의 datatype으로 설정해도 됩니다. (astype() function을 쓸 때도 마찬가지)
# (아래 4개는 다 같은 구문 )
df.describe(include=['int', 'float']).T
df.describe(include=['int64', 'float64']).T
df.describe(include=[np.int64, np.float64]).T
df.describe(include=['number']).T
df.describe(include=[np.number]).T

df.describe(percentiles=[0.01, 0.03, 0.99]).T.head(2)

# 2. non-numeric (e.g. string, categorical)
df.describe(exclude=[np.number]).T # 'top'은 "가장 많이 출현하는 단어"를 의미함

# [학습 포인트] exclude

df.describe(exclude=[np.number]).T.head()

# [학습 포인트] 참고: `quantile()` method

df['PER(배)'].quantile(.2)
df['PER(배)'].quantile([.1, .2, .3])

# `unique(), value_counts()`

# For DataFrame => nunique()
df.nunique()

# For Series => unique(), nunique(), value_counts()
df['종목명'].unique()

df['종목명'].nunique()

df['종목명'].value_counts()

df['종목명'].value_counts(normalize=True)

# value_counts() ignore np.nan
a = pd.DataFrame({'a':[np.nan, 1, 2]})['a']
print("a:\n", a)

# 위의 코드는  아래와 같음
# a = pd.Series([np.nan, 1, 2])

a.value_counts()

# example

# CSV 파일을 읽어와 데이터프레임으로 저장합니다.
a = pd.read_csv("미래에셋자산운용/sample_project/my_data/symbol_sector.csv", index_col=0)

a.head()

a.shape

a['Sector'].nunique()

a['Sector'].value_counts()

# 정렬

print_header("df.head()"); print(df.head())

# top n

df.nsmallest(5, "PER(배)")

# PER이 가장작은 100개중에서, 그 중에서 당기순이익이 가장 큰 5개 종목의 데이터
df.nsmallest(100, "PER(배)").nlargest(5, '당기순이익(억원)')

# Sort

df.sort_values("EPS(원)")

df.sort_values("EPS(원)", ascending=False).head()

df.sort_values(
    ['순이익률(%)', 'EPS(원)'],
    ascending=[True, False]
).head()

# Subset 추출하기

print_header("df.head()"); print(df.head())

df.shape

# By Columns

# string으로 인덱싱을 하면 -> Series로 반환한다
series = df['EPS(원)']
series

# list로 인덱싱을 하면 -> DataFrame로 반환한다
df2 = df[['EPS(원)', '종목명']]
df2

type(df['순이익률(%)'])                    # column명을 string으로 전달하면 -> Series 반환
type(df[['순이익률(%)', '당기순이익(억원)'] ]) # column명을 리스트로 전달하면 -> DAtaFrmae 반환

# [학습 포인트] `filter()`

df[['ROE(%)', 'ROA(%)', 'ROIC(%)']].head()

df.filter(like="RO").head()

df.filter(like="%").head()

df.filter(like="P").head()

df.filter(regex=r"P\w+R").head()

# By dtype

df.dtypes.value_counts()

df.select_dtypes(include=['float']).head()

df.select_dtypes(include=['object', 'string']).head()
# In Pandas 3.0+, use 'string' (or 'str') instead of 'object' for string columns.
# Currently, 'object' still includes strings for backward compatibility.
# df.select_dtypes(include=['str']).head()

# By Row

name_df = df.set_index("종목명")
name_df.head()

# ### iloc, loc

name_df.iloc[0]

name_df.iloc[[0, 3]]

name_df.loc["BYC"]

name_df.loc[['삼성전자', 'CJ']]

# ### Select rows by prefix

name_df.head()

# 반드시 index를 sort를 해야만 loc을 이용한 range indexing이 가능
# index가 sort된 새로운 dataframe을 return하는데 그것을 다시 name_df로 받음
name_df = name_df.sort_index()

name_df.index.is_monotonic_increasing

name_df.loc["삼성":"삼성전자"]

name_df.loc["가":"다"].head()

# ### More about and loc, iloc

name_df.loc["삼성전자"]

name_df['순이익률(%)']

# 위의 둘을 동시에 하는 방법
name_df.loc["삼성전자", "순이익률(%)"]

# 권장하지 않는 방법
# name_df.loc["삼성전자"]["순이익률(%)"]

name_df.loc[["삼성SDI", "삼성전자"], "순이익률(%)"]

name_df.loc[["삼성SDI", "삼성전자"], ["순이익률(%)", "EPS(원)"]]

# (index가 정렬이 된 경우만) -> range indexing(:)
name_df.loc["삼성":"삼성전자"]
name_df.loc["삼성":"삼성전자", :]
name_df.loc["삼성":"삼성전자", "순이익률(%)"]
name_df.loc["삼성":"삼성전자", ["순이익률(%)", "EPS(원)"]]

name_df.iloc[[0, 3], :]
name_df.iloc[[0, 3], [0,1]]

# df.iloc[[0, 3], "상장일"]   # error
# df.iloc[[0, 3], ["상장일", "종가"]]   # error

# ### iloc, loc's return type

# column indexing의 경우1. Series로 return
name_df['순이익률(%)'].head()

# column indexing의 경우2. DataFrame으로 return
name_df[['순이익률(%)', 'EPS(원)']].head()

# [학습 포인트] For Series data

a = pd.Series([1,2,3], index=['a', 'b', 'c'])
print("a:\n", a)

a.iloc[0]

a.loc['a']  # = a['a']와 결과가 같음

a.iloc[2]   # scalar
a.iloc[[2]]   # series

# [학습 포인트] For DataFrame Data -> column indexing에서와 동일한 원리

df.iloc[2]   # Series
df.iloc[[2]]   # DataFrame

# ### For Scalar Value

# [학습 포인트] use `.at` or `.iat`

df.loc[100, '순이익률(%)']

df.at[100, '순이익률(%)']

## Much faster if use `.iat` or `.at`
# => Table이 크면 클수록 더 차이가 많이 남
# get_ipython().run_line_magic('timeit', "df.loc[100, '순이익률(%)']")
# get_ipython().run_line_magic('timeit', "df.at[100, '순이익률(%)']")

# Also works with Series
# get_ipython().run_line_magic('timeit', "df['순이익률(%)'].iloc[100]")
# get_ipython().run_line_magic('timeit', "df['순이익률(%)'].iat[100]")

# Boolean selection

# ### Boolean Series

tmp_series = pd.Series({"a":1, "b":2})
tmp_series

tmp_series > 2

# ### Boolean DataFrame

tmp_df = pd.DataFrame({
    'a':[1,np.nan,3,4,np.nan],
    'b':[5, 3, 3, 4,np.nan]
})
tmp_df

tmp_df > 2

# ### DataFrame내에서 "Boolean Series" 만들기

print_header("df.head()"); print(df.head())

df['순이익률(%)']
df['영업이익률(%)']

# 이번에는 Series와 Series를 비교
# 그 결과 새로운 boolean series이 생기게 되어 변수에 저장할 수 있게 됩니다.
a = df['순이익률(%)'] > df['영업이익률(%)']
a.head()

a.sum()

a.mean()

# [학습 포인트] Boolean series로 indexing 하기

# row 갯수와, index 값이 서로 같음
df.shape
a.shape

# [학습 포인트] 대괄호를 이용하는 방법

df[a].head() # true인것들만 추출
df[a].shape

# [학습 포인트] loc를 이용하는 방법

df.loc[a]    # df[a]와 결과가 같음
df.loc[a].shape

df.loc[[0, 3]]

# df.loc[a]
# => df.loc[[9, 10, 13 ....]] 이렇게 내부적으로 변환이 된다고 생각하시면 됩니다. True에 해당하는 row의 index만 가져와서 indexing을 하는 것이죠

# shape의 변화
df.shape
df[df['순이익률(%)'] > df['영업이익률(%)']].shape
df.loc[df['순이익률(%)'] > df['영업이익률(%)']].shape

# ### Multiple boolean series

con1 = df['순이익률(%)'] > df['영업이익률(%)']
con2 = df['PBR(배)'] < 1

con1.head()
con2.head()

True and True

# if a > 1 and(or) b < 2:
#     print("!")

# [학습 포인트] and = &

# [학습 포인트] or = |

# [잘못된 방법] final_con = con1 and con2
# Pandas에서는 아래와 같이 조건식을 구성해야합니다.
final_con = con1 & con2

final_con.head()

df[final_con].head()
df[final_con].shape

df[final_con].head(2)
df.loc[final_con].head(2)
df.loc[final_con, ['ROE(%)']].head(2)

# [X] df.iloc[final_con].head(2)

# ### `isin()`

name_list = ['삼성전자', '현대건설', "삼성물산"]

# 1. multiple boolean series를 이용하는 방법

cond1 = df['종목명'] == "삼성전자"
cond2 = df['종목명'] == "현대건설"
cond3 = df['종목명'] == "삼성물산"

final_con = cond1 | cond2 | cond3
df[final_con]

# 2. index화 해서 loc으로 row-wise indexing을 이용해서 가져오는 방법

tmp_df = df.set_index('종목명')
tmp_df.head()

tmp_df.loc[['삼성전자', '현대건설', "삼성물산"]]

# 3. isin() 함수를 이용해서 가져오는 방법

cond = df['종목명'].isin(name_list)
df[cond]

df[df['종목명'].isin(name_list)].head(2)
df.loc[df['종목명'].isin(name_list)].head(2)
df.loc[df['종목명'].isin(name_list), ['종목명', 'ROA(%)', 'ROE(%)']].head(2)

# ### `all()` vs `any()`

a = df['순이익률(%)'] > 0

a.all()

a.any()

(df['순이익률(%)'] > 0).all()

(df['순이익률(%)'] > 0).any()

# 왜 결과가 False일지 생각해보세요
(df['순이익률(%)'] > -1000000).all()

# ### example

# CSV 파일을 읽어와 데이터프레임으로 저장합니다.
a = pd.read_csv("미래에셋자산운용/sample_project/my_data/symbol_sector.csv", index_col=0)
a.head()

a['Sector'].value_counts()

a['Sector'].value_counts().nlargest(5)

top_5_sector_list = a['Sector'].value_counts().nlargest(5).index
top_5_sector_list

a[a['Sector'].isin(top_5_sector_list)]

# 연산(Arithmetic)

import FinanceDataReader as fdr
price_df = fdr.DataReader("005930", '2009-09-16', '2018-03-21')
price_df.head()

# 연산 기준

# [학습 포인트] DataFrame은 기준이 columns

# [학습 포인트] Series는 기준이 index

# [학습 포인트] 따로 명시가 없다면 Series의 index가 DataFrame의 columns에 맞춰짐!

# DataFrame & Series

price_df.iloc[0]

# Subtract row Series
# DataFrame의 기준인 columns와 Series의 기준인 index가 서로 일치하기 때문에, 의도한 대로 계산 가능
(price_df - price_df.iloc[0]).head()

price_df['Open']

# Subtract column Series
# [X] DataFrame의 기준인 columns와 Series의 기준인 index가 서로 불일치 (price_df['open'] - price_df도 마찬가지로 [X] )
(price_df - price_df['Open']).head()

# DataFrame & DataFrame

# [학습 포인트] index,column 이 일치하는 것 끼리만 element-wise 연산이 이루어지고 나머지는 nan 처리

price_df[['Open', 'Low']].iloc[:2]

price_df - price_df[['Open', 'Low']].iloc[:2]

# 연산 관련 pandas built-in 함수

# [학습 포인트] axis란?

# [학습 포인트] 연산은 기본적으로 "axis를 변형(줄이거나 늘리는)하는 방식" 으로 진행된다.

#### numpy로 맛보기

# NumPy 라이브러리를 np라는 별칭으로 임포트합니다.
import numpy as np

# NumPy 배열(ndarray)을 생성합니다.
a = np.array([1,2,3])
# NumPy 배열(ndarray)을 생성합니다.
b = np.array([1,2,3])
print("a:\n", a)
print("b:\n", b)

a + b

np.sum(
    [a,b],
    axis=0
)

np.sum(
    [a,b],
    axis=1
)

# [      0  1  2
#    0  [1, 2, 2],
#    1  [1, 2, 2]
# ]
# 
# axis=0 --->       0            1
# axis=1 --->    0  1  2      0  1  2
#            [  [1, 2, 2],   [1, 2, 2] ]
# 
# axis=0 --->              0                        1
# axis=1 --->      0       1     2          0       1      2
# axis=2 --->     0  1    0, 1    0        0, 1    0, 1    0, 1
#            [  [[1, 2], [2, 4], [2]],   [[1, 2], [2, 2], [2, 2]] ]

# [학습 포인트] Note: shift+tab을 이용해서 주석 설명을 보는 것이 가장 정확함

df.head(2)

df[['순이익률(%)', 'PER(배)']].sum()  # default axis=0

df[['순이익률(%)', 'PER(배)']].mean()  # default axis=0

# DataFrame - Series의 형태
# df's columns: o,h,l,c
# series's index: o,h,l,c
# => mean()값을 통해 Normalizing
(price_df - price_df.mean()).head()

# [학습 포인트] 아래 구문은 연산 불가능했었음

close_series = price_df['Close']
price_df - close_series

# [학습 포인트] -> 하지만 DataFrame이 제공하는 연산관련 함수를 이용하면 가능!

price_df.head()

price_df.sub(close_series, axis=0).head()

# [학습 포인트] `sub()`의 경우 descrption에  'For Series input, axis to match Series index on'라고 써있음
#     - axis=0 or 1은 무조건 descrption (shift + tab) 먼저 보고 판단하고 그 후에 "axis는 해당 axis를 변형(줄이거나, 늘리는 것)" 적용하기

# [학습 포인트] “A simple brute force solution of trying both directions until achieving the desired result is one possibility”

price_df[['Open', 'Close']].sum(axis=1).head()
price_df[['Open', 'Close']].sum(axis=0).head()

# Example

# CSV 파일을 읽어와 데이터프레임으로 저장합니다.
b = pd.read_csv("미래에셋자산운용/sample_project/my_data/multi_price.csv", index_col=[0])

b.head()

b.loc["2018-08-09"]

momentum_series = b.loc["2018-08-09"] / b.loc["2017-08-09"] - 1

momentum_series.nlargest(3)

# Handling nan value

None == None

np.nan == np.nan

5 < np.nan

5 >= np.nan

5 == np.nan

# [학습 포인트] 아래 operation만 True를 return함

np.nan != 5

df1 = pd.DataFrame(
    {
        'a':[1,2,3],
        'b':[np.nan, 4, np.nan],
    }
)
df1

# nan이 있을 때의 boolean series

df1['b'] == df1['b']

df1.ge(2)   # Same with (df1 >= 2)
df1.le(2)

# [학습 포인트] 따라서 아래와같은 구문은 위험할 수 있음

print_header("df.head()"); print(df.head())

df.shape

df['PER(배)'].count()

df['PER(배)'] > 1

# [학습 포인트] 주의: 아래처럼 checking하면 `np.nan`의 갯수가 0개로 나옴

(df['PER(배)'] == np.nan).any()

# Nan checking

# ### For Series

df['순이익률(%)'].hasnans

# Generate boolean series
df['순이익률(%)'].isnull()

df['순이익률(%)'].isna().any()

df['순이익률(%)'].isna().sum()

# ### For DataFrame

df.isnull().head()

a = df.isnull()

a.any(axis=0)

df.isnull().any().any()

df.isnull().any().all()

# [학습 포인트] 참고: 두 개의 DataFrame이 같은지 판단하려면, `equals`를 사용하자

df1['b'] == df1['b']

df1['b'].equals(df1['b'])

# ### Example

_df = pd.DataFrame({'a':[1,np.nan,3], 'b':[np.nan, 2, 3]})
_df.head()

# [학습 포인트] 둘다 nan이 아닌 값들만 추출

_df['a'].notnull()
_df['b'].notnull()

# 1.
_df[ _df['a'].notnull() & _df['b'].notnull() ]

# 2.

_df.notnull().all(axis=1)

_df[_df.notnull().all(axis=1)]

# 3.
_df.dropna()

# subset에 있는 컬럼 중에 하나라도(혹은 전부, arg로 선택가능) null이면 drop한다
_df.dropna(subset=['a'])
