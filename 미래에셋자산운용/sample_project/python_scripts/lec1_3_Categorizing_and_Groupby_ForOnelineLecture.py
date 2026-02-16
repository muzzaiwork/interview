"""
[학습용] lec1_3_Categorizing_and_Groupby_ForOnelineLecture.py
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

# 데이터 준비

# CSV 파일을 읽어와 데이터프레임으로 저장합니다.
df = pd.read_csv("미래에셋자산운용/sample_project/my_data/naver_finance/2016_12.csv")

print_header("df.head()"); print(df.head())

# 수익률 구하기 (16.12 ~ 17.12)

df['rtn'] = df['price2'] / df['price'] - 1
print_header("df"); print(df)

# PER 값에 따라 group number 부여하기

# 값을 기준으로 grouping 하기 (DIFFERENT number of members in each  group)

# ### boolean selection & loc 사용

# [학습 포인트] 곧 뒤에서 배울 `cut()` 을 사용하면 아래 방법보다 더 쉽게 가능합니다. 하지만 여기서 진행하는 방식들도 매우 중요하니 반드시 익혀두세요!

(df['PER(배)'] >= 10).head()

bound1 = df['PER(배)'] >= 10
bound2 = (5 <= df['PER(배)']) & (df['PER(배)'] < 10)
bound3 = (0 <= df['PER(배)']) & (df['PER(배)'] < 5)
bound4 = df['PER(배)'] < 0

df.shape

df[bound1].shape # = df.loc[bound1].shape

df.loc[bound1, 'PER_Score'] = 1
df.loc[bound2, 'PER_Score'] = 2
df.loc[bound3, 'PER_Score'] = 3
df.loc[bound4, 'PER_Score'] = -1

df['PER_Score'].head()

df['PER_Score'].nunique()

df['PER_Score'].value_counts()

# [학습 포인트] `PER_Score`가 float number로 나오는 이유?

df['PER_Score'].hasnans

df['PER_Score'].isna().sum()

df['PER(배)'].isna().sum()

df[df['PER(배)'].isna()]

df.loc[df['PER_Score'].isna(), "PER_Score"] = 0

# 아래와 같은 방식으로도 가능
# df['PER_Score'] = df['PER_Score'].fillna(0)
# df.loc[:, 'PER_Score'] = df['PER_Score'].fillna(0)

# ### boolean series 의 연산 특성 사용

df.loc[:, "PER_Score1"] = (bound1 * 1)  + (bound2 * 2) + (bound3 * 3) + (bound4 * -1)

df['PER_Score1'].head()

df['PER_Score1'].value_counts()

df['PER_Score'].value_counts()

# ### 위의 두 score series는 서로 같을까?

df['PER_Score'].equals(df['PER_Score1'])

df['PER_Score'].dtypes
df['PER_Score1'].dtypes

df['PER_Score'].astype(int).equals(df['PER_Score1'])

# ### `cut()`

per_cuts = pd.cut(
    df['PER(배)'],
    [-np.inf, 0, 5, 10, np.inf],
)

per_cuts.head()

per_cuts.iloc[0]

per_cuts.value_counts()

per_cuts.isna().sum()

# [학습 포인트] cut()과 동시에 label 달아주기

bins = [-np.inf, 10, 20, np.inf]
labels = ['저평가주', '보통주', '고평가주']
per_cuts2 = pd.cut(
    df['PER(배)'],
    bins=bins,
    labels=labels
)
per_cuts2.head()

# df.loc[:, 'PER_score2'] = per_cuts  # or per_cuts2
# df['PER_score2'] = per_cuts         # or per_cuts2

# Group내 데이터 갯수를 기준으로 grouping 하기 (SAME number of members in each  group)

# ### `qcut()`

pd.qcut(df['PER(배)'], 3, labels=[1,2,3]).head()

df.loc[:, 'PER_Score2'] = pd.qcut(df['PER(배)'], 10, labels=range(1, 11))
print_header("df.head()"); print(df.head())

df['PER_Score2'].value_counts()

df['PER_Score2'].hasnans

df['PER_Score2'].isna().sum()

df['PER_Score2'].dtype

# [학습 포인트] 'category' type: A string variable consisting of only a few different values

# DataFrame에서 category dtype인 columns들 추출하기
# df.select_dtypes(include=['category']).columns

df['PER_Score2'].head()

df['PER_Score2'].value_counts()

df = df.dropna(subset=['PER(배)'])

df['PER_Score2'].isna().sum()

# Split - Apply - Combine

# CSV 파일을 읽어와 데이터프레임으로 저장합니다.
df = pd.read_csv("미래에셋자산운용/sample_project/my_data/naver_finance/2016_12.csv")
df.shape

df = df.dropna()
df.shape

g_df = df.copy()
g_df.head()

# Group score 생성

g_df['rtn'] = g_df['price2'] / g_df['price'] - 1

g_df.loc[:, 'PER_score'] = pd.qcut(g_df['PER(배)'], 10, labels=range(1, 11))
g_df.loc[:, 'PBR_score'] = pd.qcut(g_df['PBR(배)'], 10, labels=range(1, 11))

g_df.set_index('ticker', inplace=True)

g_df.head()

g_df.dtypes.value_counts()

# groupby() & aggregation

# [학습 포인트] `groupby()`
#     - 실제로 grouping까지는 하지 않고, grouping이 가능한지 validation만 진행(preparation)

# [학습 포인트] Aggregation
#     - 2가지 요소로 구성
#         - aggregating columns
#         - aggregating functions
#             - e.g. `sum, min, max, mean, count, variacne, std` etc

# [학습 포인트] 결국, 3가지 요소만 충족시키면 됨!
#     - Grouping columns (cateogorial data type)
#     - Aggregating columns
#     - Aggregating functions

# ### `groupby` object 살펴보기

# 특정 열을 기준으로 데이터를 그룹화합니다.
g_df.groupby('PER_score')

# 특정 열을 기준으로 데이터를 그룹화합니다.
g_df_obj = g_df.groupby(["PBR_score", "PER_score"])
g_df_obj

type(g_df_obj)

g_df_obj.ngroups

g_df['PBR_score'].nunique()
g_df['PER_score'].nunique()

# [학습 포인트] "ngroups와 (g_df['PBR_score'].nunique() x g_df['PER_score'].nunique())가 차이가 나는 이유"에 대해서 생각해보기

type(g_df_obj.size())

g_df_obj.size().head()

# Multi-level index를 가진 Series indexing하는 법
g_df_obj.size().loc[1]
g_df_obj.size().loc[(1, 1)]

# Series -> DataFrame으로 변환
g_df_obj.size().to_frame().head()

type(g_df_obj.groups)
g_df_obj.groups.keys()
g_df_obj.groups.values ()

# Retrieve specific group
g_df_obj.get_group((1, 1))

# [학습 포인트] For loop을 이용해서 grouping된 object 확인해보기 (많이는 안쓰임)

for name, group in g_df_obj:
    print(name)
    group.head(2)
    break

# 참고 :groupby()에 대해 head()를 적용하면, 기존이 head()가 작동하는 방식, 즉, 최상위 2개를 가지고 오는게 아니라
# 각 그룹별 최상위 2개를 무작위로 섞어서 하나로 합친 DataFrame을 리턴함
# 특정 열을 기준으로 데이터를 그룹화합니다.
g_df.groupby('PER_score').head(2)

# ### aggreggation

# [학습 포인트] 반드시 "aggregating" 기능이 있는 function 을 써야함
#     - min, max, mean, median, sum, var, size, nunique, idxmax

# 특정 열을 기준으로 데이터를 그룹화합니다.
g_df.groupby("PBR_score").agg(
    {
        "rtn": "mean", # =  np.mean
    }
)

# 특정 열을 기준으로 데이터를 그룹화합니다.
pbr_rtn_df = g_df.groupby("PBR_score").agg({'rtn': 'mean'})
# 특정 열을 기준으로 데이터를 그룹화합니다.
per_rtn_df = g_df.groupby("PER_score").agg({'rtn': 'mean'})

pbr_rtn_df.head()

# 다양한 방법으로 진행하기 (같은 결과)
# 특정 열을 기준으로 데이터를 그룹화합니다.
g_df.groupby("PER_score")['rtn'].agg('mean').head()
# 특정 열을 기준으로 데이터를 그룹화합니다.
g_df.groupby("PER_score")['rtn'].agg(np.mean).head()
# 특정 열을 기준으로 데이터를 그룹화합니다.
g_df.groupby("PER_score")['rtn'].mean().head()

# return type이 다를 수 있음에 주의
# 특정 열을 기준으로 데이터를 그룹화합니다.
g_df.groupby("PER_score")['rtn'].agg("mean").head(2)   # Series로 return
# 특정 열을 기준으로 데이터를 그룹화합니다.
g_df.groupby("PER_score")[['rtn']].agg("mean").head(2)  # DataFrame으로 return

# 2개 이상의 컬럼에 대해 aggregation
# 특정 열을 기준으로 데이터를 그룹화합니다.
g_df.groupby("PER_score")[['rtn', 'PBR(배)']].agg("mean").head(2)

# 2개 이상의 aggregation
# 특정 열을 기준으로 데이터를 그룹화합니다.
g_df.groupby("PER_score")[['rtn', 'PBR(배)']].agg(["mean", "std"]).head(2)

# 2개 이상의 컬럼 & 각각에 대해 다른 aggregation
# 특정 열을 기준으로 데이터를 그룹화합니다.
g_df.groupby("PBR_score").agg(
    {
        'rtn': ['mean', 'std'],
        'PER(배)': ['min']

    }
)

# [학습 포인트] aggregation function이 아닌경우 => `agg()`가 error를 발생시킴

# sqrt는 aggregation 방식의 연산이 아님!
print("np.sqrt([1, 2, 3, 4]):", np.sqrt([1, 2, 3, 4]))

# 특정 열을 기준으로 데이터를 그룹화합니다.
# g_df.groupby("PER_score")['rtn'].agg(np.sqrt) # 주석 처리: 최신 pandas에서는 aggregation이 아닌 함수 사용 시 에러 발생

# [학습 포인트] Visualization(시각화) 맛보기

# get_ipython().run_line_magic('matplotlib', 'inline')

# 데이터를 시각화합니다.
pbr_rtn_df.plot(kind='bar')

# 데이터를 시각화합니다.
pbr_rtn_df.plot(kind='bar');
# 데이터를 시각화합니다.
per_rtn_df.plot(kind='bar');

# ### Examples

# 특정 열을 기준으로 데이터를 그룹화합니다.
g_df1 = g_df.groupby(["PBR_score", "PER_score"])\
            .agg(
                {
                    'rtn': ['mean', 'std', 'min', 'max'],
                    'ROE(%)': [np.mean, 'size', 'nunique', 'idxmax']
                 }
            )
g_df1.head()

# 특정 열을 기준으로 데이터를 그룹화합니다.
a = g_df.groupby(["PBR_score", "PER_score"])[['rtn', 'ROE(%)']].agg(['sum', 'mean'])

# Multi-index라고 해서 쫄 것 없음!
a.loc[1]
a.loc[(1, 3)]
a.loc[[(1, 3), (1, 4 )]]

# ### 주의: nan은 groupby시 자동으로 filter out 되기 때문에, 미리 전처리 다 하는게 좋음

df = pd.DataFrame({
    'a':['소형주', np.nan, '대형주', '대형주'],
    'b':[np.nan, 2,         3,     np.nan],
})
print_header("df"); print(df)

# 특정 열을 기준으로 데이터를 그룹화합니다.
df.groupby(['a'])['b'].mean()

# ###  `as_index = False` : group cols들이 index가 아니라 하나의 col이 됨 (aggregate하고 reset_index()를 취한 것)

# 특정 열을 기준으로 데이터를 그룹화합니다.
a = g_df.groupby(["PER_score"]                ).agg({'rtn': ['mean', 'std']}).head(2)
# 특정 열을 기준으로 데이터를 그룹화합니다.
b = g_df.groupby(["PER_score"], as_index=False).agg({'rtn': ['mean', 'std']}).head(2)

print("a:\n", a)
print("b:\n", b)

a.index
a.columns

b.index
b.columns

a['rtn']

a[('rtn', 'mean')].head()

# ### Multi-index columns을 하나로 병합하기

g_df1.head()

level0 = g_df1.columns.get_level_values(0)
level1 = g_df1.columns.get_level_values(1)

level0
level1

g_df1.columns = level0 + '_' + level1

g_df1.head(2)

g_df1 = g_df1.reset_index()
g_df1.head()

# 실전예제: 시가총액으로 Small and Big 나누기

# CSV 파일을 읽어와 데이터프레임으로 저장합니다.
a_df = pd.read_csv("미래에셋자산운용/sample_project/my_data/Small_and_Big.csv", index_col=[0])
a_df.head()

a_df.tail()

# 특정 열을 기준으로 데이터를 그룹화합니다.
median_df = a_df.groupby(['date']).agg({'시가총액 (보통)(평균)(원)': 'median'})
median_df.head()

median_df.columns = ['시가총액_median']
median_df.head()

# [학습 포인트] 구한 median dataframe을 어떻게 가존의 원본 dataframe과 연결 시킬수있을까?
# => 다음 노트북!
