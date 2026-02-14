"""
[학습용] lec1_1_Basics_of_DataFrame_and_Series_ForOnlineLecture.py
이 스크립트는 실습 내용을 가독성 좋게 정리한 코드입니다.
"""

#!/usr/bin/env python
# coding: utf-8

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# **Pandas version 0.25.1 (`pip install pandas==0.25.1`)**

# `Series` Data type

# [학습 포인트] Numpy's ndarray + 숫자가 아닌 다른 type의 index (E.g. 문자열)

# Pandas 라이브러리를 pd라는 별칭으로 임포트합니다.
import pandas as pd

a = pd.Series([1,2,3,4])
a

# 첫번째 방법
s2 = pd.Series(
    [1, 2, 3, 4],
    index=['a', 'b', 'c', 'd']
)
s2

s2.head(2)

# 두번째방법
s2 = pd.Series({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})
s2.head()

# [학습 포인트] 한가지 data type만 가지고 있을 수 있음

# `nan`과 관련된 함수

# NumPy 라이브러리를 np라는 별칭으로 임포트합니다.
import numpy as np

np.nan

s = pd.Series([10, 0, 1, 1, 2, 3, 4, 5, 6, np.nan])
s

len(s)

s.shape

s.count()    # not count `nan`

s.unique()

# 수업에서는 다루지 않았지만, nunique()는 unique한 값들의 총 갯수를 알려주는 함수입니다.
# s.nunique()

s.value_counts()

# [학습 포인트] 이 외의 함수들에 대해서는 이후 수업에서 하나씩 다룰 예정!

# index label을 기준으로 Series간에 operation이 일어남

# [학습 포인트] Data의 '순서'가 아니라 index label이 자동으로 정렬되어 연산이 진행됨!

s3 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
s4 = pd.Series([4, 3, 2, 1], index=['d', 'c', 'b', 'a'])

s3 + s4

# `DataFrame` Data type

# [학습 포인트] 다수의 Series를 하나의 변수로 관리할 수 있도록 만든 자료형
#     - Series의 dict 형태라고 보면됨
#         - `{'컬럼명1': Series1, '컬럼명2': Series2}`
#         - 각 Series는 DataFrame의 column을 이룸
#         - 당연히 DataFrame을 이루는 Series간의 index는 서로 다 같음! => 동일 index 사용

# DataFrame을 만드는 다양한 방법들

s1 = np.arange(1, 6, 1)
s2 = np.arange(6, 11, 1)
s1
s2

df = pd.DataFrame(
    {
        'c1': s1,
        'c2': s2
    }
)
df

# 1번째 방법  (Default index and columns would be set)
pd.DataFrame(
    [
        [10,11],
        [10,12]
    ]
)
pd.DataFrame(
    np.array(
        [
            [10, 11],
            [20, 21]
        ]
    )
)

# 2번째 방법 (많이 안쓰임)
pd.DataFrame(
    [
        pd.Series(np.arange(10, 15)),   # 굳이 Series가 아니고 list형태이기만 하면 됨(=iterable한 object면 다 가능)
        pd.Series(np.arange(15, 20)),   # 굳이 Series가 아니고 list형태이기만 하면 됨(=iterable한 object면 다 가능)
    ]
)

pd.DataFrame(
    [
        np.arange(10, 15),
        np.arange(15, 20),
    ]
)

# 3번째 방법 (with column & index names)
pd.DataFrame(
    np.array(
        [
            [10, 11],
            [20, 21]
        ]
    ),
    columns=['a', 'b'],
    index=['r1', 'r2']
)

# 4번째 방법
s1 = pd.Series(np.arange(1, 6, 1))    # 굳이 Series가 아니고 list형태이기만 하면 됨(=iterable한 object면 다 가능)
s2 = pd.Series(np.arange(6, 11, 1))   # 굳이 Series가 아니고 list형태이기만 하면 됨(=iterable한 object면 다 가능)
pd.DataFrame(
    {
        'c1': [1,2,3],    # list, np.array, Series 전부 다 올 수 있음!
        'c2': [4,5,6]
    }
)

# 참고: 1줄짜리 만들 때도 dictionary의 value에 해당하는 값들은 iterable한 data type(e.g. list, np.array, Series 등)으로 설정해줘야함
pd.DataFrame({'c1': [0], 'c2': [1]})

s1 = pd.Series(np.arange(1, 6, 1), index=['a', 'b', 'c', 'd', 'e'])
s2 = pd.Series(np.arange(6, 11, 1), index=['b', 'c', 'd', 'f', 'g'])
df = pd.DataFrame(
    {
        'c1': s1,
        'c2': s2
    }
)
df

# DataFrame 생성시, Series간에 Index 기준으로 자동정렬!

s1 = pd.Series(np.arange(1, 6, 1))
s2 = pd.Series(np.arange(6, 11, 1))
s3 = pd.Series(np.arange(12, 15), index=[1, 2, 10])  # this one has index values unlike s1, s2
s1

s2

s3

df = pd.DataFrame({'c1': s1, 'c2': s2, 'c3': s3})
df

# DataFrame에 새로운 column 추가하기

my_dict['a'] = 1

df['c4'] = pd.Series([1,2,3,4], index=[0, 1, 2, 10])

df

# Reindexing

# [학습 포인트] 새로운 index label을 기반으로 기존의 "index-value" mapping은 유지한채 재배열하는 것

# ### 참고: index 자체를 바꾸는 것("index-value" mapping이 깨짐)

s = pd.Series([1,2,3,4,5])
s

s.index = ['a', 'b', 'c', 'd', 'e']
s

# ### 참고 :  `set_index()` : 특정 column을 index로 만듦

# 위의 'DataFrame 생성시, Series간에 Index 기준으로 자동정렬!' 챕터에서 정의한 dataframe입니다
df

df['c5'] = pd.Series([1,2,3,4,5,6], index=[0,1,2,3,4,10])
df

df.set_index("c5")

# ### Reindex

s2 = s.reindex(
    ['a', 'c', 'e', 'g']
)
s2

# Copied
s2['a'] = 0
s2

# s는 s2의 값을 바꿔도 안 건드려짐
s

# [X] 이렇게 하면 안됨
s1 = pd.Series([0, 1, 2], index=[0, 1, 2])
s2 = pd.Series([3, 4, 5], index=['0', '1', '2'])
s1
s2

s1 + s2

s1.index

s2 = s2.reindex(s1.index)
s2

# 첫번째 방법
s1 = pd.Series([0, 1, 2], index=[0, 1, 2])
s2 = pd.Series([3, 4, 5], index=['0', '1', '2'])

s2.index = s2.index.astype(int)

s2

s2.index

s1 + s2

# 두번째 방법
s1 = pd.Series([0, 1, 2], index=[0, 1, 2])
s2 = pd.Series([3, 4, 5], index=['0', '1', '2'])

s1.index = ['a', 'b', 'c']
s2.index = ['a', 'b', 'c']

s1 + s2

# #### `reindex()`의 유용한 Arguments

# [학습 포인트] `fill_value`

s2 = s.copy()
s2

s2.reindex(['a', 'f'])

s2.reindex(['a', 'f'], fill_value=0)  # fill 0 insteand of Nan

# [학습 포인트] `method`

s3 = pd.Series(['red', 'green', 'blue'], index=[0, 3, 5])
s3

s3.reindex(np.arange(0,7))

s3.reindex(np.arange(0,7), method='ffill')

# #### 예제

# 맨 첫 강의에서 라이브러리를 설치할 때 requirements.txt를 이용해서 설치를 했으면, 건너뛰셔도 됩니다.
# get_ipython().system('pip install finance_datareader == 0.9.1')

# NumPy 라이브러리를 np라는 별칭으로 임포트합니다.
import numpy as np
# Pandas 라이브러리를 pd라는 별칭으로 임포트합니다.
import pandas as pd

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Pandas DataFrame의 사이즈가 큰 경우, 어떻게 화면에 출력을 할지를 세팅하는 코드
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
pd.set_option("display.max_columns", None)

import FinanceDataReader as fdr

# 삼성전자
df1 = fdr.DataReader("005930", '2025-01-02', '2026-10-30')

# KODEX 200 (ETF)
df2 = fdr.DataReader("069500", '2025-01-03', '2026-10-30')

df1.head(2)
df1.tail(2)

df2.head(2)
df2.tail(2)

# 삼성전자
df1 = fdr.DataReader("005930", '2025-01-02', '2026-10-30')

# KODEX 200 (ETF)
df2 = fdr.DataReader("069500", '2025-01-02', '2026-10-30')

df1.shape
df2.shape

df2 = df2.drop(pd.to_datetime("2025-01-03"))
df2.head()

df1.head()

new_df2 = df2.reindex(df1.index)
new_df2.head()

df1.shape
new_df2.shape

new_df2.ffill()
