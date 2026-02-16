"""
[학습용] lec1_0_intro_to_numpy_ForOnlineLecture.py
이 스크립트는 실습 내용을 가독성 좋게 정리한 코드입니다.
"""

#!/usr/bin/env python
# coding: utf-8

# 수업에 사용될 라이브러리 설치

# get_ipython().system('ls   # 윈도우 사용자# 의 경우 -> dir')

# pip install numpy

# get_ipython().system('pip install -r requirements.txt')

# Numpy 소개

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
def print_header(title):
    print(f"\n{'='*20} {title} {'='*20}")


# NumPy 라이브러리를 np라는 별칭으로 임포트합니다.
import numpy as np

# ndarray dtype

# NumPy 배열(ndarray)을 생성합니다.
a = np.array([1, 2, 3])
print("a:\n", a)

type(a)

# indexing이 list와 방식이 같음
a[0]

# a.

# [학습 포인트] append() 같은 함수가 없음 -> 즉, python의 list와는 다른 data type임

# Universal function

a = [1, 2, 3]
b = [4, 5, 6]

new_list = []
for e1, e2 in zip(a, b):
    new_list.append(e1 + e2)
new_list

a + b

# NumPy 배열(ndarray)을 생성합니다.
a = np.array([1, 2, 3])
# NumPy 배열(ndarray)을 생성합니다.
b = np.array([4, 5, 6])

# 길이가 길어도 근사적으로 1초가 걸린다
a + b

# np.sum
# np.abs
# np.log
# np.exp
# np.isnan
# ...

# [학습 포인트] n차원 array

# NumPy 배열(ndarray)을 생성합니다.
a = np.array([  [1, 2],  [3, 4]   ])
print("a:\n", a)

# readability 높이기
# NumPy 배열(ndarray)을 생성합니다.
a = np.array(
    [
        [1, 2],
        [3, 4],
    ]
)

a[0][1]

a + a
