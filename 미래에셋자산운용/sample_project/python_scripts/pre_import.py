"""
[학습용] pre_import.py
이 스크립트는 실습 내용을 가독성 좋게 정리한 코드입니다.
"""

#!/usr/bin/env python
# coding: utf-8

# 앞으로의 수업시작 전, 가장 처음에 실행되었다고 가정하는 코드

# NumPy 라이브러리를 np라는 별칭으로 임포트합니다.
import numpy as np
# Pandas 라이브러리를 pd라는 별칭으로 임포트합니다.
import pandas as pd

# 하나의 cell에서 multiple output을 출력을 가능하게 하는 코드
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

a = 1
a

a = 1
b = 2

a
b

# Pandas DataFrame의 사이즈가 큰 경우, 어떻게 화면에 출력을 할지를 세팅하는 코드
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
pd.set_option("display.max_columns", None)

pd.DataFrame(np.random.randn(1000, 300))

# https://www.dropbox.com/s/32mfc9xwcg71mkg/pre_import.py
