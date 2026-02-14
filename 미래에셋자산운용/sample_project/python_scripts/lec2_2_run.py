import pandas as pd
import numpy as np
import FinanceDataReader as fdr

# Set options
pd.set_option('display.float_format', lambda x: '%.4f' % x)

print("--- 1. Simple Return vs Log Return ---")
df = pd.DataFrame({"price":[1, 1.02, 1.01, 1.05]})

# Simple Return: (p_n / p_{n-1}) - 1
simple_rtn = df['price'].pct_change().fillna(0)
print("Simple Returns:\n", simple_rtn.tolist())

# Log Return: log(p_n / p_{n-1})
log_rtn = np.log(df['price'] / df['price'].shift(1)).fillna(0)
print("Log Returns:\n", log_rtn.tolist())

print("\n--- 2. Cumulative Returns ---")
# From Simple Return: (1 + r).cumprod()
cum_rtn_simple = (1 + simple_rtn).cumprod()
print("Cumulative (Simple):\n", cum_rtn_simple.tolist())

# From Log Return: exp(sum(r))
cum_rtn_log = np.exp(log_rtn.cumsum())
print("Cumulative (Log):\n", cum_rtn_log.tolist())

print("\n--- 3. Shift and pct_change ---")
try:
    df_fin = fdr.DataReader("005930", '2018-01-02', '2018-01-05')[['Close']]
    print("Close Price:\n", df_fin)
    print("Shifted (1):\n", df_fin.shift(1))
    print("Pct Change:\n", df_fin.pct_change())
except Exception as e:
    print("FDR Error:", e)
