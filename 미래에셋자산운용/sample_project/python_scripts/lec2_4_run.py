import pandas as pd
import numpy as np
import FinanceDataReader as fdr

# Set options
pd.set_option('display.float_format', lambda x: '%.3f' % x)

print("--- 1. Simple Moving Average (SMA) ---")
try:
    df = fdr.DataReader("005930", '2020-01-01', '2020-01-31')[['Close']]
    df.columns = ["삼성전자"]
    # 5-day SMA
    df['5SMA'] = df['삼성전자'].rolling(window=5).mean()
    # SMA with min_periods
    df['5SMA_min1'] = df['삼성전자'].rolling(window=5, min_periods=1).mean()
    print(df.head(7))
except Exception as e:
    print("SMA Error:", e)

print("\n--- 2. Bollinger Bands ---")
# Upper/Lower band using 20-day SMA and 2*std
df_bb = df.copy()
df_bb['20SMA'] = df_bb['삼성전자'].rolling(window=20, min_periods=1).mean()
df_bb['std'] = df_bb['삼성전자'].rolling(window=20, min_periods=1).std()
df_bb['Upper'] = df_bb['20SMA'] + 2 * df_bb['std']
df_bb['Lower'] = df_bb['20SMA'] - 2 * df_bb['std']
print(df_bb[['삼성전자', 'Upper', 'Lower']].tail(3))

print("\n--- 3. Rolling Correlation ---")
try:
    df1 = fdr.DataReader("005930", '2020-01-01', '2020-06-30')[['Close']] # Samsung
    df2 = fdr.DataReader("069500", '2020-01-01', '2020-06-30')[['Close']] # KODEX 200
    df_corr = pd.concat([df1, df2], axis=1)
    df_corr.columns = ["Samsung", "KODEX200"]
    # 20-day rolling correlation
    rolling_corr = df_corr['Samsung'].rolling(20).corr(df_corr['KODEX200'])
    print("Rolling Corr head:\n", rolling_corr.dropna().head(3))
except Exception as e:
    print("Corr Error:", e)

print("\n--- 4. Exponentially-weighted Moving Average (EWMA) ---")
# 최근 데이터에 더 높은 가중치
df['EWMA_0.2'] = df['삼성전자'].ewm(alpha=0.2).mean()
print(df[['삼성전자', 'EWMA_0.2']].head(3))
