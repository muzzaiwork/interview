"""
[학습용] lec2_3_performance_indicator_ForOnlineLecture.py
이 스크립트는 실습 내용을 가독성 좋게 정리한 코드입니다.
"""

#!/usr/bin/env python
# coding: utf-8

# **수업을 수강하시기 전, lec2_1.ipynb의 "수강 전 필독"을 반드시 확인해주세요**

# Annualized(년 단위) returns & std

# [학습 포인트] Annualize를 하는 이유?

# [학습 포인트] **N day total return이 주어지거나 average daily return이 주어진 경우, K day expected return 구하기**

# Using Simple return

# [학습 포인트] 참고: https://financetrain.com/how-to-calculate-annualized-returns/

# [학습 포인트] Reminder
#     - $ p_1 (1 + x)^n = p_{n+1} $
#     - $ (1 + x)^n = {p_{n+1} \over p_1}$
#     - $ 1 + x = {p_{n+1} \over p_1}^{({1 \over n})}$
#     - $ x = {p_{n+1} \over p_1}^{({1 \over n})} - 1$

# [학습 포인트] Example 4: Daily Returns(n-day **total**  return 구하기)
#     Let’s say we have 0.1% daily returns. Since there are 365 days in a year, the annual returns will be:
#     ```
#     Annual returns = (1+0.001)^365 – 1 = 44.02%
#     ```

# [학습 포인트] Example 5: 100 Days Returns (n-day **average** return 구하기 & **total** return 구하기)
#     Let’s say we have 6% returns over 100 days. The annual returns will be:
#     ```
#     Annual returns = (1+0.06)^(365/100) – 1 = 23.69%
#     ```

# [학습 포인트] 결국, 위에서 구한 $x(={p_{n+1} \over p_1}^{({1 \over n})} - 1)$ 에 +1 해주고 K days 만큼 제곱한 후 다시 -1을 해서 K days expected return을 구함
#     - 0.001 * 365(or 252)과 같은 **산술연산**으로는 올바른 결과 도출 불가능

# Using Log return

# [학습 포인트] Reminder
#     - "Log return을 사용하면, '산술평균'으로 n day 수익(or 평균수익)을 계산할 수 있다보니, 통계적 특징들을 잘 이용할 수 있음"
#     - n-day average log return
#         1. n-day **total** log return 구하기
#             - $ log({p_2 \over p_1}) + log({p_3 \over p_2}) + ... + log({p_{n+1} \over p_{n}})  = log({p_{n + 1} \over p_{1}})$
#         2. n-day **average** log return 구하기(On average (= $E(X)$))
#             - $ {1 \over n}[log({p_2 \over p_1}) + log({p_3 \over p_2}) + ... + log({p_{n+1} \over p_{n}})] = {1 \over n}log({p_{n + 1} \over p_{1}})$  (log return 자체를 또 다른 하나의 개념으로 받아들이기)
#             - 참고: 이를 이용해서 실제 real return을 구하고 싶다면
#                 - (exponential) $e^{{1 \over n}log({p_{n + 1} \over p_{1}})} = ({p_{n + 1} \over p_{1}})^{1 \over n}$
#                 - (take -1) $({p_{n + 1} \over p_{1}})^{1 \over n} - 1$

# ### 확률/통계적 접근

# [학습 포인트] 선형변환 vs 독립시행
#     - $ X_n$ ~ $log({{p_n} \over {p_{n-1}}})$
#         - $ X_1 = log({p_1 \over p_0}) $
#         - $ X_2 = log({p_2 \over p_1}) $
#         - ...
#         - Independent Assumption
#     - "$ Y=250X $" vs "$ Y=X_1 + X_2 + ... + X_{250}$"
#         - $E(250X)$ vs $E(X_1 + X_2 + ... + X_{250})$

# [학습 포인트] $ E(X+Y) = E(X) + E(Y) $
#     - (랜덤)변수들간의 독립성 과는 상관없이 무조건 성립
#     - $E(X_1 + X_2 + .. + X_n) = E(X_1) + E(X_2) +  ... + E(X_n) = nE(X)$

# [학습 포인트] $ Var(X+Y) = ? $
#     - $Var(X)$
#         - $ E[(X - \mu_x)^2] = E[(X - \mu_x)(X - \mu_x)]$
#             - $= E[X^2 + -2X{\mu}_x + {{\mu}_x}^2]$
#             - $= E[X^2] - 2{\mu}_xE[X] + {{\mu}_x}^2$
#             - $= E[X^2] - 2{{\mu}_x}^2 + {{\mu}_x}^2$
#             - $= E[X^2] - {{\mu}_x}^2$
#     - $Var(X+Y)$
#         - if X, Y are not indenpendent
#             - $ E[[(X + Y) - (\mu_x + \mu_y)]^2] $
#                 - $ = E[[(X - \mu_x) + (Y - \mu_y)]^2]$
#                 - $ = E[(X - \mu_x)^2 + (Y - \mu_y)^2 + 2(X - \mu_x)(Y - \mu_y)]$
#                 - $ = E[(X - \mu_x)^2] + E[(Y - \mu_y)^2] + 2E[(X - \mu_x)(Y - \mu_y)]$
#                 - $ = Var(X) + Var(Y) + 2Cov(X, Y)$
#                     - https://ko.wikipedia.org/wiki/공분산
#         - if X, Y are indenpendent
#             - $ = Var(X) + Var(Y) $
#             - $Var(X_1 + X_2 + .. + X_n) = Var(X_1) + Var(X_2) +  ... + Var(X_n)$

# [학습 포인트] 따라서,
#     - $ E(X_1 + X_2 + .. + X_{252}) = E(X_1) + E(X_2) + .. + E(X_{252}) = 252 * E(X)$
#         - 결국 annulaized expected return = 250*(일평균 log수익률) = 250\*E(X)
#     - $ Var(X_1 + X_2 + .. + X_{252}) = Var(X_1) + Var(X_2) + .. + Var(X_{252}) = 252 * Var(X)$
#         - 결국 annulaized variance of return = 250*(일단위 log수익률 variance) = $250*Var(X)$
#         - $ Std(X_1 + X_2 + .. + X_{252}) = \sqrt{Var(X_1 + X_2 + .. + X_{252})} = \sqrt{252 * Var(X)} = \sqrt{252} * Std(X)$
#             - 참고: Annualized Std 자체를 지표로 많이 보기도 함
#     - 결국, log return의 특징으로 인해, sharpe ratio에 쓰기 부합
#         - 왜냐하면 Sharpe Ratio = annualized E(X) / annualized std(X) = $\sqrt250 \times {E(X) \over std(X)}$

# [학습 포인트] 참고: 나중엔 이런 것도 배움
#     - A, B, C 자산이 있을 때, B와 C가 same variance of return을 가지고 있더라도,  (A, B)로 구성된 포트폴리오와 (A,C)로 구성된 포트폴리오의 risk는 서로 다름! (due to the correlation effect)
#     - $Var(w_aA + w_bB + w_cC + ...)$
#     - Risk of portfolio (Covariance Matrix)

import FinanceDataReader as fdr

df = fdr.DataReader("069500", '2019-01-02', '2020-10-30')

log_rtn_df = np.log(df.pct_change() + 1).fillna(0)
log_rtn_df = log_rtn_df[['Close']]
log_rtn_df.columns = ["KODEX200"]
log_rtn_df.head()

log_rtn_df.mean() * 252

# [학습 포인트] 참고
#     - `mean(log_rtn)` $ \Rightarrow {1 \over n} ({log{p_2 \over p_1} + log{p_3 \over p_2} +  ... + log{p_{n+1} \over p_{n}}} ) = {1 \over n} ({log{p_{n+1} \over p_1}})$
#     - $ exp() \Rightarrow ({p_{n+1} \over p_1})^{1 \over n}$
#     - ${({p_{n+1} \over p_1})^{{1 \over n} \times {K \text{ days}}}}$
#     - 이 방법은 밑에서 배울 CAGR을 mean log return으로부터 구하는 방법이기도 합니다 :)

log_rtn_df.std() * np.sqrt(252)

def get_annualized_returns_series(log_returns_df, num_day_in_year=250):
    return (log_returns_df.mean() * num_day_in_year).round(2)

def get_annualized_std_series(log_returns_df, num_day_in_year=250):
    return (log_returns_df.std() * (num_day_in_year ** 0.5)).round(2)

# CAGR (compound annual growth rate)

cum_rtn_df = np.exp(log_rtn_df.cumsum())
cum_rtn_df.head()

cum_rtn_df.iloc[0]
cum_rtn_df.iloc[-1]

num_day_in_year = 252

# 1*(1 + x)^n = cum_rtn_df.iloc[-1]
# (1 + x)     = (cum_rtn_df.iloc[-1])**(1/n)
# --> (cum_rtn_df.iloc[-1])**(252/n) - 1
cagr = cum_rtn_df.iloc[-1]**(num_day_in_year/(len(cum_rtn_df))) - 1

def get_CAGR_series(cum_rtn_df, num_day_in_year=250):
    cagr_series = cum_rtn_df.iloc[-1]**(num_day_in_year/(len(cum_rtn_df))) - 1
    return cagr_series

get_CAGR_series(cum_rtn_df)

# Sharpe Ratio

# [학습 포인트] Statistic used to describe the performance of assets and portfolios

# [학습 포인트] "Additional return" per "unit additional risk achieved" by a portfolio, relative to a "risk-free source":
#     $$SharpeRatio = \frac{E[r_a - r_b]}{\sqrt{Var(r_a - r_b)}}$$
#     - $r_a$: returns on our asset
#     - $r_b$: risk-free rate of return

# [학습 포인트] 앞에서 배운 annualized
# $$K * Sharpe Ratio$$
#     - Why just multiply with K?

# [학습 포인트] K candidates for various sampling rates:
#     * Daily = sqrt(252)
#     * Weekly = sqrt(52)
#     * Monthly = sqrt(12)

yearly_rfr = 0.025
excess_rtns = log_rtn_df.mean()*252 - yearly_rfr
excess_rtns / (log_rtn_df.std() * np.sqrt(252))

def get_sharpe_ratio(log_rtn_df, yearly_rfr = 0.025):
    excess_rtns = log_rtn_df.mean()*252 - yearly_rfr
    return excess_rtns / (log_rtn_df.std() * np.sqrt(252))

# Drawdown

# 데이터를 시각화합니다.
cum_rtn_df.plot(figsize=(10, 5));

cummax_df = cum_rtn_df.cummax();

# 데이터를 시각화합니다.
ax = cummax_df.plot(figsize=(10, 5));
# 데이터를 시각화합니다.
cum_rtn_df.plot(ax=ax);

drawdown_df = cum_rtn_df / cummax_df - 1
# 데이터를 시각화합니다.
drawdown_df.plot(figsize=(10, 5));

# 1. MDD
mdd_series = drawdown_df.min()
mdd_series

# 2. longest_dd_period
max_point_df = drawdown_df[drawdown_df == 0]
max_point_df.head()

_df = max_point_df["KODEX200"]
_df.tail()

_df.last_valid_index()

drawdown_df["KODEX200"].last_valid_index()

_df.loc[drawdown_df["KODEX200"].last_valid_index()] = 0

_df.tail()

_df = _df.dropna()
_df.tail()

periods = _df.index[1:] - _df.index[:-1]
periods

# longest days
max_idx = periods.argmax()
max_idx

longest_days = periods.max().days
longest_days

# longest period info
longest_start_date = _df.index[:-1][max_idx].date()
longest_end_date = _df.index[1:][max_idx].date()

print(longest_days)
print("{} ~ {}".format(longest_start_date, longest_end_date))

periods.mean().days
periods.std().days

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

result = get_drawdown_infos(cum_rtn_df)
# 데이터를 시각화합니다.
result[0].plot();

result[1]
result[2]
