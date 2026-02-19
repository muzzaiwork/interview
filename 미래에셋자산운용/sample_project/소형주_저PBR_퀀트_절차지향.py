import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import logging
import platform
from datetime import datetime

# 1. 로깅 및 기본 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.unicode.east_asian_width', True)

print("\n" + "="*60)
print(" 소형주 저PBR 퀀트 백엔드 플랫폼 (절차지향 버전)")
print("="*60 + "\n")

# 2. 데이터 수집 (Data Ingestion)
logger.info("시장 데이터 수집 시작 (FinanceDataReader API)")
try:
    latest_market = fdr.StockListing('KRX')
    logger.info(f"수집 완료: {len(latest_market)}개 종목")
except Exception as e:
    logger.error(f"데이터 수집 중 오류 발생: {e}")
    latest_market = pd.DataFrame()

# 3. 과거 데이터 로드 및 정제 (ETL)
csv_path = "미래에셋자산운용/sample_project/my_data/fin_statement_new.csv"
logger.info(f"로컬 재무제표 데이터 로드: {csv_path}")
hist_data = pd.read_csv(csv_path)
hist_data = hist_data.rename(columns={
    "P/B(Adj., FY End)": "PBR",
    "P/E(Adj., FY End)": "PER",
    "수정주가": "Price"
})

# 4. 전략 시그널 생성 (Strategy Logic)
logger.info("전략 시그널 생성 중: 소형주(하위 20%) & 저PBR 상위 20개")

# Step 1: 연도별 시가총액 하위 20% 필터링
market_cap_limit = hist_data.groupby("year")['시가총액'].transform(lambda x: x.quantile(0.2))
small_cap_mask = hist_data['시가총액'] <= market_cap_limit

print(f"\n" + "="*50)
print(f" [Step 1] 소형주 필터링 결과 (일부)")
print(f"="*50)
temp_df = hist_data.copy()
temp_df['기준선'] = market_cap_limit
temp_df['소형주여부'] = small_cap_mask
print(temp_df[['Name', 'year', '시가총액', '기준선', '소형주여부']].head())

# Step 2: 저PBR 종목 선별 (PBR 0.2 미만 제외)
filtered_df = hist_data[small_cap_mask & (hist_data['PBR'] >= 0.2)]
print(f"\n" + "="*50)
print(f" [Step 2] PBR 0.2 이상 필터링 완료")
print(f"="*50)
print(f"대상 종목 수: {len(filtered_df)}개")

# Step 3: 연도별 저PBR 20개 선정
selected = filtered_df.sort_values('PBR').groupby('year').head(20)
print(f"\n" + "="*50)
print(f" [Step 3] 연도별 저PBR 20개 선정 완료")
print(f"="*50)
print(selected[['Name', 'year', '시가총액', 'PBR']].head())

# Step 4: 시그널 매트릭스 생성 (Pivot)
signal_df = selected.pivot(index='year', columns='Name', values='PBR').notna()
print(f"\n" + "="*50)
print(f" [Step 4] 시그널 매트릭스 (일부)")
print(f"="*50)
print(signal_df.iloc[:5, :5])

# 5. 백테스팅 실행 (Backtesting Engine)
logger.info("벡터화 백테스팅 엔진 가동")
price_matrix = hist_data.pivot(index='year', columns='Name', values='Price')

# Step 5: 수익률 계산 (T+1 수익률을 T시점으로 정렬)
returns_df = price_matrix.pct_change().shift(-1)
print(f"\n" + "="*50)
print(f" [Step 5] 수익률 매트릭스 (일부)")
print(f"="*50)
print(returns_df.iloc[:5, :5])

# Step 6: 포트폴리오 수익률 계산 (벡터화 연산 - 핵심 로직)
# (returns_df * signal_df.astype(int)): 
# - signal_df는 해당 연도에 종목을 샀으면 True(1), 안 샀으면 False(0)인 행렬입니다.
# - returns_df(수익률 행렬)와 곱하면, 내가 산 종목의 수익률만 남고 나머지는 0이 됩니다.
# .mean(axis=1):
# - 가로 방향(axis=1)으로 평균을 내어, 그 해에 보유한 종목들의 '평균 수익률'을 구합니다. (동일 비중 투자 가정)
portfolio_returns = (returns_df * signal_df.astype(int)).mean(axis=1)
portfolio_returns = portfolio_returns.fillna(0)
print(f"\n" + "="*50)
print(f" [Step 6] 포트폴리오 연간 수익률")
print(f"="*50)
print(portfolio_returns.head())

# Step 7: 누적 수익률 계산
cum_rtn = (1 + portfolio_returns).cumprod()

# 6. 성과 지표 산출 (Performance Metrics)
n_years = len(portfolio_returns)
final_return = cum_rtn.iloc[-1]
cagr = (final_return ** (1/n_years)) - 1

peak = cum_rtn.cummax()
drawdown = (cum_rtn - peak) / peak
mdd = drawdown.min()

sharpe = (portfolio_returns.mean() / portfolio_returns.std()) if portfolio_returns.std() != 0 else 0

logger.info(f"백테스팅 완료: CAGR {cagr:.2%}, MDD {mdd:.2%}")

# 7. 결과 리포팅 및 시각화
print("\n" + "-"*40)
print(" [최종 성과 리포트] ")
print(f" - Final Return: {final_return:.4f}")
print(f" - CAGR: {cagr:.4f}")
print(f" - MDD: {mdd:.4f}")
print(f" - Sharpe Ratio: {sharpe:.4f}")
print("-"*40)

plt.figure(figsize=(12, 7))
plt.plot(cum_rtn.index, cum_rtn, marker='o', color='#003366', linewidth=2, label='누적 수익률')
plt.title("퀀트 백테스팅: 소형주 저PBR 가치 전략 (절차지향)", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Year (투자 연도)", fontsize=12)
plt.ylabel("Cumulative Return (누적 수익률 지수)", fontsize=12)

strategy_desc = (
    "[전략 개요]\n"
    "1. 대상: 시가총액 하위 20% 소형주\n"
    "2. 조건: 저PBR 상위 20개 종목 선정\n"
    "3. 필터: PBR 0.2 미만 제외 (부실주 방지)\n"
    "4. 리밸런싱: 매년 1회 종목 교체"
)
plt.text(0.02, 0.95, strategy_desc, transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.grid(True, alpha=0.3)
plt.axhline(1, color='red', linestyle='--', alpha=0.5, label='원금(1.0)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
