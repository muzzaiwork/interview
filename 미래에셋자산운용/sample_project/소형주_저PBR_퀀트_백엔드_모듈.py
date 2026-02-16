import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import logging
import platform
from datetime import datetime
from typing import Dict, List, Optional

# 1. 로깅 설정: 백엔드 플랫폼에서는 실행 로그 기록이 필수적입니다.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 2. 한글 폰트 설정 (MacOS/Windows 대응)
if platform.system() == 'Darwin': # MacOS
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 3. Pandas 출력 설정: 터미널에서 DataFrame이 잘 보이도록 설정합니다.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.unicode.east_asian_width', True)

class DataPipeline:
    """
    'AI 및 데이터 파이프라인' 역할을 수행하는 클래스입니다.
    외부 API 및 로컬 DB(CSV)로부터 데이터를 수집하고 정제합니다.
    """
    def __init__(self):
        self.raw_data = None
        self.universe = None

    def fetch_market_data(self) -> pd.DataFrame:
        """최신 시장 데이터(종목 리스트 및 지표) 수집"""
        logger.info("시장 데이터 수집 시작 (FinanceDataReader API)")
        try:
            # KRX 전체 종목 리스트 및 기본 지표 수집
            df = fdr.StockListing('KRX')
            logger.info(f"수집 완료: {len(df)}개 종목")
            return df
        except Exception as e:
            logger.error(f"데이터 수집 중 오류 발생: {e}")
            return pd.DataFrame()

    def fetch_historical_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """특정 종목의 과거 주가 데이터를 API로 직접 수집"""
        logger.info(f"과거 주가 데이터 수집: {symbol} ({start_date} ~ {end_date})")
        try:
            df = fdr.DataReader(symbol, start_date, end_date)
            return df
        except Exception as e:
            logger.error(f"주가 데이터 수집 중 오류 발생: {e}")
            return pd.DataFrame()

    def load_historical_financials(self, path: str) -> pd.DataFrame:
        """백테스팅을 위한 대량의 과거 재무제표 데이터 로드"""
        logger.info(f"로컬 재무제표 데이터 로드: {path}")
        df = pd.read_csv(path)
        # 데이터 정제 (Lec 1-2 기초 활용)
        df = df.rename(columns={
            "P/B(Adj., FY End)": "PBR",
            "P/E(Adj., FY End)": "PER",
            "수정주가": "Price"
        })
        return df

class QuantStrategy:
    """
    퀀트 전략을 정의하는 엔진입니다. (Lec 1-3 그룹화 및 범주화 활용)
    """
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        소형주 + 저PBR 전략 시그널 생성
        """
        logger.info("전략 시그널 생성 중: 소형주(하위 20%) & 저PBR 상위 20개")
        
        # 1. 연도별 시가총액 하위 20% 필터링 (Lec 1-3 groupby + quantile)
        market_cap_limit = df.groupby("year")['시가총액'].transform(lambda x: x.quantile(0.2))
        small_cap_mask = df['시가총액'] <= market_cap_limit
        
        print(f"\n" + "="*50)
        print(f" [Step 1] 소형주 필터링 결과 (market_cap_limit 추가)")
        print(f"="*50)
        temp_df = df.copy()
        temp_df['기준선'] = market_cap_limit
        temp_df['소형주여부'] = small_cap_mask
        print(temp_df[['Name', 'year', '시가총액', '기준선', '소형주여부']].head())
        
        # 2. 저PBR 종목 선별 (Lec 1-2 sort_values + nsmallest)
        # PBR 0.2 미만은 데이터 오류 가능성으로 제외
        filtered_df = df[small_cap_mask & (df['PBR'] >= 0.2)]
        print(f"\n" + "="*50)
        print(f" [Step 2] PBR 0.2 이상 필터링 완료 (filtered_df)")
        print(f"="*50)
        print(f"대상 종목 수: {len(filtered_df)}개")
        print(filtered_df[['Name', 'year', '시가총액', 'PBR']].head())
        
        # 연도별로 PBR 낮은 순 20개 선정
        selected = filtered_df.sort_values('PBR').groupby('year').head(20)
        print(f"\n" + "="*50)
        print(f" [Step 3] 연도별 저PBR 20개 선정 완료 (selected)")
        print(f"="*50)
        print(selected[['Name', 'year', '시가총액', 'PBR']].head())
        
        # 시그널 매트릭스 생성 (Lec 1-4 pivot)
        signal_df = selected.pivot(index='year', columns='Name', values='PBR').notna()
        print(f"\n" + "="*50)
        print(f" [Step 4] 시그널 매트릭스 (signal_df - 일부)")
        print(f"="*50)
        print(signal_df.iloc[:5, :5])
        return signal_df

class BacktestEngine:
    """
    퀀트 매매 전략 백테스팅 및 분석 플랫폼의 핵심 엔진입니다. (Lec 2 시계열 분석 활용)
    """
    def __init__(self, price_df: pd.DataFrame, signal_df: pd.DataFrame):
        self.price_df = price_df
        self.signal_df = signal_df
        self.results = {}

    def run(self):
        """백테스팅 실행 및 성과 분석 (Lec 2-2, 2-3)"""
        logger.info("벡터화 백테스팅 엔진 가동")
        
        # 1. 수익률 계산 (Lec 2-2)
        returns_df = self.price_df.pct_change().shift(-1)
        print(f"\n" + "="*50)
        print(f" [Step 5] 수익률 매트릭스 (returns_df - 일부)")
        print(f"="*50)
        print(returns_df.iloc[:5, :5])
        
        # 2. 전략 수익률 계산 (벡터화 연산 - Lec 0)
        # signal_df와 returns_df의 인덱스/컬럼이 자동 정렬됨 (Alignment)
        portfolio_returns = (returns_df * self.signal_df.astype(int)).mean(axis=1)
        portfolio_returns = portfolio_returns.fillna(0)
        print(f"\n" + "="*50)
        print(f" [Step 6] 포트폴리오 연간 수익률 (portfolio_returns)")
        print(f"="*50)
        print(portfolio_returns.head())
        
        # 3. 누적 수익률 계산
        cum_returns = (1 + portfolio_returns).cumprod()
        
        # 4. 주요 성과 지표 계산 (Lec 2-3)
        self._calculate_metrics(portfolio_returns, cum_returns)
        
        return cum_returns

    def _calculate_metrics(self, returns, cum_returns):
        """CAGR, MDD, Sharpe Ratio 등 전문 지표 산출"""
        # CAGR (연평균 성장률)
        n_years = len(returns)
        cagr = (cum_returns.iloc[-1] ** (1/n_years)) - 1
        
        # MDD (최대 낙폭)
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        mdd = drawdown.min()
        
        # Sharpe Ratio (단순화: 무위험 수익률 0 가정)
        vol = returns.std() * np.sqrt(1) # 연간 데이터 기준
        sharpe = (returns.mean() / returns.std()) if vol != 0 else 0
        
        self.results = {
            "Final Return": cum_returns.iloc[-1],
            "CAGR": cagr,
            "MDD": mdd,
            "Sharpe Ratio": sharpe
        }
        logger.info(f"백테스팅 완료: CAGR {cagr:.2%}, MDD {mdd:.2%}")

def main():
    """
    전체 파이프라인 통합 실행
    """
    print("\n" + "="*60)
    print(" 소형주 저PBR 퀀트 백엔드 플랫폼")
    print("="*60 + "\n")

    # 1. 데이터 파이프라인 구축
    pipeline = DataPipeline()
    # 최신 데이터 (API) - 실제 업무에서는 실시간 감시 및 DB 적재에 활용
    latest_market = pipeline.fetch_market_data()
    
    # [API 활용 예시] 특정 종목의 과거 주가를 API로 직접 가져올 수도 있습니다.
    # samsung_history = pipeline.fetch_historical_prices('005930', '2020-01-01', '2023-12-31')
    
    # 과거 데이터 (CSV) - 백테스팅 엔진의 연료
    # 수천 개 종목의 과거 재무 지표(PBR, PER 등)를 API로 매번 호출하는 것은 속도가 매우 느리므로,
    # 백테스팅 시에는 전처리가 완료된 벌크 데이터(CSV/DB)를 사용하는 것이 일반적입니다.
    hist_data = pipeline.load_historical_financials("my_data/fin_statement_new.csv")

    # 2. 전략 엔진 구동
    strategy = QuantStrategy()
    signals = strategy.generate_signals(hist_data)

    # 3. 백테스팅 엔진 구동
    price_matrix = hist_data.pivot(index='year', columns='Name', values='Price')
    engine = BacktestEngine(price_matrix, signals)
    cum_rtn = engine.run()

    # 4. 결과 리포팅 (Lec 1-5 시각화)
    print("\n" + "-"*40)
    print(" [최종 성과 리포트] ")
    for k, v in engine.results.items():
        print(f" - {k}: {v:.4f}")
    print("-"*40)

    # 시각화
    plt.figure(figsize=(12, 7))
    plt.plot(cum_rtn.index, cum_rtn, marker='o', color='#003366', linewidth=2, label='누적 수익률')
    
    # 제목 및 축 설정
    plt.title("퀀트 백테스팅: 소형주 저PBR 가치 전략", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Year (투자 연도)", fontsize=12)
    plt.ylabel("Cumulative Return (누적 수익률 지수)", fontsize=12)
    
    # 전략 설명 추가 (텍스트 박스)
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
    
    # 그래프 파일 저장 (서버 환경 가정)
    # plt.savefig("strategy_result.png")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
