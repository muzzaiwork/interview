# 06. 퀀트 백테스팅 엔진 상세 (Backtest Engine)

매매 전략 백테스팅은 사용자님의 말씀대로 **저장된 과거 데이터에 특정 수식(전략)을 적용하여 시뮬레이션 연산을 수행하는 과정**이 맞습니다. 하지만 금융 플랫폼 엔지니어링 관점에서는 단순 연산을 넘어 '정합성'과 '성능'이 매우 중요합니다.

## ⚙️ 백테스팅 실행 흐름

```mermaid
sequenceDiagram
    participant P as 파이프라인 (Data)
    participant E as 백테스팅 엔진 (Engine)
    participant S as 전략 (Strategy)

    Note over P: 정제된 시계열 데이터 준비
    P->>E: 데이터 주입 (Price, Financials)
    E->>S: 시그널 요청
    Note over S: 팩터 계산 및 종목 선정
    S-->>E: 매수/매도 시그널 반환
    Note over E: 수익률 매칭 (Point-in-Time)
    Note over E: 벡터화 행렬 연산
    E->>E: 성과 지표 계산 (CAGR, MDD)
    E-->>User: 최종 리포트 및 그래프 출력
```

## ⚙️ 백테스팅 엔진의 핵심 메커니즘

백테스팅 엔진은 크게 **[시그널 생성 -> 수익률 매칭 -> 성과 지표 산출]**의 3단계를 거칩니다.

### 1. 시그널 생성 (Signal Generation)
- **과정**: 정제된 재무/시장 데이터(PBR, 시가총액 등)를 수식에 대입하여 특정 시점에 어떤 종목을 매수/매도할지 결정합니다.
- **예시**: `if PBR < 0.5 and MarketCap < 20th_percentile: return BUY`
- **엔지니어링 포인트**: 
    - **Vectorization (벡터화)**: For-문(반복문)을 돌리지 않고 Pandas/NumPy의 행렬 연산을 사용하여 수만 개의 종목을 한 번에 계산합니다.

**Python 예시:**
```python
# 1. 시가총액 하위 20% 필터링 (벡터화 연산)
market_cap_limit = df.groupby("year")['시가총액'].transform(lambda x: x.quantile(0.2))
small_cap_mask = df['시가총액'] <= market_cap_limit

# 2. 저PBR 종목 선별 및 시그널 생성 (연도 x 종목 매트릭스)
selected = df[small_cap_mask].sort_values('PBR').groupby('year').head(20)
signal_df = selected.pivot(index='year', columns='Name', values='PBR').notna()
```

### 2. 수익률 매칭 (Return Alignment)
- **과정**: 결정된 시그널(매수 종목)에 실제 그다음 날 또는 그다음 달의 수익률 데이터를 곱합니다.
- **핵심 주의사항 (Look-ahead Bias)**: 
    - 미래의 데이터를 미리 보고 오늘 매수하는 오류를 방지해야 합니다. 
    - "오늘 종가를 보고 오늘 종가에 산다"는 것은 불가능하므로, 시점 정렬(T시점 시그널 -> T+1시점 수익률)이 기술적으로 매우 중요합니다.

**Python 예시:**
```python
# 일별 수익률 계산 및 시점 정렬 (T+1 수익률을 T시점으로 당김)
returns_df = price_df.pct_change().shift(-1)

# 전략 수익률 계산 (시그널과 수익률 매트릭스의 행렬 곱)
# Pandas의 인덱스 정렬(Alignment) 기능을 통해 자동으로 종목/날짜 매칭
portfolio_returns = (returns_df * signal_df.astype(int)).mean(axis=1)
```

### 3. 성과 지표 산출 (Performance Metrics)
- **과정**: 누적된 수익률 곡선을 바탕으로 전략의 우수성을 평가하는 지표를 계산합니다.
- **주요 수식**:
    - **CAGR (연평균 성장률)**: 복리 개념을 적용한 연평균 수익률
    - **MDD (최대 낙폭)**: 특정 기간 동안 고점 대비 가장 많이 하락한 비율 (위험 측정의 핵심)
    - **Sharpe Ratio (샤프 지수)**: 변동성(위험) 대비 수익이 얼마나 높은지 나타내는 지표

**Python 예시:**
```python
# 누적 수익률 계산
cum_returns = (1 + portfolio_returns).cumprod()

# CAGR 계산 (n_years: 총 투자 연수)
cagr = (cum_returns.iloc[-1] ** (1/n_years)) - 1

# MDD 계산
peak = cum_returns.cummax()
drawdown = (cum_returns - peak) / peak
mdd = drawdown.min()
```

## 🚀 엔지니어로서의 기술적 아젠다

미래에셋자산운용의 Platform Engineering 팀에서는 이 엔진을 다음과 같이 고도화하게 됩니다.

1. **분산 연산 (Distributed Computing)**:
   - 한두 개의 전략이 아니라, 수천 개의 파라미터 조합(Grid Search)을 동시에 테스트하기 위해 Kubernetes 워커 노드에 연산을 분산시킵니다.
2. **이벤트 기반 시뮬레이션 (Event-driven)**:
   - 단순 일별 데이터가 아니라, 실제 시장에서 발생하는 호가(Orderbook) 이벤트를 재현하여 슬리피지(체결 오차)까지 계산하는 정교한 엔진 구축.
3. **C++/Rust 연동**:
   - 파이썬의 속도 한계를 극복하기 위해 핵심 연산 로직(수익률 계산부)을 C++ 등으로 작성하여 `Pybind11` 등을 통해 연결.

## 💡 면접용 핵심 요약
> "백테스팅은 과거의 데이터를 연료로 삼아 투자 전략이라는 엔진을 가동하는 시뮬레이션입니다. 저는 이 과정에서 **데이터의 시점 정렬(Point-in-Time)**을 엄격히 관리하여 결과의 왜곡을 방지하고, **벡터화 연산**을 통해 대규모 전략 테스트의 효율성을 극대화하는 플랫폼을 구축하는 데 기여하고 싶습니다."
