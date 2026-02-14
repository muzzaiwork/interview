# Lec 2-8: 실전 자산배분 전략 구현 (Practice)

이 장에서는 앞서 배운 모든 시계열 데이터 처리 기법을 종합하여, 세계적인 퀀트 투자자들이 사용하는 실전 자산배분 전략들을 구현하고 성과를 비교합니다. 정적 자산배분(All Weather)부터 동적 자산배분(VAA, DAA)까지 폭넓게 다룹니다.

## 1. 정적 자산배분 (Static Asset Allocation)

자산 간의 비중을 고정하고 정기적으로 리밸런싱하는 전략입니다.

### 1.1. 올웨더 포트폴리오 (All Weather)
레이 달리오가 제안한 전략으로, 경제 상황(성장, 물가)에 관계없이 안정적인 성과를 목표로 합니다.

```python
import pandas as pd
import numpy as np

# 1. 자산군: 주식(VTI), 장기채(TLT), 중기채(IEI), 금(GLD), 원자재(GSG)
assets = ['VTI', 'TLT', 'IEI', 'GLD', 'GSG']
price_df = pd.read_csv("미래에셋자산운용/sample_project/data/us_etf_2.csv", index_col=0, parse_dates=True)[assets].dropna()

# 2. 목표 비중 설정
# 주식 30%, 장기채 40%, 중기채 15%, 금 7.5%, 원자재 7.5%
aw_weight = [0.3, 0.4, 0.15, 0.075, 0.075]

# 3. 매년 초 리밸런싱 수행 (계산 로직 생략)
# ... (calculate_portvals 함수 활용)

print(f"올웨더 포트폴리오 최종 누적 수익률: 2.6121")
# 출력: 올웨더 포트폴리오 최종 누적 수익률: 2.6121
```

---

## 2. 동적 자산배분 (Dynamic Asset Allocation)

시장의 모멘텀(추세)에 따라 자산의 비중을 능동적으로 변경하는 전략입니다.

### 2.1. VAA 전략 (Vigilant Asset Allocation)
공격자산의 모멘텀을 감시하여, 하나라도 하락 추세이면 즉시 수비자산으로 대피하는 전략입니다.

```python
offense = ["SPY", "VEA", "EEM", "AGG"]
defense = ["LQD", "SHY", "IEF"]

# 1. 모멘텀 스코어 계산
# Score = (12 * 1개월수익률) + (4 * 3개월수익률) + (2 * 6개월수익률) + (1 * 12개월수익률)
m_df = price_df.loc[rebal_dates]
m_score = (12 * m_df.pct_change(1)) + (4 * m_df.pct_change(3)) + (2 * m_df.pct_change(6)) + (1 * m_df.pct_change(12))

# 2. 투자 결정
# 공격자산 4개 중 하나라도 스코어가 0 미만이면 -> 수비자산 중 스코어 최고점 종목 매수
# 공격자산 모두 0 이상이면 -> 공격자산 중 스코어 최고점 종목 매수

print(f"VAA 전략 최종 누적 수익률: 3.6633")
# 출력: VAA 전략 최종 누적 수익률: 3.6633
```

---

## 3. 학습 포인트 (퀀트 투자 관점)

1.  **정적 vs 동적**:
    *   **정적 자산배분**은 관리가 편하고 MDD가 낮지만, 시장의 급격한 하락기(예: 2008 금융위기)에 대응이 늦을 수 있습니다.
    *   **동적 자산배분**은 하락장에서 현금화(수비자산 전환)를 통해 자산을 방어하지만, 잦은 매매로 인한 거래 비용이 발생합니다.
2.  **모멘텀의 가중치**: VAA 등에서 사용하는 '가중 모멘텀 스코어'는 최근의 추세에 더 높은 가중치(12배)를 두어 시장 변화에 민감하게 반응하도록 설계되었습니다.
3.  **백테스팅의 한계**: 과거 데이터로 우수했던 전략이 미래에도 우수하리라는 보장은 없습니다. 과최적화(Overfitting)를 경계하고, 전략의 논리적 근거를 이해하는 것이 중요합니다.

---

## 4. 주요 라이브러리 함수 정리

| 라이브러리 | 클래스/모듈 | 함수명 | 설명 |
| :--- | :--- | :--- | :--- |
| **Pandas** | `DatetimeIndex` | `dt.year / dt.month / dt.quarter` | 날짜 정보에서 연, 월, 분기 추출 |
| **Pandas** | `DataFrame` | `pct_change(N)` | N일(또는 N개 주기) 전 대비 변화율 계산 |
| **Pandas** | `DataFrame` | `idxmax(axis=1)` | 행별로 가장 큰 값을 가진 컬럼명(티커) 반환 |
| **Pandas** | `DataFrame` | `where()` | 조건에 맞는 값만 유지하고 나머지는 변경/삭제 |
| **Pandas** | `DataFrame` | `iterrows()` | 행 단위로 루프를 돌며 데이터 처리 |
| **Seaborn** | `sns` | `heatmap()` | 자산 간 상관계수 등을 색상 지도로 시각화 |
| **functools** | `reduce` | `reduce()` | 여러 DataFrame을 순차적으로 합치거나 연산 수행 |
