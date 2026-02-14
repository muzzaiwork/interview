# Lec 2-7: 리밸런싱 기반 백테스팅 (Rebalancing-based Backtesting)

이 장에서는 정기적으로 포트폴리오의 비중을 조정하는 **리밸런싱(Rebalancing)** 전략의 구현 방법과 백테스팅 원리를 학습합니다. 단순 보유(Buy & Hold) 전략과의 차이점을 이해하고, 리밸런싱 주기에 따른 성과 변화를 시뮬레이션합니다.

## 1. Buy & Hold vs Rebalancing

### 1.1. Buy & Hold (단순 보유)
최초에 설정한 자산 비중을 그대로 유지하며 시간이 흐름에 따라 비중이 자연스럽게 변하도록 두는 방식입니다.

```python
import pandas as pd
import numpy as np

# 자산 가격 데이터
df = pd.DataFrame({
    "A": [10, 15, 12, 13, 10, 11, 12],
    "B": [10, 10, 8, 13, 12, 12, 12],
    "C": [10, 12, 14, 16, 14, 14, 16],
}, index=pd.to_datetime(["2018-01-31", "2018-02-10", "2018-02-20", "2018-02-28", "2018-03-20", "2018-03-29", "2018-04-30"]))

# 초기 비중 [0.3, 0.5, 0.2] 적용
cum_rtn_df = df / df.iloc[0]
asset_flow_df = cum_rtn_df * [0.3, 0.5, 0.2]
port_val = asset_flow_df.sum(axis=1)

print("Buy & Hold Portfolio Value:")
print(port_val.tail(3))
# 출력:
# 2018-03-20   1.1800
# 2018-03-29   1.2100
# 2018-04-30   1.2800
```

### 1.2. Periodic Rebalancing (정기 리밸런싱)
매월 말 또는 특정 주기마다 자산 비중을 목표 비중(Target Weight)으로 강제 조정합니다.

```python
# 리밸런싱 시뮬레이션 (핵심 로직)
target_weight = np.array([0.3, 0.5, 0.2])
cum_rtn_at_last_rebal = 1.0
individual_port_val_list = []

# rebal_index: 각 월의 마지막 날짜 리스트
for start, end in zip(rebal_index[:-1], rebal_index[1:]):
    sub_price_df = df.loc[start:end]
    sub_cum_rtn_df = sub_price_df / sub_price_df.iloc[0]
    
    # 주기 시작 시점의 비중 적용 및 이전 누적 가치 전파
    indi_port_val_df = (sub_cum_rtn_df * target_weight) * cum_rtn_at_last_rebal
    individual_port_val_list.append(indi_port_val_df)
    
    # 다음 주기를 위한 시작 가치 갱신
    cum_rtn_at_last_rebal = indi_port_val_df.sum(axis=1).iloc[-1]

# 전체 포트폴리오 가치 합산
full_port_val = pd.concat(individual_port_val_list).sum(axis=1)
# (중복 제거 및 병합 과정 생략)
```

---

## 2. 리밸런싱의 효과와 의미

1.  **비중 유지**: 수익률이 좋은 자산의 비중이 과도하게 커지는 것을 방지하여 포트폴리오의 리스크(변동성)를 관리합니다.
2.  **역발상 투자 (Buy Low, Sell High)**: 가격이 오른 자산을 팔고 가격이 내린 자산을 삼으로써 자연스럽게 저가 매수, 고가 매도를 실천하게 됩니다.
3.  **복리 효과 극대화**: 자산 간 상관관계가 낮은 경우 리밸런싱을 통해 변동성을 줄이면서 장기적인 복리 수익률을 향상시킬 수 있습니다 (Rebalancing Bonus).

---

## 3. 학습 포인트 (퀀트 투자 관점)

1.  **Event-based vs Vectorized**: 시그널 기반 백테스팅이 전체 기간에 대해 한 번에 연산하는 벡터화 방식 위주였다면, 리밸런싱 백테스팅은 주기별로 루프를 돌며 상태를 갱신하는 이벤트 기반(Event-based) 성격이 강합니다.
2.  **거래 비용의 반영**: 실제 리밸런싱 시에는 매매 수수료와 슬리피지(Slippage)가 발생합니다. `cum_rtn_at_last_rebal`을 갱신할 때 패널티(예: * 0.999)를 부여하여 더 정교한 백테스팅이 가능합니다.
3.  **리밸런싱 주기 최적화**: 너무 잦은 리밸런싱은 거래 비용을 증가시키고, 너무 드문 리밸런싱은 목표 비중에서 크게 벗어나게 합니다. 전략의 특성에 맞는 적절한 주기(월간, 분기 등) 선정이 필요합니다.

---

## 4. 주요 라이브러리 함수 정리

| 라이브러리 | 클래스/모듈 | 함수명 | 설명 |
| :--- | :--- | :--- | :--- |
| **Pandas** | `DataFrame` | `divide()` | DataFrame을 다른 객체로 나눔 (axis 지정 가능) |
| **Pandas** | `DataFrame` | `sum(axis=1)` | 행 방향(자산 간) 합계를 계산하여 포트폴리오 가치 산출 |
| **Pandas** | `Index` | `duplicated()` | 중복된 인덱스 존재 여부 확인 (리밸런싱 날짜 처리 시 유용) |
| **Pandas** | `pd` | `concat()` | 리밸런싱 주기별로 분절된 DataFrame들을 하나로 통합 |
| **NumPy** | `np` | `array()` | 목표 비중(Weight) 등을 담는 수치 배열 생성 |
| **itertools** | `product` | `product()` | 여러 전략 매개변수(종목, 비중, 주기)의 모든 조합 생성 |
| **functools** | `reduce` | `reduce()` | 리스트 내의 여러 DataFrame을 순차적으로 병합할 때 사용 |
