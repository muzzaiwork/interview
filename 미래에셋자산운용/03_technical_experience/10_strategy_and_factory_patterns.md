# 🏗️ 전략 패턴(Strategy) & 팩토리 패턴(Factory) 상세 설명

자산운용 플랫폼과 같은 복잡한 금융 시스템에서 **변경에 유연하고 확장이 쉬운 코드**를 작성하기 위해 가장 빈번하게 사용되는 두 가지 디자인 패턴입니다.

---

## 1. 전략 패턴 (Strategy Pattern)

### 🎯 개념
객체의 행위를 직접 구현하지 않고, **전략(Strategy)**이라 불리는 독립적인 클래스로 캡슐화하여 상황에 따라 동적으로 교체할 수 있게 만드는 패턴입니다.

### 💡 퀀트 플랫폼 적용 예시 (매매 전략)
퀀트 시스템에는 '저PBR 전략', '모멘텀 전략', '이동평균 교차 전략' 등 수많은 매매 로직이 존재합니다. 이를 `if-else`로 구현하면 새로운 전략이 추가될 때마다 엔진 코드를 수정해야 하지만, 전략 패턴을 쓰면 엔진 수정 없이 전략만 추가할 수 있습니다.

```python
from abc import ABC, abstractmethod

# [Strategy Interface] 모든 전략의 공통 분모
class TradingStrategy(ABC):
    @abstractmethod
    def generate_signals(self, data):
        pass

# [Concrete Strategy 1] 저PBR 전략
class LowPbrStrategy(TradingStrategy):
    def generate_signals(self, data):
        return "저PBR 종목 매수 시그널 생성"

# [Concrete Strategy 2] 모멘텀 전략
class MomentumStrategy(TradingStrategy):
    def generate_signals(self, data):
        return "최근 상승세 종목 매수 시그널 생성"

# [Context] 전략을 사용하는 엔진
class BacktestEngine:
    def __init__(self, strategy: TradingStrategy):
        self.strategy = strategy  # 전략 주입 (Dependency Injection)

    def run(self, data):
        return self.strategy.generate_signals(data)
```

### ✅ 장점
- **OCP(개방-폐쇄 원칙) 준수**: 기존 엔진 코드를 변경하지 않고 새로운 전략을 무한히 확장 가능.
- **코드 재사용**: 동일한 로직을 여러 곳에서 독립적으로 사용 가능.
- **테스트 용이성**: 각 전략을 독립적으로 유닛 테스트(Unit Test) 가능.

---

## 2. 팩토리 패턴 (Factory Pattern)

### 🎯 개념
객체 생성 로직을 별도의 클래스나 메서드로 분리하여, **어떤 객체를 생성할지 결정하는 책임**을 한곳에 모으는 패턴입니다.

### 💡 퀀트 플랫폼 적용 예시 (전략 생성기)
사용자가 화면에서 '저PBR' 또는 '모멘텀'을 선택했을 때, 적절한 전략 객체를 만들어주는 역할을 수행합니다.

```python
class StrategyFactory:
    @staticmethod
    def create_strategy(strategy_type: str) -> TradingStrategy:
        if strategy_type == "LOW_PBR":
            return LowPbrStrategy()
        elif strategy_type == "MOMENTUM":
            return MomentumStrategy()
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

# [Usage] 클라이언트는 구체적인 클래스명을 몰라도 됨
selected_type = "LOW_PBR" # 사용자 입력
strategy = StrategyFactory.create_strategy(selected_type)
engine = BacktestEngine(strategy)
```

### ✅ 장점
- **객체 생성 로직 캡슐화**: 클라이언트 코드는 객체가 '어떻게' 만들어지는지 몰라도 됨.
- **결합도 낮춤 (Decoupling)**: 구체적인 클래스 이름 대신 인터페이스나 부모 클래스에 의존하게 함.
- **중복 제거**: 동일한 객체 생성 로직이 여기저기 흩어지는 것을 방지.

---

## 🤝 두 패턴의 시너지 (Strategy + Factory)

실무(예: 에잇퍼센트 상환 로직)에서는 이 두 패턴을 결합하여 강력한 아키텍처를 만듭니다.

1.  **Strategy**: "어떻게(How)" 계산할 것인가? (상환 방식, 매매 로직 등)
2.  **Factory**: "어떤(Which)" 전략을 쓸 것인가? (상품 코드, 유저 설정에 따른 선택)

### 🚀 자산운용 플랫폼에서의 활용 시나리오
> "분석가가 100가지의 팩터 조합을 테스트하고 싶어 할 때, 각 조합을 **Strategy**로 정의하고, 사용자의 설정값에 따라 적절한 전략 객체를 **Factory**가 생성하여 **분산 백테스팅 엔진(Ray/Spark)**에 전달하는 구조를 설계할 수 있습니다. 이를 통해 엔진의 코어 로직은 건드리지 않으면서도 수만 개의 전략 시뮬레이션을 유연하게 지원할 수 있습니다."

---

## 📂 연관 문서
- [07. 디자인 패턴 적용 사례](./03_technical_experience/07_design_pattern_application.md)
- [06. 백테스팅 엔진 상세](../06_technical_agenda/06_backtest_engine_detail.md)
