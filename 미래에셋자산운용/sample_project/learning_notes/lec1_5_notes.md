# Lec 1-5: 데이터 시각화 (Visualization)

이 장에서는 데이터 분석의 결과를 직관적으로 이해하기 위한 시각화 도구인 **Matplotlib**, **Pandas Plot**, 그리고 **Seaborn**의 사용법을 학습합니다. 퀀트 투자에서 주가 차트, 수익률 분포, 변동성 분석 등을 수행할 때 필수적인 기술입니다.

## 1. Matplotlib 인터페이스 이해

Matplotlib은 두 가지 인터페이스를 제공하지만, 확장성과 직관성을 위해 **상태가 없는(Stateless) 객체 지향 방식**의 사용이 권장됩니다.

### 1.1. 객체 지향(Object-Oriented) 방식
`Figure`(전체 틀)와 `Axes`(그래프가 그려지는 공간) 객체를 명시적으로 생성하여 제어합니다.

```python
import matplotlib.pyplot as plt

x = [-3, 5, 7]
y = [10, 2, 5]

# Figure와 Axes 생성
fig, ax = plt.subplots(figsize=(10, 3))

# Axes 객체에 그래프 그리기 및 속성 설정
ax.plot(x, y)
ax.set_xlim(0, 10)
ax.set_ylim(-3, 8)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Line Plot')
fig.suptitle('Figure Title', size=12)

print(f"Figure 객체: {type(fig)}")
# 출력: Figure 객체: <class 'matplotlib.figure.Figure'>
print(f"Axes 객체: {type(ax)}")
# 출력: Axes 객체: <class 'matplotlib.axes._axes.Axes'>
```

### 1.2. 다중 그래프 (Subplots)
`nrows`와 `ncols`를 지정하여 한 화면에 여러 개의 그래프를 배치할 수 있습니다. 이때 `axes`는 NumPy 배열 형태로 반환됩니다.

```python
# 2행 2열의 서브플롯 생성
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

print(f"Axes 배열 타입: {type(axes)}")
# 출력: Axes 배열 타입: <class 'numpy.ndarray'>
print(f"Axes 배열 형태: {axes.shape}")
# 출력: Axes 배열 형태: (2, 2)

# 첫 번째 칸에 그래프 그리기
axes[0, 0].plot([1, 2, 3], [4, 5, 6])
```

---

## 2. Pandas 내장 Plotting

Pandas의 `Series`나 `DataFrame` 객체는 `plot()` 메서드를 통해 Matplotlib을 내부적으로 호출하여 간편하게 시각화할 수 있습니다.

```python
import FinanceDataReader as fdr
import pandas as pd

# 삼성전자 데이터 로드
samsung_series = fdr.DataReader("005930", "2017-01-01", "2017-01-15")['Close']

# 간단한 선 그래프
ax = samsung_series.plot(title="Samsung Close Price")
print(f"반환된 Axes: {type(ax)}")
# 출력: 반환된 Axes: <class 'matplotlib.axes._axes.Axes'>

# 다양한 종류의 그래프 (kind 인자 사용)
# 'line', 'bar', 'hist', 'box', 'kde', 'scatter' 등 지원
samsung_series.plot(kind='hist', bins=10)
```

---

## 3. Seaborn을 활용한 고급 시각화

Seaborn은 Matplotlib 기반의 고수준 인터페이스로, 통계적인 관계를 시각화하는 데 최적화되어 있습니다.

```python
import seaborn as sns

# 데이터 로드 및 전처리
df = pd.read_csv("미래에셋자산운용/sample_project/my_data/Small_and_Big.csv", index_col=0, parse_dates=["date"])
median_val = df['시가총액 (보통)(평균)(원)'].median()
df['size'] = np.where(df['시가총액 (보통)(평균)(원)'] < median_val, "small", "big")

# 1. 빈도 분석 (Count Plot)
sns.countplot(x="size", data=df)

# 2. 범주별 바 그래프 (Bar Plot)
# 자동으로 평균값을 계산하고 에러바(오차 막대)를 표시함
sns.barplot(data=df, x="size", y="수익률(%)")

# 3. 다차원 관계 분석 (Relational Plot)
# hue(색상), col(열 구분), size(크기) 등을 사용하여 복합적인 관계 파악 가능
sns.relplot(
    x="PBR(IFRS-연결)", y="수익률(%)", 
    col="size", hue="베타 (M,5Yr)", 
    data=df, palette="coolwarm"
)
```

---

## 4. 학습 포인트 (퀀트 투자 관점)

1.  **시각화를 통한 데이터 검증**: 수치로만 볼 때 놓치기 쉬운 **이상치(Outlier)**나 **데이터 누락** 현상을 주가 차트나 히스토그램을 통해 즉각적으로 확인할 수 있습니다.
2.  **분포 분석의 중요성**: 수익률의 분포가 정규분포를 따르는지, 혹은 꼬리가 두꺼운(Fat-tail) 분포인지 확인하는 것은 리스크 관리의 핵심입니다 (`kind='kde'` 또는 `hist` 활용).
3.  **다차원 분석 (Hue, Col)**: 종목의 시가총액 규모(size)나 특정 지표(PBR 등)에 따라 수익률의 양상이 어떻게 달라지는지 `relplot` 등을 통해 입체적으로 분석할 수 있습니다.

---

## 5. 주요 라이브러리 함수 정리

| 라이브러리 | 클래스/모듈 | 함수명 | 설명 |
| :--- | :--- | :--- | :--- |
| **Matplotlib** | `plt` | `subplots()` | Figure와 Axes(들) 객체를 동시에 생성 |
| **Matplotlib** | `plt` | `plot()` | 선(Line) 그래프 생성 |
| **Matplotlib** | `Axes` | `set_title/xlabel/ylabel` | 그래프의 제목 및 축 라벨 설정 |
| **Matplotlib** | `Axes` | `legend()` | 범례 표시 |
| **Pandas** | `DataFrame/Series` | `plot()` | Pandas 객체 데이터를 시각화 (Matplotlib 래퍼) |
| **Pandas** | `DataFrame/Series` | `hist()` | 히스토그램 생성 |
| **Seaborn** | `sns` | `countplot()` | 범주형 데이터의 빈도를 막대 그래프로 표시 |
| **Seaborn** | `sns` | `barplot()` | 범주별 수치 데이터의 평균 및 신뢰구간 표시 |
| **Seaborn** | `sns` | `relplot()` | 두 변수 간의 관계를 다차원적으로 시각화 |
| **Seaborn** | `sns` | `heatmap()` | 데이터의 밀도나 상관관계를 색상으로 표시 |
