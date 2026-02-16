import pandas as pd
import numpy as np

# 데이터 로드 및 전처리
df = pd.read_csv("미래에셋자산운용/sample_project/my_data/fin_statement_new.csv")
df = df.rename(columns={
    "P/B(Adj., FY End)": "PBR",
    "P/E(Adj., FY End)": "PER",
    "수정주가": "Price"
})

# 1단계: 소형주 마스크 생성
market_cap_limit = df.groupby("year")['시가총액'].transform(lambda x: x.quantile(0.2))
small_cap_mask = df['시가총액'] <= market_cap_limit

# 2단계: 필터링 및 선별
filtered_df = df[small_cap_mask & (df['PBR'] >= 0.2)]
selected = filtered_df.sort_values('PBR').groupby('year').head(20)

# 결과 확인 (2006년 상위 5개 예시)
print("--- 2006년 저PBR 선별 결과 (Top 5) ---")
print(selected[selected['year'] == 2006][['Name', 'year', '시가총액', 'PBR']].head(5).to_markdown(index=False))

# 결과 확인 (전체 건수 확인)
print("\n--- 연도별 선별 종목 수 ---")
print(selected.groupby('year').size())
