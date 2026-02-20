# 미래에셋자산운용 Backend Engineer 면접 준비

미래에셋자산운용 Platform Engineering 팀 Backend Engineer 포지션 면접 준비를 위한 체계적인 정리 공간입니다.

---

## 📅 면접 일정
- **일시**: 2026년 2월 20일 금요일 14:00 PM
- **장소**: 미래에셋자산운용 본사 (서울시 종로구 종로 33, Tower 1)

---

## 📌 목차 (Preparation Roadmap)

### [Phase 1. 기업 및 직무 분석 (Company & Role Analysis)](./01_company_info/01_company_info.md)
1. **[자산운용사 이해 및 증권사 비교](./01_company_info/01_company_info.md)**
2. **[미래에셋자산운용 기업 분석 (TIGER ETF 등)](./01_company_info/01_company_info.md)**
3. **[핵심 금융 용어 정리 (PBR, AUM 등)](./01_company_info/07_financial_terminology.md)**
4. **[대표적인 퀀트 투자 모델 (팩터/차익거래/리스크 패리티)](./01_company_info/08_quant_investment_models.md)**
5. **[퀀트 투자 6단계 프로세스 및 롱/숏 개념](./01_company_info/10_quant_investment_process.md)**
6. **[플랫폼 아키텍처 및 분산 처리 전략 (K8s/Spark/Ray)](./01_company_info/09_platform_architecture_prediction.md)**

### [Phase 2. 자기소개 및 역량 증명 (Self & Experience)](./02_self_introduction/02_self_introduction.md)
1. **[자기소개 및 지원/이직 사유 (정리본)](./02_self_introduction/02_self_introduction.md)**
2. **[제출 자기소개서 원본](./02_self_introduction/02_self_introduction_raw.md)**
3. **[경력 기술서 상세 원본](./03_technical_experience/03_career_history_raw.md)**
4. **[핵심 역량 및 주요 프로젝트 성과](./03_technical_experience/03_technical_experience.md)**
5. **[경력 기반 직무 강점 분석 (Career Alignment)](./03_technical_experience/09_career_alignment_and_value.md)**

### [Phase 3. 설계 역량 및 기술 스택 (Design & Tech Stack)](./03_technical_experience/07_design_pattern_application.md)
1. **[디자인 패턴 사례 (Strategy/Factory)](./03_technical_experience/07_design_pattern_application.md)**
2. **[전략 & 팩토리 패턴 상세 설명](./03_technical_experience/10_strategy_and_factory_patterns.md)**
3. **[객체지향 설계 (Mixin vs ABC)](./03_technical_experience/08_mixin_concept_detail.md)**
4. **[핵심 기술 상세: Apache Airflow](./06_technical_agenda/04_technical_stack_detail.md)**

### [Phase 4. 기술적 업무 상세 (Technical Agenda Deep-Dive)](./06_technical_agenda/02_job_and_architecture_overview.md)
1. **[직무의 큰 그림 및 통합 아키텍처 개요](./06_technical_agenda/02_job_and_architecture_overview.md)**
2. **[데이터 파이프라인(Data Pipeline) 상세](./06_technical_agenda/05_data_pipeline_detail.md)**
3. **[백테스팅 엔진(Backtest Engine) 상세](./06_technical_agenda/06_backtest_engine_detail.md)**
4. **[데이터 저장 및 전달 전략 (Storage Strategy)](./06_technical_agenda/09_storage_strategy_detail.md)**
5. **[인프라 기술: AWS S3 이해 및 활용](./06_technical_agenda/10_aws_s3_detail.md)**
6. **[쿠버네티스(Kubernetes) 상세 활용 시나리오](./06_technical_agenda/11_kubernetes_deep_dive.md)**
7. **[데이터 서빙 및 피처 스토어 (Feature Store)](./06_technical_agenda/12_data_serving_and_feature_store.md)**
8. **[암호화 핵심 개념 (대칭/비대칭/키 관리)](./06_technical_agenda/13_cryptography_essentials.md)**
9. **[핵심 기술 스택 요약 (LLM/Docker/K8s/Kafka/Redis)](./06_technical_agenda/14_technical_stack_essentials.md)**

### [Phase 5. 실전 프로젝트 분석 (Practical Project Case Study)](./sample_project/소형주_저PBR_퀀트_백엔드_모듈_설명.md)
1. **[소형주 저PBR 퀀트 백엔드 모듈 상세 가이드](./sample_project/소형주_저PBR_퀀트_백엔드_모듈_설명.md)**
2. **[데이터 무결성 및 시점 정렬(PIT) 구현 로직](./sample_project/소형주_저PBR_퀀트_백엔드_모듈_설명.md)**
3. **[성과 지표(CAGR/MDD) 산출 및 벡터화 연산](./sample_project/소형주_저PBR_퀀트_백엔드_모듈_설명.md)**
4. **[직관적 이해를 위한 절차지향 코드 분석](./sample_project/소형주_저PBR_퀀트_절차지향.py)**

### [Phase 6. 최종 리허설 (Final Rehearsal)](./04_expected_questions/04_expected_questions.md)
1. **[예상 질문 및 답변 가이드 (인성/기술)](./04_expected_questions/04_expected_questions.md)**
2. **[회사에 대한 역질문 리스트](./05_questions_for_company/05_questions_for_company.md)**

---

## 💡 면접 핵심 전략
- **데이터 정합성(Data Integrity)**: 21억 건 데이터 처리 경험을 바탕으로 금융 데이터의 무결성 강조.
- **인프라 확장성(Scalability)**: K8s와 Airflow를 활용한 대규모 분산 처리 설계 역량 어필.
- **도메인 융합**: 금융 로직(상환, 결제)을 기술적으로 풀어낸(디자인 패턴) 경험 강조.

---
## 💡 면접 Tip
- **전문성 어필**: AI 기반 글로벌 자산운용 플랫폼 개발에 적합한 본인의 강점 강조
- **태도**: 성실함, 겸손함, 자신감, 긍정적인 마인드 유지
- **성과 중심**: 모든 답변은 구체적인 성과와 수치를 바탕으로 설명
