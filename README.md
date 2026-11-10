# 한국어 영화 리뷰 감성 분석 (NSMC) 딥러닝 프로젝트

- **과목명:** 딥러닝 자연어처리
- **태스크:** 한국어 영화 리뷰 텍스트의 긍정(1) / 부정(0) 이진 감성 분류 (Binary Sentiment Classification)
- **핵심 목표:** 
  1. KoNLPy 형태소 기반 자체 어휘 사전(`vocab.json`) 구축 및 통계적 기준선(Naive Bayes) 수립
  2. 선택 딥러닝 모델(기본 LSTM)의 패딩 손실 원인 규명 및 **단일 요인(One-Factor: Masked Mean Pooling) 통제 실험**
  3. Mean Pooling의 한계를 극복하기 위한 **심화 확장 모델(2층 Bi-LSTM + Self-Attention)** 설계 및 추가 성능 검증

---

## 1. 데이터셋 및 전처리 파이프라인

- **데이터 출처:** NSMC (Naver Sentiment Movie Corpus, https://github.com/e9t/nsmc)
- **정제 및 분할:**
  - 결측치(NaN) 및 중복 리뷰를 완전히 제거
  - Train:Valid = 8:2 층화 분할(Stratified Split, `random_state=42`) 적용
- **분할별 데이터 수:**
  - **Train:** 116,945개 (어휘 구축 및 모델 가중치 학습)
  - **Validation:** 29,237개 (미등록어 UNK 측정 및 모델 검증)
  - **Test:** 49,157개 (공식 Test 정제본, 최종 일반화 성능 평가 및 오류 분석)
- **어휘 사전 (`vocab.json`):**
  - Train 세트에서만 `Okt` 형태소 분석기를 적용하여 빈도 집계 (Data Leakage 방지)
  - 최소 등장 빈도(`min_freq=2`) 필터링 및 특수 토큰(`<PAD>: 0`, `<UNK>: 1`) 정의 (총 41,681개 어휘)
  - Validation 세트 미등록어(UNK) 비율: **4.45%**

---

## 2. 모델 아키텍처 및 실험 설계

1. **Baseline A (기준선 모델 - Naive Bayes)**
   - 구축된 `vocab.json`을 단어 사전으로 고정한 `CountVectorizer` BoW 빈도 벡터 + `MultinomialNB`
2. **선택 모델 B (기본 LSTM - Last Hidden State)**
   - `Embedding(41681, 100)` → `LSTM(100, 128, 1층 단방향)` → `Linear(128, 1)` → `Sigmoid()`
   - **문제점:** 미니배치 내 가변 길이 문장의 후반부 `<PAD>(0)` 영벡터 유입으로 앞단 정보가 희석(Dilution)되어 학습 실패 (`Val Acc: 0.5749`)
3. **모델 C (단일 요인 변경 모델 - LSTM + Masked Mean Pooling)**
   - 모델 B와 모든 하이퍼파라미터(임베딩 100, 은닉 128, 1층 단방향, lr 0.001, batch 64, epoch 3)를 완벽히 고정
   - **One-Factor 변경:** 시퀀스 집약 방식을 Last Hidden State에서 패딩 마스킹 평균 풀링(**Masked Mean Pooling**)으로 단일 변경하여 성능을 **+27.94%p** 비약적 개선 (`Val Acc: 0.8543`)
4. **[심화 확장 모델] 2층 Bi-LSTM + Self-Attention**
   - **연구 배경:** Mean Pooling의 구조적 한계(모든 유효 토큰에 1/N 동일 가중치 부여로 핵심 감성어가 희석되는 문제) 극복
   - **구조:** `Embedding(41681, 128)` → `Dropout(0.4)` → `2층 Bidirectional LSTM(128, 128)` → `Self-Attention Pooling` → `Linear` → `Sigmoid`
   - 문맥 속 핵심 감성 단어에 적응형 가중치를 부여하여 최고 성능 달성 (`Test Acc: 0.8545`)

---

## 3. 주요 실험 결과 비교

| 모델 구분 | 모델명 및 핵심 설정 | Validation Accuracy | Validation Macro-F1 | Test Accuracy | Test Macro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline A** | Naive Bayes (BoW 빈도 기반) | 0.8431 | 0.8431 | - | - |
| **선택 모델 B** | 기본 LSTM (Last Hidden State) | 0.5749 | 0.5747 | - | - |
| **모델 C (One-Factor)** | LSTM + Masked Mean Pooling | **0.8543** | **0.8543** | **0.8475** | **0.8475** |
| **[심화 추가] 모델** | 2층 Bi-LSTM + Self-Attention | **0.8591** | **0.8591** | **0.8545** | **0.8545** |

---

## 4. 환경 설치 및 단계별 실행 가이드

### 1) 환경 구성
```bash
pip install -r requirements.txt
