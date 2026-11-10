# 한국어 영화 리뷰 감성 분석 (NSMC) 딥러닝 프로젝트

- **과목명:** 딥러닝 자연어처리
- **태스크:** 한국어 영화 리뷰 텍스트의 긍정(1) / 부정(0) 이진 감성 분류 (Binary Sentiment Classification)
- **핵심 연구 목표:**
  1. KoNLPy 형태소 분석 기반 자체 어휘 사전(`vocab.json`) 구축 및 통계적 기준선(Naive Bayes) 수립
  2. 선택 딥러닝 모델(기본 LSTM)의 패딩 손실 원인 규명 및 **단일 요인(One-Factor: Masked Mean Pooling) 통제 실험**
  3. Mean Pooling의 한계를 극복하기 위한 **심화 확장 모델(2층 Bi-LSTM + Self-Attention)** 설계 및 추가 일반화 성능 검증

---

## 1. 데이터셋 및 전처리 파이프라인

- **데이터 출처:** NSMC (Naver Sentiment Movie Corpus, https://github.com/e9t/nsmc)
- **정제 및 분할:**
  - 결측치(NaN) 및 리뷰 본문 중복 문장 완전 제거
  - Train:Valid = 8:2 층화 분할(Stratified Split, `random_state=42`)로 클래스 균형(1:1) 유지
- **데이터 구성:**
  - **Train 세트:** 116,945개 (어휘 구축 및 모델 가중치 학습)
  - **Validation 세트:** 29,237개 (미등록어 UNK 측정 및 모델 검증)
  - **Test 세트:** 49,157개 (공식 Test 정제본, 최종 일반화 성능 평가 및 오류 분석)
- **어휘 사전 구축 (`vocab.json`):**
  - 데이터 누수(Data Leakage) 방지를 위해 오직 Train 데이터에서만 KoNLPy `Okt` 형태소 분석기 적용
  - 최소 등장 빈도(`min_freq=2`) 필터링 및 특수 토큰(`<PAD>: 0`, `<UNK>: 1`) 정의 (총 41,681개 단어)
  - Validation 세트 미등록어(UNK) 비율: **4.45%**

---

## 2. 모델 아키텍처 및 실험 설계

1. **Baseline A (기준선 모델 - Naive Bayes)**
   - `vocab.json`을 단어 사전으로 고정한 `CountVectorizer` BoW 빈도 벡터 + `MultinomialNB`
2. **선택 모델 B (기본 LSTM - Last Hidden State)**
   - `Embedding(41681, 100)` → `LSTM(100, 128, 1층 단방향)` → `Linear(128, 1)` → `Sigmoid()`
   - **문제점 규명:** 미니배치 가변 길이 패딩(`<PAD>=0`)의 영벡터 유입으로 인해 앞단 단어 정보가 소실(희석)되어 학습 수렴 실패 (`Val Acc: 0.5749`)
3. **모델 C (One-Factor 단일 요인 통제 모델 - LSTM + Masked Mean Pooling)**
   - 모델 B와 모든 조건(임베딩 100, 은닉 128, 1층 단방향, lr 0.001, batch 64, epoch 3)을 동일하게 고정
   - **One-Factor 변경:** 시퀀스 집약 방식만 Last Hidden State에서 패딩 마스킹 평균 풀링(**Masked Mean Pooling**)으로 단일 변경하여 성능을 **+27.94%p** 향상 (`Val Acc: 0.8543`)
4. **[심화 확장 연구] 2층 Bi-LSTM + Self-Attention**
   - **연구 배경:** Mean Pooling의 구조적 한계(모든 유효 토큰에 1/N 동일 가중치 부여로 핵심 감성어가 희석되는 문제) 극복
   - **구조:** `Embedding(41681, 128)` → `Dropout(0.4)` → `2층 Bidirectional LSTM(128, 128)` → `Self-Attention Pooling` → `Linear` → `Sigmoid`
   - 문맥 속 핵심 감성 어휘에 적응형 어텐션 가중치를 부여하여 최고 성능 달성 (`Test Acc: 0.8545`)

---

## 3. 주요 비교 실험 결과표

| 모델 구분 | 모델명 및 핵심 설정 (변경 요인) | Validation Accuracy | Validation Macro-F1 | Test Accuracy | Test Macro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline A** | Naive Bayes (BoW 빈도 기반) | 0.8431 | 0.8431 | - | - |
| **선택 모델 B** | 기본 LSTM (Last Hidden State) | 0.5749 | 0.5747 | - | - |
| **모델 C (One-Factor)** | LSTM + Masked Mean Pooling | **0.8543** | **0.8543** | **0.8475** | **0.8475** |
| **[심화 추가] 모델** | 2층 Bi-LSTM + Self-Attention | **0.8591** | **0.8591** | **0.8545** | **0.8545** |

---

## 4. 환경 설치 및 단계별 실행 가이드

### 1) 실행 환경 준비

#### 방법 A. Google Colab 환경 (권장: GPU 가속 및 Java 자동 내장)
Google Colab(T4 GPU 런타임)에서는 Java가 기본 설치되어 있어 가장 빠르고 안정적으로 실행할 수 있습니다.
새 코드 셀에 아래 명령어를 실행하여 저장소를 복제하고 필수 패키지를 설치합니다:

```bash
# 저장소 복제 및 작업 디렉터리 이동
!git clone https://github.com/woojun0730-ctrl/nsmc-sentiment-analysis.git
%cd nsmc-sentiment-analysis

# 필수 라이브러리 일괄 설치
!pip install -q -r requirements.txt
```

#### 방법 B. 로컬 PC 환경 (Windows / Mac / Linux)
- **사전 요구사항:** Python 3.8 이상, **Java JDK 8 이상** (KoNLPy 실행 필수)
터미널 또는 명령 프롬프트(CMD)에서 아래 명령어를 순서대로 실행합니다:

```bash
# 1. 저장소 복제 및 디렉터리 이동
git clone https://github.com/woojun0730-ctrl/nsmc-sentiment-analysis.git
cd nsmc-sentiment-analysis

# 2. 가상환경 생성 및 활성화 (권장)
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac / Linux:
source venv/bin/activate

# 3. 필수 라이브러리 일괄 설치
pip install -r requirements.txt
```

---

### 2) 단계별 파이프라인 실행

데이터 다운로드부터 최종 모델 학습 및 평가까지 순서대로 실행합니다:

```bash
# 1단계: 원본 NSMC 데이터 자동 다운로드, 결측치·중복 정제 및 8:2 층화 분할
python preprocess.py

# 2단계: 형태소 어휘 사전(vocab.json) 구축 및 Validation UNK 비율(4.45%) 측정
python build_vocab.py

# 3단계: 기준선 모델 (Baseline A: Naive Bayes) 학습 및 평가
python train_nb.py

# 4단계: 딥러닝 파이프라인 통합 실행
#  - One-Factor 모델 C (LSTM + Masked Mean Pooling) 학습 및 Test 평가
#  - Test 세트 오답 사례 10개 및 신규 문장 5개 실시간 추론 시연
#  - [심화 연구] 2층 Bi-LSTM + Self-Attention 모델 순차 학습 및 Test 최종 평가
python train_eval_dl.py
```

> **Google Colab 일괄 실행 팁:**  
> 코랩 단일 셀에서 전체 파이프라인을 한 번에 실행하려면 아래 명령어를 실행하세요.
> ```bash
> !git clone https://github.com/woojun0730-ctrl/nsmc-sentiment-analysis.git
> %cd nsmc-sentiment-analysis
> !pip install -q -r requirements.txt
> !python preprocess.py
> !python build_vocab.py
> !python train_nb.py
> !python train_eval_dl.py
> ```

---

## 5. 저장소 디렉터리 및 파일 구성

```text
├── preprocess.py        # 데이터 다운로드, 결측치/중복 정제 및 8:2 층화 분할
├── build_vocab.py       # Okt 형태소 토큰화 및 어휘 사전(vocab.json) 구축
├── train_nb.py          # Baseline A (Naive Bayes) 학습 및 검증
├── models.py            # 모델군 정의 (BasicLSTM, LSTMMeanPool, BiLSTMAttention)
├── train_eval_dl.py     # 모델 C 및 심화 모델 학습, Test 평가, 오답/새문장 분석 파이프라인
├── vocab.json           # Train 세트 기반 최종 어휘 사전 파일 (41,681개)
├── requirements.txt     # 의존 패키지 목록 (torch, konlpy, scikit-learn 등)
├── README.md            # 프로젝트 개요 및 환경별 실행 가이드
├── changes.md           # 직접 구현 내역, One-Factor 실험 분석 및 AI 활용 명세서
└── .gitignore           # 대용량 원본 데이터, 모델 가중치(*.pt), 가상환경 제외 규칙
```