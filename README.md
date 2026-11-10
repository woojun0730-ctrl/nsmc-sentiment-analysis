# 한국어 영화 리뷰 감성 분석 (NSMC) 프로젝트

## 1. 과제 주제 및 데이터 출처
- **주제:** 한국어 영화 리뷰 긍정(1) / 부정(0) 이진 감성 분류
- **데이터 출처:** NSMC (Naver Sentiment Movie Corpus, https://github.com/e9t/nsmc)
- **데이터 다운로드 방법:** `preprocess.py` 실행 시 공식 GitHub 저장소에서 `ratings_train.txt`, `ratings_test.txt`를 자동으로 다운로드하고 결측치·중복 제거 후 `train_clean.csv`(116,945개), `valid_clean.csv`(29,237개), `test_clean.csv`(49,157개)를 생성합니다.

## 2. 환경 설치 및 실행 명령
```bash
# 1) 필수 패키지 설치
pip install -r requirements.txt

# 2) 데이터 다운로드 및 전처리, 분할
python preprocess.py

# 3) 형태소 기반 어휘 사전(vocab.json) 구축 및 미등록어(UNK) 비율 확인
python build_vocab.py

# 4) 기준선 모델(Baseline A: Naive Bayes) 학습 및 평가
python train_nb.py

# 5) 딥러닝 One-Factor 모델(모델 C) 학습, Test 평가, 오답 분석 및 새 입력 추론
python train_eval_dl.py