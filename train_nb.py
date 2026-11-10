import json
import pandas as pd
from konlpy.tag import Okt
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

def main():
    print("=== [1/3] 어휘 사전 및 데이터 로드 ===")
    with open("vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_words = list(vocab.keys())[2:]

    train_df = pd.read_csv("train_clean.csv").dropna()
    valid_df = pd.read_csv("valid_clean.csv").dropna()
    okt = Okt()

    def tokenize_and_join(texts, desc):
        return [" ".join(okt.morphs(str(text))) for text in tqdm(texts, desc=desc)]

    print("\n=== [2/3] 형태소 분석 및 BoW 벡터화 ===")
    X_train_str = tokenize_and_join(train_df['document'], "Train 형태소 분리")
    X_valid_str = tokenize_and_join(valid_df['document'], "Valid 형태소 분리")

    vectorizer = CountVectorizer(vocabulary=vocab_words, lowercase=False)
    X_train = vectorizer.fit_transform(X_train_str)
    X_valid = vectorizer.transform(X_valid_str)

    y_train = train_df['label']
    y_valid = valid_df['label']

    print("\n=== [3/3] Naive Bayes 모델 학습 및 평가 ===")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    y_pred = nb_model.predict(X_valid)
    acc = accuracy_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred, average='macro')

    print("\n" + "=" * 45)
    print("[Baseline A: Naive Bayes 검증 세트 평가]")
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation Macro-F1: {f1:.4f}")
    print("=" * 45)

if __name__ == "__main__":
    main()