import json
import pandas as pd
from collections import Counter
from konlpy.tag import Okt
from tqdm import tqdm

def main():
    print("=== [1/4] 데이터 로드 ===")
    train_df = pd.read_csv("train_clean.csv").dropna()
    valid_df = pd.read_csv("valid_clean.csv").dropna()
    okt = Okt()

    print("\n=== [2/4] Train 데이터 형태소 토큰화 및 어휘 구축 ===")
    counter = Counter()
    for doc in tqdm(train_df['document'], desc="Train 토큰 빈도 집계"):
        tokens = okt.morphs(str(doc))
        counter.update(tokens)

    min_freq = 2
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for token, freq in counter.most_common():
        if freq >= min_freq:
            vocab[token] = idx
            idx += 1

    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    print(f"\nvocab.json 저장 완료 (총 어휘 수: {len(vocab):,}개)")

    print("\n=== [3/4] 문장 -> 토큰 -> ID 변환 예시 3개 ===")
    sample_docs = valid_df['document'].iloc[:3]
    for i, doc in enumerate(sample_docs, 1):
        tokens = okt.morphs(str(doc))
        ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
        print(f"[예시 {i}]")
        print(f"  원문: {doc}")
        print(f"  토큰: {tokens}")
        print(f"  ID  : {ids}\n")

    print("=== [4/4] Validation 세트 미등록어(UNK) 비율 측정 ===")
    total_valid_tokens = 0
    unk_valid_tokens = 0
    for doc in tqdm(valid_df['document'], desc="Valid UNK 비율 계산"):
        tokens = okt.morphs(str(doc))
        total_valid_tokens += len(tokens)
        unk_valid_tokens += sum(1 for t in tokens if t not in vocab)

    unk_ratio = (unk_valid_tokens / total_valid_tokens) * 100 if total_valid_tokens > 0 else 0
    print(f"\nValidation 미등록어(UNK) 비율: {unk_ratio:.2f}%")

if __name__ == "__main__":
    main()