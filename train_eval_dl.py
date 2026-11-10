import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from konlpy.tag import Okt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from models import LSTMMeanPoolClassifier, BiLSTMAttentionClassifier

class FastNSMCDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]

def pre_tokenize(df, vocab, tokenizer, desc):
    encoded_list = []
    for text in tqdm(df['document'], desc=desc):
        tokens = tokenizer.morphs(str(text))
        ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
        if not ids:
            ids = [vocab["<UNK>"]]
        encoded_list.append(torch.tensor(ids, dtype=torch.long))
    labels = torch.tensor(df['label'].values, dtype=torch.float)
    return encoded_list, labels

def collate_fn(batch):
    sequences = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded, labels

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            probs = model(texts)
            preds = torch.round(probs)
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1, all_preds, all_probs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")

    print("\n=== [1/5] 어휘 사전 및 데이터 사전 캐싱 ===")
    with open('vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    train_df = pd.read_csv("train_clean.csv").dropna().reset_index(drop=True)
    valid_df = pd.read_csv("valid_clean.csv").dropna().reset_index(drop=True)
    test_df = pd.read_csv("test_clean.csv").dropna().reset_index(drop=True)

    okt = Okt()
    train_seqs, train_labels = pre_tokenize(train_df, vocab, okt, "Train 토큰화")
    valid_seqs, valid_labels = pre_tokenize(valid_df, vocab, okt, "Valid 토큰화")
    test_seqs, test_labels = pre_tokenize(test_df, vocab, okt, "Test 토큰화")

    train_loader = DataLoader(FastNSMCDataset(train_seqs, train_labels), batch_size=64, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(FastNSMCDataset(valid_seqs, valid_labels), batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(FastNSMCDataset(test_seqs, test_labels), batch_size=64, shuffle=False, collate_fn=collate_fn)

    print("\n=== [2/5] 모델 C (LSTM + Masked Mean Pooling) 학습 및 가중치 저장 ===")
    model_c = LSTMMeanPoolClassifier(len(vocab), embed_dim=100, hidden_dim=128, output_dim=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_c.parameters(), lr=0.001)

    for epoch in range(3):
        model_c.train()
        epoch_loss = 0
        for texts, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model_c(texts)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        val_acc, val_f1, _, _ = evaluate(model_c, valid_loader, device)
        print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    torch.save(model_c.state_dict(), "best_lstm_model.pt")
    print("모델 저장 완료: best_lstm_model.pt")

    print("\n=== [3/5] 저장된 모델 C 불러오기 및 Test 세트 평가 ===")
    eval_model = LSTMMeanPoolClassifier(len(vocab), embed_dim=100, hidden_dim=128, output_dim=1).to(device)
    eval_model.load_state_dict(torch.load("best_lstm_model.pt", map_location=device))
    test_acc, test_f1, test_preds, test_probs = evaluate(eval_model, test_loader, device)

    print("\n" + "=" * 45)
    print(f"[모델 C 최종 Test 세트 성능]")
    print(f"Test Accuracy: {test_acc:.4f} | Test Macro-F1: {test_f1:.4f}")
    print("=" * 45)

    print("\n=== [4/5] Test 세트 오답 사례 10개 추출 ===")
    err_cnt = 0
    for i in range(len(test_df)):
        if int(test_df['label'].iloc[i]) != int(test_preds[i]):
            err_cnt += 1
            t_label = "긍정(1)" if int(test_df['label'].iloc[i]) == 1 else "부정(0)"
            p_label = "긍정(1)" if int(test_preds[i]) == 1 else "부정(0)"
            print(f"사례 {err_cnt}. 원문: {test_df['document'].iloc[i]}")
            print(f"   -> 정답: {t_label} | 예측: {p_label} (긍정 확률: {test_probs[i]:.4f})")
            if err_cnt == 10:
                break

    print("\n=== [5/5] 새로운 입력 문장 5개 시연 ===")
    new_reviews = [
        "배우들 연기력이 미쳤고 연출도 완벽해서 시간 가는 줄 모르고 봤네요 최고의 명작!",
        "스토리 개연성이 하나도 없고 지루해서 보다가 중간에 나왔습니다 돈 아까워요",
        "초반에는 좀 루즈한가 싶었는데 후반부 몰입감이 대박이네요 여운이 남는 영화",
        "예고편이 본편보다 재밌는 영화... 기대 많이 했는데 실망스럽네요",
        "그냥 킬링타임용으로 무난하게 볼 만합니다 엄청 좋지도 나쁘지도 않아요"
    ]
    eval_model.eval()
    with torch.no_grad():
        for idx, text in enumerate(new_reviews, 1):
            tokens = okt.morphs(text)
            ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
            in_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            prob = eval_model(in_tensor).item()
            pred = "긍정(1)" if prob >= 0.5 else "부정(0)"
            print(f"입력 {idx}: {text} -> 예측: {pred} (긍정 확률: {prob:.4f})")

if __name__ == "__main__":
    main()