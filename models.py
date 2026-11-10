import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicLSTMClassifier(nn.Module):
    """선택 모델 B: 시퀀스의 마지막 타임스텝 Hidden State만을 사용하는 기본 LSTM"""
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, output_dim=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, _) = self.lstm(embedded)
        hidden = hidden.squeeze(0)
        out = self.fc(hidden)
        return self.sigmoid(out).squeeze(1)


class BiLSTMAttentionClassifier(nn.Module):
    """모델 C (최종): 2층 양방향 LSTM과 Self-Attention 가중치 부여 풀링이 결합된 모델"""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, output_dim=1, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        self.attn_fc = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, _ = self.lstm(embedded)

        # 1. Attention 스코어 계산
        attn_scores = self.attn_fc(output).squeeze(-1)

        # 2. 패딩(<PAD>=0) 마스킹 처리
        mask = (text != 0)
        attn_scores = attn_scores.masked_fill(~mask, -1e9)

        # 3. Softmax 확률 부여 및 가중치 합산
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
        context_vector = torch.sum(output * attn_weights, dim=1)

        out = self.fc(self.dropout(context_vector))
        return self.sigmoid(out).squeeze(1)