import torch
from torch import nn


class SiameseLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SiameseLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=3
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_dim),
        )

    def forward_once(self, X):
        embedded = self.embedding(X)

        _, (hidden, _) = self.lstm(embedded)
        return torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        combined = torch.cat(
            (output1, output2, output1 - output2, output1 * output2), dim=1
        )

        return self.classifier(combined)

class PairClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=1):
        super(PairClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=0.3, num_layers=3)
        self.lstm2 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=0.3, num_layers=3)
        
        # Với bidirectional, hidden_dim sẽ gấp đôi
        actual_hidden_dim = hidden_dim * 2
        
        # Classifier để phân loại cặp - Tăng regularization
        self.classifier = nn.Sequential(
            nn.Linear(actual_hidden_dim * 4, actual_hidden_dim * 2),  # concat + difference + element-wise product
            nn.ReLU(),
            nn.Dropout(0.5),  # Tăng từ 0.3 lên 0.5
            nn.Linear(actual_hidden_dim * 2, actual_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),  # Tăng từ 0.3 lên 0.5
            nn.Linear(actual_hidden_dim, output_dim)
        )
    
    def forward(self, seq1, seq2):
        # Embed và encode sequences
        emb1 = self.embedding(seq1)
        emb2 = self.embedding(seq2)
        
        _, (h1, _) = self.lstm1(emb1)
        _, (h2, _) = self.lstm2(emb2)
        
        # Với bidirectional, concat forward và backward hidden states
        h1 = torch.cat((h1[-2, :, :], h1[-1, :, :]), dim=1)
        h2 = torch.cat((h2[-2, :, :], h2[-1, :, :]), dim=1)
        
        # Combine features như SiameseLSTM
        combined = torch.cat([h1, h2, torch.abs(h1 - h2), h1 * h2], dim=1)
        
        return self.classifier(combined)