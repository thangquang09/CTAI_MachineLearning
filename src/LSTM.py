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


class TextClassificationLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=1, num_layers=3):
        super(TextClassificationLSTM, self).__init__()
        
        # Embedding layer với padding_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM với dropout
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True, 
            dropout=0.3 if num_layers > 1 else 0.0,
            num_layers=num_layers
        )
        
        # Với bidirectional, hidden_dim thực tế gấp đôi
        actual_hidden_dim = hidden_dim * 2
        
        # Classifier với regularization tốt
        self.classifier = nn.Sequential(
            nn.Linear(actual_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Dropout cho embedding
        self.embedding_dropout = nn.Dropout(0.2)
    
    def forward(self, seq):
        # Embedding + dropout
        embedded = self.embedding(seq)
        embedded = self.embedding_dropout(embedded)
        
        # LSTM forward pass
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Sử dụng hidden state cuối cùng (concat forward và backward)
        # hidden shape: [num_layers * num_directions, batch, hidden_dim]
        # Lấy layer cuối cùng, cả forward và backward
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        # Classification
        return self.classifier(hidden)


class TextClassificationLSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=1, num_layers=3):
        super(TextClassificationLSTMWithAttention, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True, 
            dropout=0.3 if num_layers > 1 else 0.0,
            num_layers=num_layers
        )
        
        actual_hidden_dim = hidden_dim * 2
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(actual_hidden_dim, actual_hidden_dim),
            nn.Tanh(),
            nn.Linear(actual_hidden_dim, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(actual_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.embedding_dropout = nn.Dropout(0.2)
    
    def forward(self, seq):
        # Embedding + dropout
        embedded = self.embedding(seq)
        embedded = self.embedding_dropout(embedded)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(embedded)  # [batch, seq_len, hidden_dim * 2]
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum of LSTM outputs
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)  # [batch, hidden_dim * 2]
        
        # Classification
        return self.classifier(attended_output)

