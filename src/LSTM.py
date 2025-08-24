import torch
from torch import nn


class SiameseLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SiameseLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
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
