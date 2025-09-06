import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Residual block for deep LSTM classifiers"""
    def __init__(self, hidden_size, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second sub-layer: linear -> dropout + residual
        x = self.linear2(x)
        x = self.dropout(x)
        x = x + residual  # Residual connection
        x = self.norm2(x)
        
        return x


class PairClassifierLSTM(nn.Module):
    """Enhanced LSTM for text pair classification with ResidualBlocks and improved regularization"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=1, 
                 num_layers=3, num_residual_blocks=3, dropout=0.4, 
                 embedding_dropout=0.3, residual_dropout=0.4):
        super(PairClassifierLSTM, self).__init__()
        
        # Embedding layer with improved regularization
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        
        # Shared LSTM encoder (using same LSTM for both texts to reduce parameters)
        self.lstm_encoder = nn.LSTM(
            embedding_dim, hidden_dim, 
            batch_first=True, bidirectional=True, 
            dropout=dropout if num_layers > 1 else 0.0, 
            num_layers=num_layers
        )
        
        # With bidirectional, hidden_dim doubles
        actual_hidden_dim = hidden_dim * 2
        
        # Add batch normalization for stability
        self.lstm_norm = nn.BatchNorm1d(actual_hidden_dim)
        
        # Interaction layer with more regularization
        # concat_features: 2 * actual_hidden_dim, diff: actual_hidden_dim, product: actual_hidden_dim, abs_diff: actual_hidden_dim
        interaction_dim = actual_hidden_dim * 5  # 2 + 1 + 1 + 1 = 5
        
        # Deep classifier with improved architecture
        self.input_projection = nn.Sequential(
            nn.Linear(interaction_dim, actual_hidden_dim),
            nn.BatchNorm1d(actual_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Reduced number of residual blocks for less overfitting
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(actual_hidden_dim, residual_dropout) 
            for _ in range(num_residual_blocks)
        ])
        
        # Progressive dimension reduction with stronger regularization
        self.classifier = nn.Sequential(
            nn.Linear(actual_hidden_dim, actual_hidden_dim // 2),
            nn.BatchNorm1d(actual_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.5),
            
            nn.Linear(actual_hidden_dim // 2, actual_hidden_dim // 4),
            nn.BatchNorm1d(actual_hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.5),
            
            nn.Linear(actual_hidden_dim // 4, actual_hidden_dim // 8),
            nn.BatchNorm1d(actual_hidden_dim // 8),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(actual_hidden_dim // 8, output_dim)
        )
    
    def encode_sequence(self, seq):
        """Encode a single sequence"""
        emb = self.embedding_dropout(self.embedding(seq))
        lstm_out, (hidden, _) = self.lstm_encoder(emb)
        
        # Use last hidden state (concat forward and backward)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        # Apply batch normalization for stability
        hidden = self.lstm_norm(hidden)
        
        return hidden
    
    def forward(self, seq1, seq2):
        # Encode both sequences using shared encoder
        h1 = self.encode_sequence(seq1)
        h2 = self.encode_sequence(seq2)
        
        # Rich interaction features
        concat_features = torch.cat([h1, h2], dim=1)
        diff_features = h1 - h2
        product_features = h1 * h2
        abs_diff_features = torch.abs(h1 - h2)
        
        # Combine all interaction features
        combined = torch.cat([concat_features, diff_features, product_features, abs_diff_features], dim=1)
        
        # Project to residual dimension
        x = self.input_projection(combined)
        
        # Pass through residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        # Final classification
        return self.classifier(x)


class TextClassificationLSTMWithAttention(nn.Module):
    """Enhanced LSTM with attention for text classification and ResidualBlocks"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=1, 
                 num_layers=3, num_residual_blocks=4, dropout=0.3,
                 embedding_dropout=0.2, residual_dropout=0.3):
        super(TextClassificationLSTMWithAttention, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, 
            batch_first=True, bidirectional=True, 
            dropout=dropout if num_layers > 1 else 0.0, 
            num_layers=num_layers
        )
        
        actual_hidden_dim = hidden_dim * 2
        
        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=actual_hidden_dim, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm after attention
        self.attention_norm = nn.LayerNorm(actual_hidden_dim)
        
        # Pooling strategies
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Deep classifier with ResidualBlocks
        # Features: attended + avg_pool + max_pool
        classifier_input_dim = actual_hidden_dim * 3
        
        self.input_projection = nn.Linear(classifier_input_dim, actual_hidden_dim)
        self.input_norm = nn.LayerNorm(actual_hidden_dim)
        
        # Multiple residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(actual_hidden_dim, residual_dropout) 
            for _ in range(num_residual_blocks)
        ])
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(actual_hidden_dim, actual_hidden_dim // 2),
            nn.LayerNorm(actual_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(actual_hidden_dim // 2, actual_hidden_dim // 4),
            nn.LayerNorm(actual_hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(actual_hidden_dim // 4, output_dim)
        )
    
    def forward(self, seq):
        # Embedding + dropout
        embedded = self.embedding_dropout(self.embedding(seq))
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(embedded)  # [batch, seq_len, hidden_dim * 2]
        
        # Multi-head self-attention
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attended_out = self.attention_norm(attended_out + lstm_out)  # Residual connection
        
        # Global pooling
        # attended_out: [batch, seq_len, hidden_dim * 2]
        # For pooling, we need [batch, hidden_dim * 2, seq_len]
        pooling_input = attended_out.transpose(1, 2)
        
        avg_pooled = self.global_avg_pool(pooling_input).squeeze(-1)  # [batch, hidden_dim * 2]
        max_pooled = self.global_max_pool(pooling_input).squeeze(-1)  # [batch, hidden_dim * 2]
        
        # Use mean of attended sequence as primary representation
        attended_mean = torch.mean(attended_out, dim=1)  # [batch, hidden_dim * 2]
        
        # Combine all representations
        combined = torch.cat([attended_mean, avg_pooled, max_pooled], dim=1)
        
        # Project to residual dimension
        x = self.input_norm(self.input_projection(combined))
        
        # Pass through residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        # Final classification
        return self.classifier(x)

