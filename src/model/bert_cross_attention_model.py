from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.init_text_embedding import build_text_embedding

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, premise, hypothesis, premise_mask=None, hypothesis_mask=None):
        batch_size = premise.size(0)
        
        # Linear transformations and reshape for multi-head attention
        Q = self.query(premise).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(hypothesis).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(hypothesis).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply hypothesis mask if provided
        if hypothesis_mask is not None:
            hypothesis_mask = hypothesis_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(hypothesis_mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(attended)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + premise)
        
        return output, attention_weights

class BERT_CrossAttention_Model(nn.Module):
    def __init__(self, config: Dict, num_labels: int):
        super(BERT_CrossAttention_Model, self).__init__()
        self.num_labels = num_labels
        self.d_model = config["model"]["d_model"]
        self.num_heads = config["model"]["num_heads"]
        self.dropout_rate = config["model"]["dropout"]
        self.max_length = config["tokenizer"]["max_length"]
        
        # Text embedding (BERT/RoBERTa)
        self.text_embedding = build_text_embedding(config)
        
        # Cross-attention layers
        self.cross_attention_p2h = CrossAttentionLayer(self.d_model, self.num_heads, self.dropout_rate)
        self.cross_attention_h2p = CrossAttentionLayer(self.d_model, self.num_heads, self.dropout_rate)
        
        # Pooling and classification
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Enhanced classifier with multiple features
        classifier_input_dim = self.d_model * 4  # premise, hypothesis, p2h_attended, h2p_attended
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model // 2, num_labels)
        )
        
        self.criterion = nn.CrossEntropyLoss()

    def separate_premise_hypothesis(self, embedded, attention_mask):
        """Separate premise and hypothesis from concatenated sequence"""
        batch_size, seq_len, hidden_dim = embedded.shape
        half_len = seq_len // 2
        
        # Split roughly in half (adjust for special tokens)
        premise = embedded[:, :half_len, :]
        hypothesis = embedded[:, half_len:, :]
        
        premise_mask = attention_mask[:, :half_len] if attention_mask is not None else None
        hypothesis_mask = attention_mask[:, half_len:] if attention_mask is not None else None
        
        return premise, hypothesis, premise_mask, hypothesis_mask

    def pool_sequence(self, sequence, mask=None):
        """Pool sequence to get fixed-size representation"""
        if mask is not None:
            # Masked average pooling
            mask_expanded = mask.unsqueeze(-1).expand_as(sequence)
            sequence_masked = sequence * mask_expanded
            sum_embeddings = torch.sum(sequence_masked, dim=1)
            sum_mask = torch.sum(mask, dim=1, keepdim=True)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            # Simple average pooling
            pooled = torch.mean(sequence, dim=1)
        
        return pooled

    def forward(self, id1_text: List[str], id2_text: List[str], labels: Optional[torch.LongTensor] = None):
        # Get BERT embeddings
        embedded, padding_mask = self.text_embedding(id1_text, id2_text)
        
        # Convert padding mask to attention mask (1 for valid tokens, 0 for padding)
        attention_mask = (padding_mask != -10e4).float() if padding_mask is not None else None
        
        # Separate premise and hypothesis
        premise, hypothesis, premise_mask, hypothesis_mask = self.separate_premise_hypothesis(
            embedded, attention_mask
        )
        
        # Cross-attention: premise attends to hypothesis
        p2h_attended, p2h_weights = self.cross_attention_p2h(
            premise, hypothesis, premise_mask, hypothesis_mask
        )
        
        # Cross-attention: hypothesis attends to premise  
        h2p_attended, h2p_weights = self.cross_attention_h2p(
            hypothesis, premise, hypothesis_mask, premise_mask
        )
        
        # Pool all representations
        premise_pooled = self.pool_sequence(premise, premise_mask)
        hypothesis_pooled = self.pool_sequence(hypothesis, hypothesis_mask)
        p2h_pooled = self.pool_sequence(p2h_attended, premise_mask)
        h2p_pooled = self.pool_sequence(h2p_attended, hypothesis_mask)
        
        # Concatenate all features
        features = torch.cat([
            premise_pooled,
            hypothesis_pooled, 
            p2h_pooled,
            h2p_pooled
        ], dim=-1)
        
        # Apply dropout and classify
        features = self.dropout(features)
        logits = self.classifier(features)
        
        if labels is not None:
            loss = self.criterion(logits, labels)
            return logits, loss
        else:
            return logits

def createBERT_CrossAttention_Model(config: Dict, answer_space: List[str]) -> BERT_CrossAttention_Model:
    return BERT_CrossAttention_Model(config, num_labels=len(answer_space))
