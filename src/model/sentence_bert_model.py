from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.init_text_embedding import build_text_embedding

class SentenceBERT_Model(nn.Module):
    def __init__(self, config: Dict, num_labels: int):
        super(SentenceBERT_Model, self).__init__()
        self.num_labels = num_labels
        self.d_model = config["text_embedding"]["d_model"]
        self.dropout_rate = config["model"]["dropout"]
        
        # Text embedding (BERT/RoBERTa)
        self.text_embedding = build_text_embedding(config)
        
        # Sentence pooling method
        self.pooling_method = config["model"].get("pooling_method", "mean")  # mean, max, cls
        
        # Enhanced classifier with multiple similarity features
        # Features: [u, v, |u-v|, u*v] where u=premise, v=hypothesis
        classifier_input_dim = self.d_model * 4
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(), 
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model // 2, num_labels)
        )
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.criterion = nn.CrossEntropyLoss()

    def pool_sentence(self, embeddings, attention_mask, method="mean"):
        """Pool sentence embeddings using different strategies"""
        if method == "cls":
            # Use [CLS] token (first token)
            return embeddings[:, 0, :]
        
        elif method == "max":
            # Max pooling
            embeddings = embeddings.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
            return torch.max(embeddings, dim=1)[0]
        
        else:  # mean pooling (default)
            # Masked average pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(embeddings)
            embeddings_masked = embeddings * mask_expanded
            sum_embeddings = torch.sum(embeddings_masked, dim=1)
            sum_mask = torch.sum(attention_mask, dim=1, keepdim=True)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask

    def get_sentence_embeddings(self, text_list):
        """Get sentence embeddings for a list of texts"""
        # Tokenize and encode single sentences
        embedded, padding_mask = self.text_embedding(text_list, None)
        
        # Convert padding mask to attention mask
        attention_mask = (padding_mask != -10e4).float() if padding_mask is not None else None
        
        # Pool to get sentence representation
        sentence_emb = self.pool_sentence(embedded, attention_mask, self.pooling_method)
        
        return sentence_emb

    def forward(self, id1_text: List[str], id2_text: List[str], labels: Optional[torch.LongTensor] = None):
        # Get sentence embeddings separately
        premise_emb = self.get_sentence_embeddings(id1_text)  # [batch, d_model]
        hypothesis_emb = self.get_sentence_embeddings(id2_text)  # [batch, d_model]
        
        # Compute similarity features (SBERT approach)
        # Feature 1: premise embedding
        u = premise_emb
        
        # Feature 2: hypothesis embedding  
        v = hypothesis_emb
        
        # Feature 3: element-wise difference |u - v|
        diff = torch.abs(u - v)
        
        # Feature 4: element-wise product u * v
        prod = u * v
        
        # Concatenate all features
        features = torch.cat([u, v, diff, prod], dim=-1)  # [batch, d_model * 4]
        
        # Apply dropout and classify
        features = self.dropout(features)
        logits = self.classifier(features)
        
        if labels is not None:
            loss = self.criterion(logits, labels)
            return logits, loss
        else:
            return logits

def createSentenceBERT_Model(config: Dict, answer_space: List[str]) -> SentenceBERT_Model:
    return SentenceBERT_Model(config, num_labels=len(answer_space))
