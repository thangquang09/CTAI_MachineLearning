import torch
from transformers import AutoModel, AutoTokenizer
from torch import nn

class ResidualBlock(nn.Module):
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

class SiamesePretrainedModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.interaction_head = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.3),
            
            # Multiple residual blocks
            ResidualBlock(hidden_size, dropout=0.3),
            ResidualBlock(hidden_size, dropout=0.3),
            ResidualBlock(hidden_size, dropout=0.4),
            ResidualBlock(hidden_size, dropout=0.4),
            
            # Compression layers
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size // 4, 1)
        )

    def forward_one(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # lấy [CLS] token

    def forward(self, input_ids_A, attention_mask_A, input_ids_B, attention_mask_B, labels=None):
        vec_A = self.forward_one(input_ids_A, attention_mask_A)
        vec_B = self.forward_one(input_ids_B, attention_mask_B)

        diff = vec_A - vec_B
        prod = vec_A * vec_B
        combined_vec = torch.cat((vec_A, vec_B, diff, prod), dim=1)

        logits = self.interaction_head(combined_vec)
        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits.view(-1), labels.float())
        return (loss, logits)


class TextClassificationPretrainedModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            # Initial normalization
            nn.LayerNorm(hidden_size),
            
            # Multiple residual blocks
            ResidualBlock(hidden_size, dropout=0.1),
            ResidualBlock(hidden_size, dropout=0.1),
            ResidualBlock(hidden_size, dropout=0.2),
            ResidualBlock(hidden_size, dropout=0.2),
            
            # Gradual compression
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.LayerNorm(hidden_size // 8),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size // 8, num_labels)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # lấy [CLS] token
        logits = self.classifier(cls_output)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return (loss, logits)
    
# class CrossEncoderModel(nn.Module):
