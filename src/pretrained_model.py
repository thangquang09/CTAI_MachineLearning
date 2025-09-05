import torch
from transformers import AutoModel, AutoTokenizer
from torch import nn

class SiamesePretrainedModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.interaction_head = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

    def forward_one(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # lấy [CLS] token

    def forward(self, input_ids_A, attention_mask_A, input_ids_B, attention_mask_B, labels=None):
        vec_A = self.forward_one(input_ids_A, attention_mask_A)
        vec_B = self.forward_one(input_ids_B, attention_mask_B)

        # 🔑 Cách họ combine features
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
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size // 4, num_labels)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # lấy [CLS] token
        logits = self.classifier(cls_output)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return (loss, logits)