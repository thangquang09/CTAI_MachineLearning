from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class PretrainedNLI_Model(nn.Module):
    def __init__(self, config: Dict, num_labels: int):
        super(PretrainedNLI_Model, self).__init__()
        self.num_labels = num_labels
        self.max_length = config["tokenizer"]["max_length"]
        self.dropout_rate = config["model"]["dropout"]
        
        # Load pre-trained NLI model
        model_name = config["text_embedding"]["text_encoder"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Get actual hidden size from model
        self.hidden_size = self.bert.config.hidden_size
        
        # Simple classifier on top of [CLS] token
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, id1_text: List[str], id2_text: List[str], labels: Optional[torch.LongTensor] = None):
        # Update device
        self.device = next(self.bert.parameters()).device
        
        # Tokenize premise and hypothesis pairs
        # For NLI models, use the standard format: [CLS] premise [SEP] hypothesis [SEP]
        inputs = self.tokenizer(
            id1_text, id2_text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Forward through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation for classification
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Apply dropout and classify
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        if labels is not None:
            loss = self.criterion(logits, labels)
            return logits, loss
        else:
            return logits

def createPretrainedNLI_Model(config: Dict, answer_space: List[str]) -> PretrainedNLI_Model:
    return PretrainedNLI_Model(config, num_labels=len(answer_space))
