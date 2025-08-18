from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.init_text_embedding import build_text_embedding
from utils.svm_kernel import get_kernel

class SVM_Model(nn.Module):
    def __init__(self,config: Dict, num_labels: int):
     
        super(SVM_Model, self).__init__()
        self.num_labels = num_labels
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.dropout=config["model"]["dropout"]
        self.max_length = config['tokenizer']['max_length']
        self.gamma = config['svm']['gamma']
        self.kernel_type=config['svm']['kernel_type']
        self.degree = config['svm']['degree']
        self.r=config['svm']['r']
        self.text_embedding = build_text_embedding(config)
        self.classifier = get_kernel(self.kernel_type, self.intermediate_dims,
                                     self.num_labels, 
                                     self.gamma, self.r, self.degree)
        self.attention_weights = nn.Linear(self.intermediate_dims, 1)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.criterion = nn.NLLLoss()

    def forward(self, id1_text: List[str], id2_text: List[str], labels: Optional[torch.LongTensor] = None):

        embedded, mask = self.text_embedding(id1_text, id2_text)
        feature_attended = self.attention_weights(torch.tanh(embedded))
        
        attention_weights = torch.softmax(feature_attended, dim=1)
        feature_attended = torch.sum(attention_weights * embedded, dim=1)
        
        # Apply dropout before classifier
        feature_attended = self.dropout_layer(feature_attended)
        
        logits = self.classifier(feature_attended)
        logits = F.log_softmax(logits, dim=-1)

        if labels is not None:
            loss = self.criterion(logits, labels)
            return logits,loss
        else:
            return logits

def createSVM_Model(config: Dict, answer_space: List[str]) ->SVM_Model:
    return SVM_Model(config, num_labels=len(answer_space))