import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch.nn.functional as F


class PretrainedTextClassifier(nn.Module):
    """
    Text Classification using pretrained transformer models (BERT, RoBERTa, etc.)
    Similar to TextClassificationLSTM but using transformer encoder
    """
    def __init__(self, model_name="bert-base-uncased", output_dim=1, freeze_base=False):
        super(PretrainedTextClassifier, self).__init__()
        
        self.model_name = model_name
        self.freeze_base = freeze_base
        
        # Load pretrained model and config
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Get hidden size from config
        hidden_size = self.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, output_dim)
        )
        
        # Additional dropout for CLS token
        self.cls_dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass for single text classification
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use CLS token representation ([CLS] is at position 0)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        cls_output = self.cls_dropout(cls_output)
        
        # Classification
        logits = self.classifier(cls_output)
        return logits


class PretrainedPairClassifier(nn.Module):
    """
    Pair Classification using pretrained transformer models
    Similar to PairClassifier but using transformer encoder
    """
    def __init__(self, model_name="bert-base-uncased", output_dim=1, freeze_base=False, fusion_method="concat"):
        super(PretrainedPairClassifier, self).__init__()
        
        self.model_name = model_name
        self.freeze_base = freeze_base
        self.fusion_method = fusion_method  # "concat", "subtract", "cosine"
        
        # Load pretrained model
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        hidden_size = self.config.hidden_size
        
        # Determine input size for classifier based on fusion method
        if fusion_method == "concat":
            classifier_input_size = hidden_size * 4  # [h1, h2, h1-h2, h1*h2]
        elif fusion_method == "subtract":
            classifier_input_size = hidden_size * 3  # [h1, h2, h1-h2]
        elif fusion_method == "cosine":
            classifier_input_size = hidden_size * 2 + 1  # [h1, h2, cosine_sim]
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Pair classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, output_dim)
        )
        
        self.cls_dropout = nn.Dropout(0.1)
    
    def encode_text(self, input_ids, attention_mask=None):
        """Encode single text to representation vector"""
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # Use CLS token
        return outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
    
    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        """
        Forward pass for pair classification
        Args:
            input_ids1, input_ids2: [batch_size, seq_len]
            attention_mask1, attention_mask2: [batch_size, seq_len]
        """
        # Encode both texts
        h1 = self.encode_text(input_ids1, attention_mask1)
        h2 = self.encode_text(input_ids2, attention_mask2)
        
        # Apply dropout
        h1 = self.cls_dropout(h1)
        h2 = self.cls_dropout(h2)
        
        # Fusion based on method
        if self.fusion_method == "concat":
            # Concatenate with interaction features (preserve direction)
            combined = torch.cat([h1, h2, h1 - h2, h1 * h2], dim=1)
        elif self.fusion_method == "subtract":
            # Simple concatenation with difference
            combined = torch.cat([h1, h2, h1 - h2], dim=1)
        elif self.fusion_method == "cosine":
            # Add cosine similarity as feature
            cosine_sim = F.cosine_similarity(h1, h2, dim=1, eps=1e-8).unsqueeze(1)
            combined = torch.cat([h1, h2, cosine_sim], dim=1)
        
        # Classification
        logits = self.classifier(combined)
        return logits


class PretrainedSiameseModel(nn.Module):
    """
    Siamese Network using pretrained transformer models
    Similar to SiameseLSTM but using transformer encoder
    """
    def __init__(self, model_name="bert-base-uncased", output_dim=1, freeze_base=False):
        super(PretrainedSiameseModel, self).__init__()
        
        self.model_name = model_name
        self.freeze_base = freeze_base
        
        # Load pretrained model
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        hidden_size = self.config.hidden_size
        
        # Siamese classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),  # Same fusion as LSTM version
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_dim)
        )
        
        self.cls_dropout = nn.Dropout(0.1)
    
    def forward_once(self, input_ids, attention_mask=None):
        """Encode single text (shared encoder)"""
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # Use CLS token representation
        return outputs.last_hidden_state[:, 0, :]
    
    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        """
        Forward pass for siamese network
        """
        # Encode both texts using shared encoder
        output1 = self.forward_once(input_ids1, attention_mask1)
        output2 = self.forward_once(input_ids2, attention_mask2)
        
        # Apply dropout
        output1 = self.cls_dropout(output1)
        output2 = self.cls_dropout(output2)
        
        # Combine features (same as SiameseLSTM)
        combined = torch.cat([
            output1, 
            output2, 
            output1 - output2, 
            output1 * output2
        ], dim=1)
        
        # Classification
        return self.classifier(combined)


class PretrainedTextClassifierWithAttention(nn.Module):
    """
    Text Classifier with custom attention over transformer outputs
    """
    def __init__(self, model_name="bert-base-uncased", output_dim=1, freeze_base=False):
        super(PretrainedTextClassifierWithAttention, self).__init__()
        
        self.model_name = model_name
        self.freeze_base = freeze_base
        
        # Load pretrained model
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        hidden_size = self.config.hidden_size
        
        # Custom attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, output_dim)
        )
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass with attention pooling
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # All hidden states: [batch_size, seq_len, hidden_size]
        hidden_states = outputs.last_hidden_state
        
        # Apply attention
        attention_weights = self.attention(hidden_states)  # [batch_size, seq_len, 1]
        
        # Mask attention weights where input is padded
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(
                attention_mask.unsqueeze(-1) == 0, float('-inf')
            )
        
        # Softmax over sequence length
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum
        attended_output = torch.sum(attention_weights * hidden_states, dim=1)
        
        # Classification
        return self.classifier(attended_output)


# Utility function to create models
def create_pretrained_model(model_type, model_name="bert-base-uncased", **kwargs):
    """
    Factory function to create pretrained models
    
    Args:
        model_type: "text_classifier", "pair_classifier", "siamese", "text_classifier_attention"
        model_name: HuggingFace model name (bert-base-uncased, roberta-base, etc.)
        **kwargs: Additional arguments for model initialization
    """
    if model_type == "text_classifier":
        return PretrainedTextClassifier(model_name=model_name, **kwargs)
    elif model_type == "pair_classifier":
        return PretrainedPairClassifier(model_name=model_name, **kwargs)
    elif model_type == "siamese":
        return PretrainedSiameseModel(model_name=model_name, **kwargs)
    elif model_type == "text_classifier_attention":
        return PretrainedTextClassifierWithAttention(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Commonly used model configurations
PRETRAINED_MODELS = {
    "bert-base": "bert-base-uncased",
    "bert-large": "bert-large-uncased",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "distilbert": "distilbert-base-uncased",
    "electra-base": "google/electra-base-discriminator",
    "electra-large": "google/electra-large-discriminator",
    "deberta-base": "microsoft/deberta-base",
    "deberta-large": "microsoft/deberta-large",
}

def get_model_name(model_key):
    """Get full model name from key"""
    return PRETRAINED_MODELS.get(model_key, model_key)


# Example usage:
"""
# Text Classification
model = PretrainedTextClassifier(model_name="bert-base-uncased", output_dim=1)

# Pair Classification  
model = PretrainedPairClassifier(model_name="roberta-base", fusion_method="concat")

# Siamese Network
model = PretrainedSiameseModel(model_name="distilbert-base-uncased")

# With attention
model = PretrainedTextClassifierWithAttention(model_name="electra-base")

# Using factory function
model = create_pretrained_model("text_classifier", "bert-base-uncased", freeze_base=True)
"""
