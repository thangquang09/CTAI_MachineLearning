import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional
from mask.masking import generate_padding_mask
from data_utils.vocab import create_vocab
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


def Text_tokenizer(config: Dict):
    tokenizer = AutoTokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
    new_tokens, _ = create_vocab(config)
    new_tokens = list(set(new_tokens) - set(tokenizer.get_vocab().keys()))
    if len(new_tokens) > 0:
        tokenizer.add_tokens(new_tokens)
    return tokenizer


class Text_Embedding(nn.Module):
    def __init__(self, config: Dict, max_len: int = None):
        super(Text_Embedding, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_len if max_len is not None else config["tokenizer"]["max_length"]
        self.padding = config["tokenizer"]["padding"]
        self.truncation = config["tokenizer"]["truncation"]
        self.return_attention_mask = config["tokenizer"]["return_attention_mask"]

        # load tokenizer + add new tokens nếu cần
        tokenizer = AutoTokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
        if config["text_embedding"]["add_new_token"]:
            new_tokens, _ = create_vocab(config)
            new_tokens = set(new_tokens) - set(tokenizer.get_vocab().keys())
            tokenizer.add_tokens(list(new_tokens))
        self.tokenizer = tokenizer

        # load model
        self.embedding = AutoModel.from_pretrained(config["text_embedding"]["text_encoder"]).to(self.device)
        if config["text_embedding"]["add_new_token"]:
            self.embedding.resize_token_embeddings(len(self.tokenizer))

        # freeze pretrained nếu cần
        if config["text_embedding"]["freeze"]:
            for param in self.embedding.parameters():
                param.requires_grad = False

        # LoRA nếu cần
        if not config["text_embedding"]["freeze"] and config["text_embedding"]["use_lora"]:
            lora_config = LoraConfig(
                r=config['text_embedding']['lora_r'],
                lora_alpha=config['text_embedding']['lora_alpha'],
                lora_dropout=config['text_embedding']['lora_dropout'],
                bias="none",
                task_type=TaskType.SEQ_CLS,
            )
            self.embedding = get_peft_model(self.embedding, lora_config)

        # projection
        self.proj = nn.Linear(config["text_embedding"]['d_features'], config["text_embedding"]['d_model'])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config["text_embedding"]['dropout'])

    def forward(self, text1: List[str], text2: List[str] = None):
        # tokenize riêng từng text, max_length = 256
        enc1 = self.tokenizer(
            text1,
            padding=False,
            truncation=True,
            max_length=256,
            return_tensors="pt",
            return_attention_mask=True
        )
        input_ids1 = enc1["input_ids"]
        attention_mask1 = enc1["attention_mask"]

        if text2 is not None:
            enc2 = self.tokenizer(
                text2,
                padding=False,
                truncation=True,
                max_length=256,
                return_tensors="pt",
                return_attention_mask=True
            )
            input_ids2 = enc2["input_ids"]
            attention_mask2 = enc2["attention_mask"]

            # gộp text1 + text2 theo chiều seq_len
            input_ids = torch.cat([input_ids1, input_ids2[:, 1:]], dim=1)  # bỏ [CLS] của text2
            attention_mask = torch.cat([attention_mask1, attention_mask2[:, 1:]], dim=1)
        else:
            input_ids = input_ids1
            attention_mask = attention_mask1

        # chuyển device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        self.embedding.to(self.device)

        # forward
        outputs = self.embedding(input_ids=input_ids, attention_mask=attention_mask)
        features = outputs.last_hidden_state

        # tạo mask padding
        padding_mask = generate_padding_mask(input_ids, padding_idx=self.tokenizer.pad_token_id)

        # projection
        out = self.proj(features)
        out = self.dropout(self.gelu(out))
        return out, padding_mask
