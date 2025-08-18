import torch
from torch import nn
from transformers import T5Tokenizer, T5Model
from typing import List, Dict, Optional
from mask.masking import generate_padding_mask
from data_utils.vocab import create_vocab

# ✅ peft API mới
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


def Text_tokenizer(config: Dict):
    tokenizer = T5Tokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
    new_tokens, _ = create_vocab(config)
    new_tokens = list(set(new_tokens) - set(tokenizer.get_vocab().keys()))
    if len(new_tokens) > 0:
        tokenizer.add_tokens(new_tokens)
    return tokenizer


class T5_Embedding(nn.Module):
    """
    Thiết kế cho viT5 / T5 tiếng Anh.
    """
    def __init__(self, config: Dict, max_len: Optional[int] = None):
        super().__init__()

        text_cfg = config["text_embedding"]
        tok_cfg = config["tokenizer"]

        # --- Tokenizer & backbone ---
        if text_cfg.get("add_new_token", False):
            self.tokenizer = Text_tokenizer(config)
            self.embedding = T5Model.from_pretrained(text_cfg["text_encoder"])
            self.embedding.resize_token_embeddings(len(self.tokenizer))
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(text_cfg["text_encoder"])
            self.embedding = T5Model.from_pretrained(text_cfg["text_encoder"])

        # --- Freeze hoặc LoRA ---
        if text_cfg.get("freeze", False):
            for p in self.embedding.parameters():
                p.requires_grad = False
        else:
            if text_cfg.get("use_lora", True):
                # Chuẩn bị k-bit training (4/8 bit). Nếu model không load bằng bitsandbytes thì sẽ bỏ qua.
                try:
                    self.embedding = prepare_model_for_kbit_training(self.embedding)
                except Exception:
                    pass

                lora_kwargs = {
                    "r": text_cfg.get("lora_r", 8),
                    "lora_alpha": text_cfg.get("lora_alpha", 16),
                    "lora_dropout": text_cfg.get("lora_dropout", 0.05),
                    "bias": "none",
                    "task_type": TaskType.SEQ_2_SEQ_LM,  # với T5 hợp lý hơn SEQ_CLS
                }
                target_modules = text_cfg.get("lora_target_modules", None)
                if target_modules:
                    lora_kwargs["target_modules"] = target_modules

                lora_config = LoraConfig(**lora_kwargs)
                self.embedding = get_peft_model(self.embedding, lora_config)

        # --- Projection & layers ---
        self.proj = nn.Linear(text_cfg["d_features"], text_cfg["d_model"])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(text_cfg["dropout"])

        # --- Tokenizer runtime options ---
        self.padding = tok_cfg.get("padding", True)
        self.truncation = tok_cfg.get("truncation", True)
        self.max_length = max_len if max_len is not None else tok_cfg.get("max_length", 128)

        # Device từ backbone
        self.device = next(self.embedding.parameters()).device if next(self.embedding.parameters(), None) is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, text1: List[str], text2: Optional[List[str]] = None):
        if text2 is not None:
            # Tokenize text1 và text2 riêng biệt với max_length/2 mỗi cái để đảm bảo đóng góp bằng nhau
            half_length = self.max_length // 2
            
            # Tokenize text1 với half_length
            inputs1 = self.tokenizer(
                text1,
                max_length=half_length,
                truncation=self.truncation,
                return_tensors="pt",
                padding="max_length",  # Đảm bảo padding đến half_length
            )
            
            # Tokenize text2 với half_length
            inputs2 = self.tokenizer(
                text2,
                max_length=half_length,
                truncation=self.truncation,
                return_tensors="pt",
                padding="max_length",  # Đảm bảo padding đến half_length
            )
            
            # T5 không có CLS token, chỉ cần concatenate trực tiếp
            # Loại bỏ EOS token cuối của text1 và concatenate với text2
            input_ids1 = inputs1["input_ids"]  # [batch_size, half_length]
            input_ids2 = inputs2["input_ids"]  # [batch_size, half_length]
            
            # Tìm vị trí EOS token trong text1 để loại bỏ
            eos_token_id = self.tokenizer.eos_token_id
            
            # Tạo mask để loại bỏ EOS token cuối của text1
            batch_size = input_ids1.shape[0]
            combined_ids = []
            
            for i in range(batch_size):
                # Tìm vị trí EOS token cuối trong text1
                text1_ids = input_ids1[i]
                text2_ids = input_ids2[i]
                
                # Loại bỏ EOS token cuối của text1 nếu có
                eos_positions = (text1_ids == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    last_eos_pos = eos_positions[-1]
                    text1_ids = text1_ids[:last_eos_pos]
                
                # Concatenate text1 + text2
                combined = torch.cat([text1_ids, text2_ids])
                
                # Truncate nếu vượt quá max_length
                if len(combined) > self.max_length:
                    combined = combined[:self.max_length]
                # Pad nếu thiếu
                elif len(combined) < self.max_length:
                    pad_length = self.max_length - len(combined)
                    pad_tokens = torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=combined.dtype)
                    combined = torch.cat([combined, pad_tokens])
                
                combined_ids.append(combined)
            
            input_ids = torch.stack(combined_ids).to(self.device)
            
        else:
            inputs = self.tokenizer(
                text=text1,
                max_length=self.max_length,
                truncation=self.truncation,
                return_tensors="pt",
                padding=self.padding,
            )
            input_ids = inputs["input_ids"].to(self.device)

        # Padding mask
        padding_mask = generate_padding_mask(input_ids, padding_idx=self.tokenizer.pad_token_id)

        # Lấy hidden states từ encoder
        features = self.embedding.encoder(input_ids=input_ids).last_hidden_state

        features = self.proj(features)
        out = self.dropout(self.gelu(features))
        return out, padding_mask
