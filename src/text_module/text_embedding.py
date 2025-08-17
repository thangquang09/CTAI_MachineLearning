import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional
from mask.masking import generate_padding_mask
from data_utils.vocab import create_vocab

# ✅ peft API mới
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


def Text_tokenizer(config: Dict):
    tokenizer = AutoTokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
    new_tokens, _ = create_vocab(config)
    # chỉ add token chưa có
    new_tokens = list(set(new_tokens) - set(tokenizer.get_vocab().keys()))
    if len(new_tokens) > 0:
        tokenizer.add_tokens(new_tokens)
    return tokenizer


class Text_Embedding(nn.Module):
    """
    Thiết kế cho phobert, xlm-roberta, viberberta, bartpho; model tiếng Anh cũng hỗ trợ.
    """
    def __init__(self, config: Dict, max_len: Optional[int] = None):
        super().__init__()

        text_cfg = config["text_embedding"]
        tok_cfg = config["tokenizer"]

        # --- Tokenizer & backbone ---
        if text_cfg.get("add_new_token", False):
            self.tokenizer = Text_tokenizer(config)
            self.embedding = AutoModel.from_pretrained(text_cfg["text_encoder"])
            # resize khi đã add token mới
            self.embedding.resize_token_embeddings(len(self.tokenizer))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(text_cfg["text_encoder"])
            self.embedding = AutoModel.from_pretrained(text_cfg["text_encoder"])

        # --- Freeze hoặc LoRA ---
        if text_cfg.get("freeze", False):
            for p in self.embedding.parameters():
                p.requires_grad = False
        else:
            # Dùng LoRA nếu bật
            if text_cfg.get("use_lora", True):
                # Chuẩn bị cho k-bit training (4/8-bit) – an toàn cho peft mới
                # Gợi ý: nếu bạn load model với load_in_8bit|4bit, hàm này là cần thiết.
                try:
                    self.embedding = prepare_model_for_kbit_training(self.embedding)
                except Exception:
                    # Nếu không load k-bit (hoặc không cần), bỏ qua bước này cũng OK
                    pass

                lora_kwargs = {
                    "r": text_cfg.get("lora_r", 8),
                    "lora_alpha": text_cfg.get("lora_alpha", 16),
                    "lora_dropout": text_cfg.get("lora_dropout", 0.05),
                    "bias": "none",
                    "task_type": TaskType.SEQ_CLS,
                }
                # Cho phép cấu hình target_modules nếu cần (tùy backbone)
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
        self.return_attention_mask = tok_cfg.get("return_attention_mask", True)
        self.max_length = max_len if max_len is not None else tok_cfg.get("max_length", 128)

        # Lấy device từ chính model (đảm bảo đồng bộ)
        self.device = next(self.embedding.parameters()).device if next(self.embedding.parameters(), None) is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, text: List[str], text2: Optional[List[str]] = None):
        if text2 is not None:
            inputs = self.tokenizer(
                text, text2,
                padding=self.padding,
                max_length=self.max_length,
                truncation=self.truncation,
                return_tensors="pt",
                return_attention_mask=self.return_attention_mask,
            )
        else:
            inputs = self.tokenizer(
                text=text,
                padding=self.padding,
                max_length=self.max_length,
                truncation=self.truncation,
                return_tensors="pt",
                return_attention_mask=self.return_attention_mask,
            )

        # Đưa inputs lên đúng device của model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Lấy hidden states
        outputs = self.embedding(**inputs)
        # Một số backbone trả về BaseModelOutput với .last_hidden_state
        features = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

        # Padding mask (True ở vị trí padding)
        pad_token_id = self.tokenizer.pad_token_id
        padding_mask = generate_padding_mask(inputs["input_ids"], padding_idx=pad_token_id)

        # Project
        out = self.proj(features)
        out = self.dropout(self.gelu(out))
        return out, padding_mask
