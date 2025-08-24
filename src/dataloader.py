import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import nltk


train_df = pd.read_csv("./data/data_for_dl/case1_train.csv")
valid_df = pd.read_csv("./data/data_for_dl/case1_valid.csv")


def preprocessing_text(text: str) -> str:
    text = text.lower()
    # thay the tat ca so bang token num
    text = re.sub(r'\d+', 'num', text)
    # loai bo cac dau
    text = re.sub(r'[^\w\s]', '', text)
    
    text = text.strip()
    return nltk.word_tokenize(text)

# 1. Build Vocabulary
word_counts = Counter()
all_files = list(train_df['file_1']) + list(train_df['file_2'])

for file in all_files:
    tokens = preprocessing_text(file)
    word_counts.update(tokens)

# Tạo map từ từ sang số (thêm token cho padding và từ không biết)
vocab = {word: i+2 for i, word in enumerate(word_counts)}
vocab['<pad>'] = 0
vocab['<unk>'] = 1
vocab_size = len(vocab)
print(f"\nKích thước Vocabulary: {vocab_size}")


# print(preprocessing_text(train_df['file_1'][0]))