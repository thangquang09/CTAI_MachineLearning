import re
import string

# count: words and vocabulary
from collections import Counter, OrderedDict
from itertools import chain

import pandas as pd
import torch
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader, Dataset
import nltk

MAX_LEN = 600


# 4. Tạo từ điển (vocabulary)
# Tạo một class đơn giản để mô phỏng đối tượng vocabulary của torchtext
class SimpleVocab:
    def __init__(self, tokens, specials):
        self.specials = specials
        self.itos = specials + tokens  # itos: index-to-string
        self.stoi = {
            token: i for i, token in enumerate(self.itos)
        }  # stoi: string-to-index
        self.unk_index = self.stoi.get("<unk>", None)

    def __getitem__(self, token):
        # Trả về index của token, hoặc unk_index nếu không tìm thấy
        return self.stoi.get(token, self.unk_index)

    def __len__(self):
        return len(self.itos)

    def set_default_index(self, index):
        # Đảm bảo hàm getitem hoạt động đúng khi token không tồn tại
        self.unk_index = index


class TextComparisonDataset(Dataset):
    def __init__(self, dataframe, vocab):
        self.dataframe = dataframe
        self.vocab = vocab

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text1 = self.dataframe.iloc[idx]["file_1"]
        text2 = self.dataframe.iloc[idx]["file_2"]
        label = self.dataframe.iloc[idx]["label"]

        seq1 = text_to_sequence(text1, self.vocab)
        seq2 = text_to_sequence(text2, self.vocab)

        label = torch.tensor(label - 1, dtype=torch.float)

        return seq1, seq2, label


def preprocessing(text: str) -> str:
    text = text.replace("\n", " ")

    for char in string.punctuation:
        text = text.replace(char, " ")

    url_pattern = re.compile(r"https?://\s+\wwww\.\s+")
    text = url_pattern.sub(r" ", text)

    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U0001f1f2-\U0001f1f4"  # Macau flag
        "\U0001f1e6-\U0001f1ff"  # flags
        "\U0001f600-\U0001f64f"
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "\U0001f926-\U0001f937"
        "\U0001f1f2"
        "\U0001f1f4"
        "\U0001f620"
        "\u200d"
        "\u2640-\u2642"
        "]+",
        flags=re.UNICODE,
    )

    text = emoji_pattern.sub(r" ", text)

    text = " ".join(text.split())

    return text.lower()


def yield_token(sequences):
    for sequence in sequences:
        yield word_tokenize(sequence)


def text_to_sequence(text, vocab):
    tokens = word_tokenize(text)
    seq = [vocab[token] for token in tokens]

    if len(seq) < MAX_LEN:
        seq += [vocab["<pad>"]] * (MAX_LEN - len(seq))
    else:
        seq = seq[:MAX_LEN]
    return torch.tensor(seq, dtype=torch.long)


def get_dataset(case: int = 1):
    dataset = load_dataset("thangquang09/fake-new-imposter-hunt-in-texts")

    nltk.download('punkt')
    train_df = dataset[f"case{case}_train"].to_pandas()
    val_df = dataset[f"case{case}_validation"].to_pandas()

    train_df.dropna(inplace=True)
    val_df.dropna(inplace=True)

    train_df["file_1"] = train_df["file_1"].apply(preprocessing)
    train_df["file_2"] = train_df["file_2"].apply(preprocessing)

    val_df["file_1"] = val_df["file_1"].apply(preprocessing)
    val_df["file_2"] = val_df["file_2"].apply(preprocessing)

    # 1. Lấy tất cả các token và đếm tần suất
    all_tokens = chain.from_iterable(
        yield_token(pd.concat([train_df["file_1"], train_df["file_2"]]))
    )
    token_counts = Counter(all_tokens)

    # 2. Sắp xếp theo tần suất giảm dần
    sorted_by_freq = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

    # 3. Lọc theo min_freq và giới hạn kích thước từ vựng
    min_freq = 1
    vocab_size = 9125
    specials = ["<pad>", "<s>", "<unk>"]

    # Lấy các từ đủ điều kiện
    ordered_tokens = [token for token, freq in sorted_by_freq if freq >= min_freq]

    # Giới hạn kích thước, trừ đi số lượng token đặc biệt sẽ được thêm vào
    ordered_tokens = ordered_tokens[: vocab_size - len(specials)]

    # Khởi tạo vocabulary
    vocabulary = SimpleVocab(ordered_tokens, specials)
    vocabulary.set_default_index(vocabulary["<unk>"])

    print(f"Kích thước từ vựng: {len(vocabulary)}")
    print(f"Index của '<pad>': {vocabulary['<pad>']}")
    print(f"Index của một từ ngẫu nhiên 'hello': {vocabulary['hello']}")
    print(f"Index của một từ không có trong từ điển: {vocabulary['từ_không_tồn_tại']}")

    train_dataset = TextComparisonDataset(train_df, vocabulary)
    val_dataset = TextComparisonDataset(val_df, vocabulary)

    return train_dataset, val_dataset, vocabulary
