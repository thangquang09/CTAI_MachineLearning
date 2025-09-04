import re
import string
import os
import time
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import load_dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from CONFIG import PretrainedModelConfig
from pretrained_model import SiamesePretrainedModel


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
        "\U0001f1e0-\U0001f1ff"  # flags
        "\U00002702-\U000027b0"  # dingbats
        "\U000024c2-\U0001f251"  # enclosed characters
        "\U0001f926-\U0001f937"  # supplemental symbols
        "\u200d"  # zero-width joiner
        "\u2640-\u2642"  # gender symbols
        "]+",
        flags=re.UNICODE,
    )

    text = emoji_pattern.sub(r" ", text)

    text = " ".join(text.split())

    return text.lower() if text.strip() else ""


class TextPairDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        """
        df: DataFrame với các cột ['file_1', 'file_2', 'label']
        tokenizer: tokenizer của DeBERTa (AutoTokenizer.from_pretrained(...))
        max_len: độ dài tối đa khi tokenize
        """
        self.texts1 = df["file_1"].values
        self.texts2 = df["file_2"].values
        self.labels = df["label"].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, idx):
        text1 = str(self.texts1[idx])
        text2 = str(self.texts2[idx])
        label = self.labels[idx]
        
        # Chuyển đổi label từ {1, 2} thành {0, 1} cho BCE loss
        label = label - 1  # 1->0, 2->1

        # Tokenize từng văn bản
        encoding_A = self.tokenizer(
            text1,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        encoding_B = self.tokenizer(
            text2,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        # Flatten để bỏ chiều batch=1
        item = {
            "input_ids_A": encoding_A["input_ids"].flatten(),
            "attention_mask_A": encoding_A["attention_mask"].flatten(),
            "input_ids_B": encoding_B["input_ids"].flatten(),
            "attention_mask_B": encoding_B["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.float),  # BCE loss cần float
        }

        return item


def evaluate(model, valid_dataloader, device):
    model.eval()
    total_loss = 0
    running_correct = 0
    total = 0
    with torch.no_grad():
        for batch in valid_dataloader:
            input_ids_A = batch["input_ids_A"].to(device)
            attention_mask_A = batch["attention_mask_A"].to(device)
            input_ids_B = batch["input_ids_B"].to(device)
            attention_mask_B = batch["attention_mask_B"].to(device)
            labels = batch["labels"].to(device)

            loss, logits = model(
                input_ids_A, attention_mask_A, input_ids_B, attention_mask_B, labels
            )
            total_loss += loss.item()

            predicted = (torch.sigmoid(logits.squeeze(1)) > 0.5).float()
            running_correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * running_correct / total
    avg_loss = total_loss / len(valid_dataloader)
    return avg_loss, accuracy


def train_model(
    model,
    max_epoch,
    train_dataloader,
    valid_dataloader,
    optimizer,
    scheduler,
    device,
    early_stopping_patience=None,
    use_early_stopping=True,
):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    best_weights = None
    best_test_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(max_epoch):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        epoch_start_time = time.time()

        for i, batch in enumerate(train_dataloader):
            input_ids_A = batch["input_ids_A"].to(device)
            attention_mask_A = batch["attention_mask_A"].to(device)
            input_ids_B = batch["input_ids_B"].to(device)
            attention_mask_B = batch["attention_mask_B"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss, logits = model(
                input_ids_A, attention_mask_A, input_ids_B, attention_mask_B, labels
            )

            running_loss += loss.item()
            predicted = (torch.sigmoid(logits.squeeze(1)) > 0.5).float()
            total += labels.size(0)
            running_correct += (predicted == labels).sum().item()

            loss.backward()
            optimizer.step()

        epoch_accuracy = 100 * running_correct / total
        epoch_loss = running_loss / len(train_dataloader)

        test_loss, test_accuracy = evaluate(model, valid_dataloader, device)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_weights = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            "| Epoch {:3d} | Time: {:5.2f}s | Train Accuracy {:8.3f}% | Train Loss {:8.3f} "
            "| Valid Accuracy {:8.3f}% | Valid Loss {:8.3f} ".format(
                epoch + 1,
                time.time() - epoch_start_time,
                epoch_accuracy,
                epoch_loss,
                test_accuracy,
                test_loss,
            )
        )

        scheduler.step(test_loss)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        if (
            use_early_stopping
            and early_stopping_patience
            and epochs_no_improve >= early_stopping_patience
        ):
            print(
                f"\nEarly stopping triggered after {early_stopping_patience} epochs with no improvement."
            )
            break

    values_dict = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies,
    }

    return values_dict, best_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=1, help="Case number (1 or 2)")
    args = parser.parse_args()
    case = args.case

    print("\n📊 Loading dataset...")
    dataset = load_dataset("thangquang09/fake-new-imposter-hunt-in-texts")

    tokenizer = AutoTokenizer.from_pretrained(
        PretrainedModelConfig.MODEL_NAME,
        use_fast=False,
        trust_remote_code=True,
    )

    train_df = dataset[f"case{case}_train"].to_pandas()
    valid_df = dataset[f"case{case}_validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    print("🔧 Preprocessing texts...")
    train_df["file_1"] = train_df["file_1"].apply(preprocessing)
    train_df["file_2"] = train_df["file_2"].apply(preprocessing)
    valid_df["file_1"] = valid_df["file_1"].apply(preprocessing)
    valid_df["file_2"] = valid_df["file_2"].apply(preprocessing)
    test_df["file_1"] = test_df["file_1"].apply(preprocessing)
    test_df["file_2"] = test_df["file_2"].apply(preprocessing)

    print("Data Augmentation...")
    copy_train_df = train_df.copy()

    # Hoán đổi đúng cách
    f1 = copy_train_df["file_1"].copy()
    f2 = copy_train_df["file_2"].copy()
    copy_train_df["file_1"] = f2
    copy_train_df["file_2"] = f1

    # Đảo nhãn (vì labels sẽ được chuyển từ {1,2} thành {0,1} trong __getitem__)
    # Nên ta đảo từ {1,2} thành {2,1}
    copy_train_df["label"] = copy_train_df["label"].map({1: 2, 2: 1})

    # Ghép vào
    train_df = pd.concat([train_df, copy_train_df], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_dataset = TextPairDataset(train_df, tokenizer, PretrainedModelConfig.MAX_LEN)
    valid_dataset = TextPairDataset(valid_df, tokenizer, PretrainedModelConfig.MAX_LEN)
    # TODO: test_dataset = TextPairDataset(test_df, tokenizer, PretrainedModelConfig.MAX_LEN)
    train_loader = DataLoader(
        train_dataset, batch_size=PretrainedModelConfig.BATCH_SIZE, shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=PretrainedModelConfig.BATCH_SIZE, shuffle=False
    )

    print(f"Train size: {len(train_dataset)} | Valid size: {len(valid_dataset)}")

    # Initialize model
    model = SiamesePretrainedModel(PretrainedModelConfig.MODEL_NAME)
    device = torch.device(PretrainedModelConfig.DEVICE)
    model.to(device)

    # Optimizer (không cần criterion vì model đã tính loss internal)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PretrainedModelConfig.LEARNING_RATE,
        weight_decay=PretrainedModelConfig.WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=2)

    # Start training
    print("\n🚀 Starting training...")
    history, best_weights = train_model(
        model,
        PretrainedModelConfig.NUM_EPOCHS,
        train_loader,
        valid_loader,
        optimizer,
        scheduler,
        device,
        early_stopping_patience=PretrainedModelConfig.EARLY_STOPPING_PATIENCE,
        use_early_stopping=True,
    )

    # Save best model
    if best_weights:
        os.makedirs("models", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = (
            model._get_name()
            if hasattr(model, "_get_name")
            else "SiamesePretrainedModel"
        )
        model_save_path = os.path.join(
            "models", f"{model_name}_case{case}_{timestamp}.pth"
        )

        torch.save(best_weights, model_save_path)
        print(f"\nModel with best validation loss saved to: {model_save_path}")
    else:
        print("\nTraining completed, but no best model was saved.")