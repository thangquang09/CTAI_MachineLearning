import argparse
import datetime
import os
import time
import re
import string
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import nltk

from PretrainedModel import PretrainedPairClassifier, PretrainedSiameseModel, get_model_name
from CONFIG import BATCH_SIZE, NUM_EPOCHS

parser = argparse.ArgumentParser()
parser.add_argument('--case', type=int, default=1, help='Case number (1 or 2)')
parser.add_argument('--model', type=str, default='bert-base', help='Model name (bert-base, roberta-base, etc.)')
parser.add_argument('--model_type', type=str, default='pair_classifier', choices=['pair_classifier', 'siamese'], help='Model type')
parser.add_argument('--freeze_base', action='store_true', help='Freeze base transformer weights')
parser.add_argument('--fusion_method', type=str, default='concat', choices=['concat', 'subtract', 'cosine'], help='Fusion method for pair classifier')
parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
args = parser.parse_args()

CASE = args.case
MODEL_NAME = get_model_name(args.model)
MAX_LENGTH = args.max_length

print(f"Using model: {MODEL_NAME}")
print(f"Model type: {args.model_type}")
print(f"Case: {CASE}")
print(f"Freeze base: {args.freeze_base}")
print(f"Fusion method: {args.fusion_method}")
print(f"Max length: {MAX_LENGTH}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PretrainedPairDataset(Dataset):
    """Dataset for pretrained models with tokenizer"""
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text1 = str(row['text1'])
        text2 = str(row['text2'])
        label = row['label']
        
        # Tokenize both texts
        encoding1 = self.tokenizer(
            text1,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoding2 = self.tokenizer(
            text2,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids1': encoding1['input_ids'].squeeze(0),
            'attention_mask1': encoding1['attention_mask'].squeeze(0),
            'input_ids2': encoding2['input_ids'].squeeze(0),
            'attention_mask2': encoding2['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }


def preprocessing(text: str) -> str:
    """Preprocessing function from build_dataset_dataloader.py"""
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


def load_datasets(case):
    """Load train and validation datasets from HuggingFace"""
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("thangquang09/fake-new-imposter-hunt-in-texts")
    
    # Download nltk data if needed
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass
    
    # Load dataframes
    train_df = dataset[f"case{case}_train"].to_pandas()
    val_df = dataset[f"case{case}_validation"].to_pandas()
    
    # Clean data
    train_df.dropna(inplace=True)
    val_df.dropna(inplace=True)
    
    # Apply preprocessing
    train_df["file_1"] = train_df["file_1"].apply(preprocessing)
    train_df["file_2"] = train_df["file_2"].apply(preprocessing)
    val_df["file_1"] = val_df["file_1"].apply(preprocessing) 
    val_df["file_2"] = val_df["file_2"].apply(preprocessing)
    
    # Rename columns for consistency
    train_df = train_df.rename(columns={'file_1': 'text1', 'file_2': 'text2'})
    val_df = val_df.rename(columns={'file_1': 'text1', 'file_2': 'text2'})
    
    print(f"Train dataset: {len(train_df)} samples")
    print(f"Val dataset: {len(val_df)} samples")
    
    return train_df, val_df


# Load data
train_df, val_df = load_datasets(CASE)

# Create datasets
train_dataset = PretrainedPairDataset(train_df, tokenizer, MAX_LENGTH)
val_dataset = PretrainedPairDataset(val_df, tokenizer, MAX_LENGTH)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Create model
if args.model_type == 'pair_classifier':
    model = PretrainedPairClassifier(
        model_name=MODEL_NAME,
        output_dim=1,
        freeze_base=args.freeze_base,
        fusion_method=args.fusion_method
    )
elif args.model_type == 'siamese':
    model = PretrainedSiameseModel(
        model_name=MODEL_NAME,
        output_dim=1,
        freeze_base=args.freeze_base
    )

model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=2)

print(f"Model: {model.__class__.__name__}")
print(f"Device: {device}")
print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


def evaluate(model, valid_dataloader, criterion):
    model.eval()
    total_loss = 0
    running_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in valid_dataloader:
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            loss = criterion(outputs.squeeze(1), labels)
            total_loss += loss.item()

            predicted = (torch.sigmoid(outputs.squeeze(1)) > 0.5).float()
            running_correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * running_correct / total
    total_loss = total_loss / len(valid_dataloader)
    return total_loss, accuracy


def train(
    model,
    max_epoch,
    train_dataloader,
    valid_dataloader,
    criterion,
    optimizer,
    scheduler,
    device,
    early_stopping_patience=10,
    use_early_stopping=True,
):
    model.to(device)
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    best_weights = None
    best_test_loss = float("inf")
    epochs_no_improve = 0

    print("Epoch | Time  | Train Acc | Train Loss | Val Acc | Val Loss | LR")
    print("-" * 80)

    for epoch in range(max_epoch):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        epoch_start_time = time.time()

        for batch in train_dataloader:
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            loss = criterion(outputs.squeeze(1), labels)
            running_loss += loss.item()
            
            predicted = (torch.sigmoid(outputs.squeeze(1)) > 0.5).float()
            total += labels.size(0)
            running_correct += (predicted == labels).sum().item()
            
            loss.backward()
            optimizer.step()

        epoch_accuracy = 100 * running_correct / total
        epoch_loss = running_loss / len(train_dataloader)

        test_loss, test_accuracy = evaluate(model, valid_dataloader, criterion)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_weights = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"{epoch + 1:5d} | {time.time() - epoch_start_time:5.2f}s | "
            f"{epoch_accuracy:8.3f}% | {epoch_loss:9.6f} | "
            f"{test_accuracy:6.3f}% | {test_loss:8.6f} | {current_lr:.2e}"
        )

        scheduler.step(test_loss)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        if use_early_stopping and epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {early_stopping_patience} epochs with no improvement.")
            break

    history = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies,
    }

    return history, best_weights


# Start training
print("=" * 80)
print("STARTING TRAINING")
print("=" * 80)

history, best_weights = train(
    model,
    NUM_EPOCHS,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    scheduler,
    device,
    use_early_stopping=True,
)

# Save model
if best_weights:
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_info = f"{args.model}_{args.model_type}"
    if args.freeze_base:
        model_info += "_frozen"
    if args.model_type == 'pair_classifier':
        model_info += f"_{args.fusion_method}"
    
    model_save_path = os.path.join("models", f"{timestamp}_{model_info}_case{CASE}.pth")
    torch.save(best_weights, model_save_path)

    print(f"\n✅ Model with best validation loss saved to: {model_save_path}")
else:
    print("\n❌ Training completed, but no best model was saved.")

# Create plots
os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(15, 6))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(history["train_losses"], label="Training Loss")
plt.plot(history["test_losses"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history["train_accuracies"], label="Training Accuracy")
plt.plot(history["test_accuracies"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save plots
timestamp_plot = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = os.path.join("plots", f"pretrained_{model_info}_case{CASE}_{timestamp_plot}.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"📊 Plots saved to: {plot_path}")
print("=" * 80)
