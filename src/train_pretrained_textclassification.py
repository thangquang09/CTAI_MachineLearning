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

from PretrainedModel import PretrainedTextClassifier, PretrainedTextClassifierWithAttention, get_model_name
from CONFIG import BATCH_SIZE, NUM_EPOCHS, EARLY_STOPPING

parser = argparse.ArgumentParser()
parser.add_argument('--case', type=int, default=1, help='Case number (1 or 2)')
parser.add_argument('--model', type=str, default='bert-base', help='Model name (bert-base, roberta-base, etc.)')
parser.add_argument('--model_type', type=str, default='text_classifier', choices=['text_classifier', 'text_classifier_attention'], help='Model type')
parser.add_argument('--freeze_base', action='store_true', help='Freeze base transformer weights')
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
print(f"Max length: {MAX_LENGTH}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PretrainedTextDataset(Dataset):
    """Dataset for individual text classification with pretrained models"""
    def __init__(self, df, tokenizer, max_length=512):
        # Convert pair data to individual texts
        texts = []
        labels = []
        
        for _, row in df.iterrows():
            text1, text2, pair_label = row['text1'], row['text2'], row['label']
            
            # Add text1 with its label (0 means text1 is real, 1 means text2 is real)
            texts.append(str(text1))
            labels.append(1.0 if pair_label == 0 else 0.0)  # 1 if real, 0 if fake
            
            # Add text2 with its label
            texts.append(str(text2))
            labels.append(1.0 if pair_label == 1 else 0.0)  # 1 if real, 0 if fake
        
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }


class PretrainedPairDataset(Dataset):
    """Dataset for pair evaluation with pretrained models"""
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
    
    print(f"Train pairs: {len(train_df)} -> Individual texts: {len(train_df) * 2}")
    print(f"Val pairs: {len(val_df)} -> Individual texts: {len(val_df) * 2}")
    
    return train_df, val_df


# Load data
train_df, val_df = load_datasets(CASE)

# Create datasets
train_dataset = PretrainedTextDataset(train_df, tokenizer, MAX_LENGTH)  # Individual texts
val_dataset = PretrainedTextDataset(val_df, tokenizer, MAX_LENGTH)     # Individual texts
val_pair_dataset = PretrainedPairDataset(val_df, tokenizer, MAX_LENGTH)  # Pairs for evaluation

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_pair_dataloader = DataLoader(val_pair_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Create model
if args.model_type == 'text_classifier':
    model = PretrainedTextClassifier(
        model_name=MODEL_NAME,
        output_dim=1,
        freeze_base=args.freeze_base
    )
elif args.model_type == 'text_classifier_attention':
    model = PretrainedTextClassifierWithAttention(
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


def predict_pair_confidence(model, batch):
    """Predict which text is REAL using confidence-based approach"""
    model.eval()
    with torch.no_grad():
        input_ids1 = batch['input_ids1']
        attention_mask1 = batch['attention_mask1']
        input_ids2 = batch['input_ids2']
        attention_mask2 = batch['attention_mask2']
        
        # Get predictions for both texts
        logit1 = model(input_ids1, attention_mask1)
        logit2 = model(input_ids2, attention_mask2)
        
        # Convert to probabilities
        prob1 = torch.sigmoid(logit1)
        prob2 = torch.sigmoid(logit2)
        
        predictions = []
        
        for i in range(len(prob1)):
            p1, p2 = prob1[i].item(), prob2[i].item()
            
            if p1 > 0.5 and p2 > 0.5:
                # Both predicted as REAL -> choose higher confidence
                pred = 0 if p1 > p2 else 1
            elif p1 < 0.5 and p2 < 0.5:
                # Both predicted as FAKE -> choose less fake (higher prob)
                pred = 0 if p1 > p2 else 1
            else:
                # Normal case: one REAL, one FAKE
                pred = 0 if p1 > 0.5 else 1
            
            predictions.append(pred)
        
        return torch.tensor(predictions, device=input_ids1.device)


def evaluate_individual(model, valid_dataloader, criterion):
    """Evaluate on individual text classification"""
    model.eval()
    total_loss = 0
    running_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in valid_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(1), labels)
            total_loss += loss.item()

            predicted = (torch.sigmoid(outputs.squeeze(1)) > 0.5).float()
            running_correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * running_correct / total
    total_loss = total_loss / len(valid_dataloader)
    return total_loss, accuracy


def evaluate_pairs(model, pair_dataloader):
    """Evaluate on original pair comparison task"""
    model.eval()
    running_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in pair_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['label']
            
            predictions = predict_pair_confidence(model, batch)
            running_correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * running_correct / total
    return accuracy


def train(
    model,
    max_epoch,
    train_dataloader,
    valid_dataloader,
    pair_dataloader,
    criterion,
    optimizer,
    scheduler,
    device,
):
    model.to(device)
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    pair_accuracies = []

    # Track both metrics
    best_val_loss = float("inf")
    best_pair_accuracy = 0.0
    best_weights_val_loss = None
    best_weights_pair_acc = None
    best_val_loss_epoch = 0
    best_pair_acc_epoch = 0

    print("Epoch | Time  | Train Acc | Train Loss | Val Acc | Val Loss | Pair Acc | Best VL | Best PA | LR")
    print("-" * 110)

    for epoch in range(max_epoch):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        epoch_start_time = time.time()

        # Training on individual texts
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(1), labels)
            running_loss += loss.item()
            
            predicted = (torch.sigmoid(outputs.squeeze(1)) > 0.5).float()
            total += labels.size(0)
            running_correct += (predicted == labels).sum().item()
            
            loss.backward()
            optimizer.step()

        # Calculate training metrics
        epoch_accuracy = 100 * running_correct / total
        epoch_loss = running_loss / len(train_dataloader)

        # Validation on individual texts
        val_loss, val_accuracy = evaluate_individual(model, valid_dataloader, criterion)
        
        # Evaluation on pairs (original task)
        pair_accuracy = evaluate_pairs(model, pair_dataloader)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Track best models
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights_val_loss = model.state_dict().copy()
            best_val_loss_epoch = epoch + 1

        if pair_accuracy > best_pair_accuracy:
            best_pair_accuracy = pair_accuracy
            best_weights_pair_acc = model.state_dict().copy()
            best_pair_acc_epoch = epoch + 1

        # Print progress
        val_loss_indicator = "🔥" if val_loss == best_val_loss else "  "
        pair_acc_indicator = "🎯" if pair_accuracy == best_pair_accuracy else "  "
        current_lr = optimizer.param_groups[0]['lr']
        
        print(
            f"{epoch + 1:5d} | {time.time() - epoch_start_time:5.2f}s | "
            f"{epoch_accuracy:8.3f}% | {epoch_loss:9.6f} | "
            f"{val_accuracy:6.3f}% | {val_loss:8.6f} | {pair_accuracy:7.3f}% | "
            f"E{best_val_loss_epoch:2d}{val_loss_indicator} | E{best_pair_acc_epoch:2d}{pair_acc_indicator} | {current_lr:.2e}"
        )

        # Store metrics
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        pair_accuracies.append(pair_accuracy)

    history = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "pair_accuracies": pair_accuracies,
        "best_val_loss": best_val_loss,
        "best_pair_accuracy": best_pair_accuracy,
        "best_val_loss_epoch": best_val_loss_epoch,
        "best_pair_acc_epoch": best_pair_acc_epoch,
    }

    return history, best_weights_val_loss, best_weights_pair_acc


# Start training
print("=" * 110)
print("STARTING TRAINING")
print("=" * 110)

history, best_weights_val_loss, best_weights_pair_acc = train(
    model,
    NUM_EPOCHS,
    train_dataloader,
    val_dataloader,
    val_pair_dataloader,
    criterion,
    optimizer,
    scheduler,
    device,
)

# Save models
os.makedirs("models", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

model_info = f"{args.model}_{args.model_type}"
if args.freeze_base:
    model_info += "_frozen"

# Model 1: Best validation loss
if best_weights_val_loss:
    model_save_path_val = os.path.join("models", f"{timestamp}_{model_info}_case{CASE}_best_val_loss.pth")
    torch.save(best_weights_val_loss, model_save_path_val)
    print(f"\n🔥 Model with best validation loss (epoch {history['best_val_loss_epoch']}) saved to:")
    print(f"   {model_save_path_val}")
    print(f"   Best val loss: {history['best_val_loss']:.6f}")

# Model 2: Best pair accuracy
if best_weights_pair_acc:
    model_save_path_pair = os.path.join("models", f"{timestamp}_{model_info}_case{CASE}_best_pair_acc.pth")
    torch.save(best_weights_pair_acc, model_save_path_pair)
    print(f"\n🎯 Model with best pair accuracy (epoch {history['best_pair_acc_epoch']}) saved to:")
    print(f"   {model_save_path_pair}")
    print(f"   Best pair accuracy: {history['best_pair_accuracy']:.3f}%")

# Create plots
os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(20, 8))

# Plot 1: Loss
plt.subplot(1, 4, 1)
plt.plot(history["train_losses"], label="Training Loss")
plt.plot(history["val_losses"], label="Validation Loss")
plt.axvline(x=history['best_val_loss_epoch']-1, color='red', linestyle='--', alpha=0.7, label=f'Best Val Loss (E{history["best_val_loss_epoch"]})')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

# Plot 2: Individual text accuracy
plt.subplot(1, 4, 2)
plt.plot(history["train_accuracies"], label="Training Accuracy")
plt.plot(history["val_accuracies"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Individual Text Classification Accuracy")
plt.legend()
plt.grid(True)

# Plot 3: Pair comparison accuracy
plt.subplot(1, 4, 3)
plt.plot(history["pair_accuracies"], label="Pair Accuracy", color='green', linewidth=2)
plt.axvline(x=history['best_pair_acc_epoch']-1, color='red', linestyle='--', alpha=0.7, label=f'Best Pair Acc (E{history["best_pair_acc_epoch"]})')
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Pair Comparison Accuracy")
plt.legend()
plt.grid(True)

# Plot 4: Best epochs comparison
plt.subplot(1, 4, 4)
epochs = list(range(1, len(history["val_losses"]) + 1))
plt.plot(epochs, history["val_losses"], label="Val Loss", color='blue')
plt.plot(epochs, [acc/100 for acc in history["pair_accuracies"]], label="Pair Acc (scaled)", color='green')
plt.axvline(x=history['best_val_loss_epoch'], color='blue', linestyle='--', alpha=0.7, label=f'Best Val Loss E{history["best_val_loss_epoch"]}')
plt.axvline(x=history['best_pair_acc_epoch'], color='green', linestyle='--', alpha=0.7, label=f'Best Pair Acc E{history["best_pair_acc_epoch"]}')
plt.xlabel("Epochs")
plt.ylabel("Normalized Values")
plt.title("Best Epochs Comparison")
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save plots
timestamp_plot = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = os.path.join("plots", f"pretrained_{model_info}_case{CASE}_{timestamp_plot}.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n📊 Plots saved to: {plot_path}")

# Final evaluation
print(f"\n{'='*60}")
print("FINAL COMPARISON:")
print(f"{'='*60}")

if best_weights_val_loss:
    model.load_state_dict(best_weights_val_loss)
    final_pair_acc_val = evaluate_pairs(model, val_pair_dataloader)
    print(f"🔥 Best Val Loss Model  (E{history['best_val_loss_epoch']:2d}): Pair Accuracy = {final_pair_acc_val:.3f}%")

if best_weights_pair_acc:
    model.load_state_dict(best_weights_pair_acc)
    final_pair_acc_pair = evaluate_pairs(model, val_pair_dataloader)
    print(f"🎯 Best Pair Acc Model (E{history['best_pair_acc_epoch']:2d}): Pair Accuracy = {final_pair_acc_pair:.3f}%")

print(f"{'='*60}")
print("✅ TRAINING COMPLETED")
print(f"{'='*60}")
