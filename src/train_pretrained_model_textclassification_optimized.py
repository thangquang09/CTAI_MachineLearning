import re
import string
import os
import time
import argparse
import gc
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score

from CONFIG import PretrainedModelConfig
from pretrained_model import TextClassificationPretrainedModel


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


def build_text_classification_data(df):
    """Convert text pair data to text classification data"""
    new_data = []
    new_labels = []

    for _, row in df.iterrows():
        text1 = row["file_1"]
        text2 = row["file_2"]
        label = row["label"]

        if label == 1:
            new_data.append(text1)
            new_labels.append(1)  # real
            new_data.append(text2)
            new_labels.append(0)  # fake
        else:  # label == 2
            new_data.append(text1)
            new_labels.append(0)  # fake
            new_data.append(text2)
            new_labels.append(1)  # real

    return pd.DataFrame({"text": new_data, "label": new_labels})


class TextClassificationDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        """
        df: DataFrame với các cột ['text', 'label']
        tokenizer: tokenizer của DeBERTa (AutoTokenizer.from_pretrained(...))
        max_len: độ dài tối đa khi tokenize
        """
        self.texts = df["text"].values
        self.labels = df["label"].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize văn bản
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        # Flatten để bỏ chiều batch=1
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),  # CrossEntropy loss cần long
        }

        return item


def evaluate(model, valid_dataloader, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in valid_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, logits = model(input_ids, attention_mask, labels)
            
            # Handle DataParallel loss (average across GPUs)
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            
            total_loss += loss.item()

            predicted = torch.argmax(logits, dim=1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
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
    # Enable multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs for training!")
        model = torch.nn.DataParallel(model)
    else:
        print(f"🔧 Using single GPU: {device}")
    
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    best_weights = None
    best_test_loss = float("inf")
    epochs_no_improve = 0
    
    # Get gradient accumulation steps from config
    accumulation_steps = getattr(PretrainedModelConfig, 'GRADIENT_ACCUMULATION_STEPS', 1)
    print(f"📊 Gradient accumulation steps: {accumulation_steps}")
    print(f"📊 Effective batch size: {PretrainedModelConfig.BATCH_SIZE * accumulation_steps}")

    for epoch in range(max_epoch):
        model.train()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        epoch_start_time = time.time()
        
        # Zero gradients at the beginning of each epoch
        optimizer.zero_grad()

        for i, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, logits = model(input_ids, attention_mask, labels)
            
            # Handle DataParallel loss (average across GPUs)
            if torch.cuda.device_count() > 1:
                loss = loss.mean()  # DataParallel returns loss for each GPU
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            
            running_loss += loss.item() * accumulation_steps
            predicted = torch.argmax(logits, dim=1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            loss.backward()
            
            # Only step optimizer every accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Handle remaining gradients if batch doesn't divide evenly
        if (len(train_dataloader) % accumulation_steps) != 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_accuracy = accuracy_score(all_labels, all_predictions) * 100
        epoch_loss = running_loss / len(train_dataloader)

        test_loss, test_accuracy = evaluate(model, valid_dataloader, device)
        test_accuracy *= 100  # Convert to percentage

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            # Handle DataParallel model state dict
            if torch.cuda.device_count() > 1:
                best_weights = model.module.state_dict()
            else:
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


def eval_text_pair(model, tokenizer, pair_df, device, max_len, make_submission=False):
    """Evaluate text pair using the trained text classification model"""
    model.eval()
    
    correct_pairs = 0
    total_pairs = len(pair_df)
    pred_labels = []
    true_labels = pair_df["label"].tolist() if not make_submission else None

    for i in range(total_pairs):
        text1 = str(pair_df.iloc[i]["file_1"])
        text2 = str(pair_df.iloc[i]["file_2"])
        
        # Tokenize both texts
        encoding1 = tokenizer(
            text1,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        encoding2 = tokenizer(
            text2,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        
        with torch.no_grad():
            # Predict for both texts
            input_ids1 = encoding1["input_ids"].to(device)
            attention_mask1 = encoding1["attention_mask"].to(device)
            input_ids2 = encoding2["input_ids"].to(device)
            attention_mask2 = encoding2["attention_mask"].to(device)
            
            _, logits1 = model(input_ids1, attention_mask1)
            _, logits2 = model(input_ids2, attention_mask2)
            
            # Get probabilities for "real" class (class 1)
            probs1 = torch.softmax(logits1, dim=1)[0, 1].item()
            probs2 = torch.softmax(logits2, dim=1)[0, 1].item()
            
            # Decision logic: text with higher "real" probability is from real author
            pred_label = 1 if probs1 > probs2 else 2
            pred_labels.append(pred_label)
            
            if not make_submission and pred_label == true_labels[i]:
                correct_pairs += 1

    if make_submission:
        return pred_labels
    
    pair_accuracy = correct_pairs / total_pairs
    return pair_accuracy


if __name__ == "__main__":
    # === MEMORY OPTIMIZATION SETUP ===
    print("🔧 Setting up memory optimization...")
    
    # Suppress DataParallel scalar gathering warning
    import warnings
    warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")
    
    # Enable memory efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        print("✅ Flash Attention enabled")
    except Exception:
        print("⚠️ Flash Attention not available")
    
    # Set memory allocation strategy
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Print GPU information
    if torch.cuda.is_available():
        print(f"🔍 Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=1, help="Case number (1 or 2)")
    parser.add_argument("--model", type=str, default=None, help="Model name to use (overrides config)")
    args = parser.parse_args()
    case = args.case
    
    # Use specified model or default from config
    model_name = args.model if args.model else PretrainedModelConfig.MODEL_NAME
    print(f"🤖 Using model: {model_name}")

    print("\n📊 Loading dataset...")
    dataset = load_dataset("thangquang09/fake-new-imposter-hunt-in-texts")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
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

    # Build text classification data (không cần augmentation)
    print("🏗️ Building text classification datasets...")
    train_textclf_df = build_text_classification_data(train_df)
    valid_textclf_df = build_text_classification_data(valid_df)

    print(f"📏 Train samples: {len(train_textclf_df)}, Valid samples: {len(valid_textclf_df)}")

    train_dataset = TextClassificationDataset(train_textclf_df, tokenizer, PretrainedModelConfig.MAX_LEN)
    valid_dataset = TextClassificationDataset(valid_textclf_df, tokenizer, PretrainedModelConfig.MAX_LEN)
    
    train_loader = DataLoader(
        train_dataset, batch_size=PretrainedModelConfig.BATCH_SIZE, shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=PretrainedModelConfig.BATCH_SIZE, shuffle=False
    )

    print(f"Train size: {len(train_dataset)} | Valid size: {len(valid_dataset)}")

    # Initialize model (2 classes: fake=0, real=1)
    print("🤖 Initializing TextClassificationPretrainedModel...")
    model = TextClassificationPretrainedModel(model_name, num_labels=2)
    
    # Enable gradient checkpointing for memory saving (disable if causing issues)
    # Note: Gradient checkpointing might conflict with DataParallel + gradient accumulation
    if hasattr(model.backbone, 'gradient_checkpointing_enable'):
        # Temporarily disable gradient checkpointing to avoid backward graph conflicts
        # model.backbone.gradient_checkpointing_enable()
        print("⚠️ Gradient checkpointing disabled to avoid conflicts")
    
    device = torch.device(PretrainedModelConfig.DEVICE)
    model.to(device)

    # Optimizer (model đã tính loss internal)
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

    # === EVALUATION ON PAIR CLASSIFICATION DURING TRAINING ===
    if best_weights:
        print("\n📊 Evaluating pair classification performance...")
        
        # Load best model (handle DataParallel wrapper)
        if hasattr(model, 'module'):
            # Remove DataParallel wrapper for evaluation
            model = model.module
        model.load_state_dict(best_weights)
        model.eval()
        
        # Evaluate on train pairs
        train_pair_accuracy = eval_text_pair(
            model, tokenizer, train_df, device, PretrainedModelConfig.MAX_LEN
        )
        print(f"🎯 Train Pair Accuracy: {train_pair_accuracy:.4f}")
        
        # Evaluate on validation pairs
        valid_pair_accuracy = eval_text_pair(
            model, tokenizer, valid_df, device, PretrainedModelConfig.MAX_LEN
        )
        print(f"🎯 Validation Pair Accuracy: {valid_pair_accuracy:.4f}")

    # Save best model
    if best_weights:
        os.makedirs("models", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = "TextClassificationPretrainedModel"
        model_save_path = os.path.join(
            "models", f"{model_name}_case{case}_{timestamp}.pth"
        )

        torch.save(best_weights, model_save_path)
        print(f"\nModel with best validation loss saved to: {model_save_path}")
    else:
        print("\nTraining completed, but no best model was saved.")

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

    timestamp_plot = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join("plots", f"textclf_pretrained_loss_accuracy_case{case}_{timestamp_plot}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Plots saved to: {plot_path}")

    # === MAKE SUBMISSION ===
    if best_weights:
        print("\n📤 Making predictions on test set...")
        
        # Model đã được load best weights ở trên rồi, không cần load lại
        # Predict on test set
        predictions = eval_text_pair(
            model, tokenizer, test_df, device, PretrainedModelConfig.MAX_LEN, make_submission=True
        )
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            "id": test_df.index,
            "real_text_id": np.array(predictions).astype(int)
        }).sort_values("id")
        
        # Create submission filename
        model_name_for_file = "TextClassificationPretrainedModel"
        safe_model_name = model_name.replace("/", "_").replace("-", "_")
        submission_filename = f"submission_case{case}_{model_name_for_file}_{safe_model_name}.csv"
        
        # Save submission
        submission.to_csv(submission_filename, index=False)
        print(f"✅ Submission saved to: {submission_filename}")
        print(f"📊 Test predictions: {len(predictions)} samples")
        print(f"📊 Prediction distribution: Class 1: {(np.array(predictions) == 1).sum()}, Class 2: {(np.array(predictions) == 2).sum()}")
        
        # === SUMMARY ===
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED - TextClassificationPretrainedModel")
        print("="*60)
        print(f"📊 Final Text Classification Accuracy: {history['test_accuracies'][-1]:.2f}%")
        print(f"🎯 Train Pair Accuracy: {train_pair_accuracy:.4f}")
        print(f"🎯 Validation Pair Accuracy: {valid_pair_accuracy:.4f}")
        print(f"📤 Submission: {submission_filename}")
        print("="*60)
    else:
        print("\n❌ Cannot make submission: no trained model available.")
        
    print("\n🎉 Training and submission completed!")
