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

from CONFIG import PretrainedModelConfig
from pretrained_model import CrossEncoder


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


class CrossEncoderDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        """
        CrossEncoder Dataset: concat 2 texts với [SEP] token
        df: DataFrame với các cột ['file_1', 'file_2', 'label']
        tokenizer: tokenizer của model (AutoTokenizer.from_pretrained(...))
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

        # 🔗 CrossEncoder: concat 2 texts với tokenizer tự động thêm [SEP]
        # Format: [CLS] text1 [SEP] text2 [SEP] [PAD]...
        encoding = self.tokenizer(
            text1,
            text2,  # text_pair sẽ tự động thêm [SEP] giữa 2 texts
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, logits = model(input_ids, attention_mask, labels)
            
            # Handle DataParallel loss (average across GPUs)
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            
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
        running_correct = 0
        total = 0
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
                loss = loss.mean()
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

            running_loss += loss.item() * accumulation_steps
            predicted = (torch.sigmoid(logits.squeeze(1)) > 0.5).float()
            total += labels.size(0)
            running_correct += (predicted == labels).sum().item()

            loss.backward()
            
            # Only step optimizer every accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Handle remaining gradients if batch doesn't divide evenly
        if (len(train_dataloader) % accumulation_steps) != 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_accuracy = 100 * running_correct / total
        epoch_loss = running_loss / len(train_dataloader)

        test_loss, test_accuracy = evaluate(model, valid_dataloader, device)

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

    # Hoán đổi texts
    f1 = copy_train_df["file_1"].copy()
    f2 = copy_train_df["file_2"].copy()
    copy_train_df["file_1"] = f2
    copy_train_df["file_2"] = f1

    # Đảo nhãn (vì labels sẽ được chuyển từ {1,2} thành {0,1} trong __getitem__)
    copy_train_df["label"] = copy_train_df["label"].map({1: 2, 2: 1})

    # Ghép vào
    train_df = pd.concat([train_df, copy_train_df], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_dataset = CrossEncoderDataset(train_df, tokenizer, PretrainedModelConfig.MAX_LEN)
    valid_dataset = CrossEncoderDataset(valid_df, tokenizer, PretrainedModelConfig.MAX_LEN)
    
    train_loader = DataLoader(
        train_dataset, batch_size=PretrainedModelConfig.BATCH_SIZE, shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=PretrainedModelConfig.BATCH_SIZE, shuffle=False
    )

    print(f"Train size: {len(train_dataset)} | Valid size: {len(valid_dataset)}")
    # Display label distribution
    print("\n📊 Dataset Label Distribution:")
    print(f"Train labels - Class 1: {(train_df['label'] == 1).sum()} | Class 2: {(train_df['label'] == 2).sum()}")
    print(f"Valid labels - Class 1: {(valid_df['label'] == 1).sum()} | Class 2: {(valid_df['label'] == 2).sum()}")

    # Initialize model
    print("🤖 Initializing CrossEncoder with Deep Classifier...")
    model = CrossEncoder(PretrainedModelConfig.MODEL_NAME, num_labels=1)
    
    # Disable gradient checkpointing to avoid conflicts with DataParallel + gradient accumulation
    if hasattr(model.backbone, 'gradient_checkpointing_enable'):
        print("⚠️ Gradient checkpointing disabled to avoid conflicts")
    
    device = torch.device(PretrainedModelConfig.DEVICE)
    model.to(device)

    # Optimizer
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
        model_name = "CrossEncoder"
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
    plot_path = os.path.join("plots", f"cross_encoder_loss_accuracy_case{case}_{timestamp_plot}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Plots saved to: {plot_path}")

    # === MAKE SUBMISSION ===
    if best_weights:
        print("\n📤 Making predictions on test set...")
        
        # Load best model (handle DataParallel wrapper)
        if hasattr(model, 'module'):
            model = model.module
        model.load_state_dict(best_weights)
        model.eval()
        
        # Create test dataset
        test_dataset = CrossEncoderDataset(test_df, tokenizer, PretrainedModelConfig.MAX_LEN)
        test_loader = DataLoader(
            test_dataset, batch_size=PretrainedModelConfig.BATCH_SIZE, shuffle=False
        )
        
        # Predict on test set
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                # Forward pass without labels (inference mode)
                _, logits = model(input_ids, attention_mask)
                
                # Convert logits to probabilities and then to predictions
                probs = torch.sigmoid(logits.squeeze(1))
                pred_labels = (probs > 0.5).float()
                
                # Convert back to {1, 2} format for submission
                pred_labels = pred_labels + 1  # {0, 1} -> {1, 2}
                predictions.extend(pred_labels.cpu().numpy())
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            "id": test_df.index,
            "real_text_id": np.array(predictions).astype(int)
        }).sort_values("id")
        
        # Create submission filename
        model_name = "CrossEncoder"
        safe_model_name = PretrainedModelConfig.MODEL_NAME.replace("/", "_").replace("-", "_")
        submission_filename = f"submission_case{case}_{model_name}_{safe_model_name}.csv"
        
        # Save submission
        submission.to_csv(submission_filename, index=False)
        print(f"✅ Submission saved to: {submission_filename}")
        print(f"📊 Test predictions: {len(predictions)} samples")
        print(f"📊 Prediction distribution: Class 1: {(np.array(predictions) == 1).sum()}, Class 2: {(np.array(predictions) == 2).sum()}")
        
        # === SUMMARY ===
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED - CrossEncoder Deep")
        print("="*60)
        print(f"📊 Final CrossEncoder Accuracy: {history['test_accuracies'][-1]:.2f}%")
        print(f"📤 Submission: {submission_filename}")
        print("="*60)
    else:
        print("\n❌ Cannot make submission: no trained model available.")
        
    print("\n🎉 Training and submission completed!")
