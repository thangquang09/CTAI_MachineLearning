
import argparse
import datetime
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from build_dataset_dataloader import get_dataset_1, get_dataset, text_to_sequence
from CONFIG import BATCH_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_EPOCHS, LEARNING_RATE, SEQ_LENGTH, EARLY_STOPPING
from LSTM import TextClassificationLSTM, TextClassificationLSTMWithAttention

parser = argparse.ArgumentParser()
parser.add_argument('--case', type=int, default=1, help='Case number (1 or 2)')
args = parser.parse_args()
CASE = args.case

# Load both datasets
train_dataset, val_dataset, vocabulary = get_dataset_1(case=CASE)  # For training individual texts
_, val_pair_dataset, _ = get_dataset(case=CASE)  # For pair evaluation

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_pair_dataloader = DataLoader(val_pair_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model selection
model = TextClassificationLSTM(
    vocab_size=len(vocabulary),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
)

# Alternative model with attention
# model = TextClassificationLSTMWithAttention(
#     vocab_size=len(vocabulary),
#     embedding_dim=EMBEDDING_DIM,
#     hidden_dim=HIDDEN_DIM,
#     output_dim=OUTPUT_DIM,
# )

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_pair_confidence(model, seq1, seq2):
    """
    Predict which text is REAL using confidence-based approach
    Returns: prediction (0 or 1), confidence score, explanation
    """
    model.eval()
    with torch.no_grad():
        # Get predictions for both texts
        logit1 = model(seq1)
        logit2 = model(seq2)
        
        # Convert to probabilities
        prob1 = torch.sigmoid(logit1)
        prob2 = torch.sigmoid(logit2)
        
        # Determine prediction based on confidence
        predictions = []
        explanations = []
        
        for i in range(len(prob1)):
            p1, p2 = prob1[i].item(), prob2[i].item()
            
            if p1 > 0.5 and p2 > 0.5:
                # Both predicted as REAL -> choose higher confidence
                if p1 > p2:
                    pred = 0  # text1 is REAL (label=1 in original format)
                    exp = f"Both REAL, text1 more confident ({p1:.4f} vs {p2:.4f})"
                else:
                    pred = 1  # text2 is REAL (label=2 in original format)
                    exp = f"Both REAL, text2 more confident ({p2:.4f} vs {p1:.4f})"
            elif p1 < 0.5 and p2 < 0.5:
                # Both predicted as FAKE -> choose less fake (higher prob)
                if p1 > p2:
                    pred = 0  # text1 less fake
                    exp = f"Both FAKE, text1 less fake ({p1:.4f} vs {p2:.4f})"
                else:
                    pred = 1  # text2 less fake
                    exp = f"Both FAKE, text2 less fake ({p2:.4f} vs {p1:.4f})"
            else:
                # Normal case: one REAL, one FAKE
                if p1 > 0.5:
                    pred = 0  # text1 is REAL
                    exp = f"Text1 REAL ({p1:.4f}), Text2 FAKE ({p2:.4f})"
                else:
                    pred = 1  # text2 is REAL
                    exp = f"Text2 REAL ({p2:.4f}), Text1 FAKE ({p1:.4f})"
            
            predictions.append(pred)
            explanations.append(exp)
        
        return torch.tensor(predictions, device=seq1.device), explanations


def evaluate_individual(model, valid_dataloader, criterion):
    """Evaluate on individual text classification"""
    model.eval()
    total_loss = 0
    running_correct = 0
    total = 0
    
    with torch.no_grad():
        for seq, labels in valid_dataloader:
            seq, labels = seq.to(device), labels.to(device).float()
            outputs = model(seq)
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
        for seq1, seq2, labels in pair_dataloader:
            seq1, seq2, labels = seq1.to(device), seq2.to(device), labels.to(device)
            
            # Get pair predictions using confidence approach
            predictions, _ = predict_pair_confidence(model, seq1, seq2)
            
            # Convert original labels (0,1) to predictions format
            # Original: 0 means text1 is REAL, 1 means text2 is REAL
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

    # Track cả 2 metrics để lưu 2 models
    best_val_loss = float("inf")
    best_pair_accuracy = 0.0
    best_weights_val_loss = None
    best_weights_pair_acc = None
    best_val_loss_epoch = 0
    best_pair_acc_epoch = 0

    print("Epoch | Time  | Train Acc | Train Loss | Val Acc | Val Loss | Pair Acc | Best VL | Best PA")
    print("-" * 95)

    for epoch in range(max_epoch):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        epoch_start_time = time.time()

        # Training on individual texts
        for seq, labels in train_dataloader:
            seq, labels = seq.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(seq)
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

        # Track best models cho cả 2 metrics
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights_val_loss = model.state_dict().copy()
            best_val_loss_epoch = epoch + 1

        if pair_accuracy > best_pair_accuracy:
            best_pair_accuracy = pair_accuracy
            best_weights_pair_acc = model.state_dict().copy()
            best_pair_acc_epoch = epoch + 1

        # Print progress với best epoch indicators
        val_loss_indicator = "🔥" if val_loss == best_val_loss else "  "
        pair_acc_indicator = "🎯" if pair_accuracy == best_pair_accuracy else "  "
        
        print(
            f"{epoch + 1:5d} | {time.time() - epoch_start_time:5.2f}s | "
            f"{epoch_accuracy:8.3f}% | {epoch_loss:9.6f} | "
            f"{val_accuracy:6.3f}% | {val_loss:8.6f} | {pair_accuracy:7.3f}% | "
            f"E{best_val_loss_epoch:2d}{val_loss_indicator} | E{best_pair_acc_epoch:2d}{pair_acc_indicator}"
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
print(f"Training {model._get_name()} on case {CASE}")
print(f"Train dataset: {len(train_dataset)} samples")
print(f"Val dataset: {len(val_dataset)} samples")
print(f"Pair validation: {len(val_pair_dataset)} pairs")
print(f"Device: {device}")
print(f"Early stopping: DISABLED - Training full {NUM_EPOCHS} epochs")
print("-" * 95)

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

# Save cả 2 models
os.makedirs("models", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = model._get_name()

# Model 1: Best validation loss
if best_weights_val_loss:
    model_save_path_val = os.path.join("models", f"{model_name}_case{CASE}_best_val_loss_{timestamp}.pth")
    torch.save(best_weights_val_loss, model_save_path_val)
    print(f"\n🔥 Model with best validation loss (epoch {history['best_val_loss_epoch']}) saved to:")
    print(f"   {model_save_path_val}")
    print(f"   Best val loss: {history['best_val_loss']:.6f}")

# Model 2: Best pair accuracy
if best_weights_pair_acc:
    model_save_path_pair = os.path.join("models", f"{model_name}_case{CASE}_best_pair_acc_{timestamp}.pth")
    torch.save(best_weights_pair_acc, model_save_path_pair)
    print(f"\n🎯 Model with best pair accuracy (epoch {history['best_pair_acc_epoch']}) saved to:")
    print(f"   {model_save_path_pair}")
    print(f"   Best pair accuracy: {history['best_pair_accuracy']:.3f}%")

# Enhanced visualization
os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(20, 8))

# Plot 1: Loss với best epoch markers
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

# Plot 3: Pair comparison accuracy với best epoch marker
plt.subplot(1, 4, 3)
plt.plot(history["pair_accuracies"], label="Pair Accuracy", color='green', linewidth=2)
plt.axvline(x=history['best_pair_acc_epoch']-1, color='red', linestyle='--', alpha=0.7, label=f'Best Pair Acc (E{history["best_pair_acc_epoch"]})')
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Pair Comparison Accuracy")
plt.legend()
plt.grid(True)

# Plot 4: So sánh best epochs
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
plot_path = os.path.join("plots", f"dual_textclassification_case{CASE}_{timestamp_plot}.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n📊 Enhanced plots saved to: {plot_path}")

# Final evaluation với cả 2 models
print(f"\n{'='*60}")
print(f"FINAL COMPARISON:")
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

# ===================== MAKE SUBMISSION =====================
print("\n" + "="*60)
print("MAKING SUBMISSION")
print("="*60)

import pandas as pd
from load_data import read_texts_from_dir

# Load test data
print("Loading test data...")
df_test = read_texts_from_dir('/home/thangquang09/CODE/CTAI_MachineLearning/data/fake-or-real-the-impostor-hunt/data/test')
df_test['label'] = 3  # Dummy label

print(f"Test dataset size: {len(df_test)} samples")

# Create test dataset using pair comparison format for confidence-based prediction
from build_dataset_dataloader import TextComparisonDataset
test_dataset_pair = TextComparisonDataset(df_test, vocabulary)
test_dataloader_pair = DataLoader(test_dataset_pair, batch_size=BATCH_SIZE, shuffle=False)

# Function to make submission with confidence-based approach
def make_submission_with_confidence(model_weights, model_name_suffix):
    if model_weights is None:
        print(f"❌ No {model_name_suffix} model available")
        return None
        
    # Load model
    model.load_state_dict(model_weights)
    model.to(device)
    model.eval()
    
    print(f"Making predictions with {model_name_suffix} model...")
    full_predicted = []
    
    with torch.no_grad():
        for seq1, seq2, labels in test_dataloader_pair:
            seq1, seq2 = seq1.to(device), seq2.to(device)
            
            # Use confidence-based prediction
            predictions, explanations = predict_pair_confidence(model, seq1, seq2)
            full_predicted.append(predictions)
    
    # Concatenate all predictions
    full_predicted = torch.cat(full_predicted, dim=0)
    full_predicted = full_predicted.cpu().numpy()
    full_predicted = full_predicted + 1  # Convert 0,1 to 1,2
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        "id": df_test.index,
        "real_text_id": full_predicted.astype(int)
    }).sort_values("id")
    
    # Create submission directory if it doesn't exist
    os.makedirs("submission", exist_ok=True)
    
    # Create submission filename with timestamp
    timestamp_sub = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_filename = f"submission_{model_name.lower()}_{model_name_suffix}_case{CASE}_{timestamp_sub}.csv"
    submission_path = os.path.join("submission", submission_filename)
    
    # Save submission
    submission.to_csv(submission_path, index=False)
    print(f"✅ {model_name_suffix} submission saved to: {submission_path}")
    print(f"   Predictions shape: {full_predicted.shape}")
    print(f"   Unique predictions: {sorted(submission['real_text_id'].unique())}")
    
    return submission_path

# Make submissions for both models
print("\n1. Creating submission with BEST VAL LOSS model:")
submission_path_val = make_submission_with_confidence(best_weights_val_loss, "best_val_loss")

print("\n2. Creating submission with BEST PAIR ACCURACY model:")
submission_path_pair = make_submission_with_confidence(best_weights_pair_acc, "best_pair_acc")

# Compare predictions if both models exist
if submission_path_val and submission_path_pair:
    print(f"\n📊 SUBMISSION COMPARISON:")
    
    sub_val = pd.read_csv(submission_path_val)
    sub_pair = pd.read_csv(submission_path_pair)
    
    # Calculate agreement
    agreement = (sub_val['real_text_id'] == sub_pair['real_text_id']).mean() * 100
    print(f"   Agreement between models: {agreement:.1f}%")
    
    # Show prediction distribution
    print(f"   Val Loss Model  - Label 1: {(sub_val['real_text_id'] == 1).sum():3d}, Label 2: {(sub_val['real_text_id'] == 2).sum():3d}")
    print(f"   Pair Acc Model  - Label 1: {(sub_pair['real_text_id'] == 1).sum():3d}, Label 2: {(sub_pair['real_text_id'] == 2).sum():3d}")
    
    if agreement < 100:
        print(f"\n⚠️  Models disagree on {100-agreement:.1f}% of predictions!")
        disagreements = sub_val[sub_val['real_text_id'] != sub_pair['real_text_id']]
        print(f"   First few disagreements (IDs): {disagreements['id'].head().tolist()}")

print("="*60)
