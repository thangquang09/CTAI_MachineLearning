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
from CONFIG import *
from LSTM import TextClassificationLSTM, TextClassificationLSTMWithAttention

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
    early_stopping_patience=10,
    use_early_stopping=True,
):
    model.to(device)
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    pair_accuracies = []

    best_weights = None
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print("Epoch | Time  | Train Acc | Train Loss | Val Acc | Val Loss | Pair Acc")
    print("-" * 75)

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

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Print progress
        print(
            f"{epoch + 1:5d} | {time.time() - epoch_start_time:5.2f}s | "
            f"{epoch_accuracy:8.3f}% | {epoch_loss:9.6f} | "
            f"{val_accuracy:6.3f}% | {val_loss:8.6f} | {pair_accuracy:7.3f}%"
        )

        # Store metrics
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        pair_accuracies.append(pair_accuracy)

        # Early stopping
        if use_early_stopping and epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {early_stopping_patience} epochs with no improvement.")
            break

    history = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "pair_accuracies": pair_accuracies,
    }

    return history, best_weights


# Start training
print(f"Training {model._get_name()} on case {CASE}")
print(f"Train dataset: {len(train_dataset)} samples")
print(f"Val dataset: {len(val_dataset)} samples")
print(f"Pair validation: {len(val_pair_dataset)} pairs")
print(f"Device: {device}")
print("-" * 75)

history, best_weights = train(
    model,
    NUM_EPOCHS,
    train_dataloader,
    val_dataloader,
    val_pair_dataloader,
    criterion,
    optimizer,
    scheduler,
    device,
    use_early_stopping=EARLY_STOPPING,
)

# Save best model
if best_weights:
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model._get_name()
    model_save_path = os.path.join("models", f"{model_name}_case{CASE}_{timestamp}.pth")
    
    torch.save(best_weights, model_save_path)
    print(f"\nModel with best validation loss saved to: {model_save_path}")
else:
    print("\nTraining completed, but no best model was saved.")

# Create visualization
os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(18, 6))

# Plot 1: Loss
plt.subplot(1, 3, 1)
plt.plot(history["train_losses"], label="Training Loss")
plt.plot(history["val_losses"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

# Plot 2: Individual text accuracy
plt.subplot(1, 3, 2)
plt.plot(history["train_accuracies"], label="Training Accuracy")
plt.plot(history["val_accuracies"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Individual Text Classification Accuracy")
plt.legend()
plt.grid(True)

# Plot 3: Pair comparison accuracy
plt.subplot(1, 3, 3)
plt.plot(history["pair_accuracies"], label="Pair Accuracy", color='red')
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Pair Comparison Accuracy (Original Task)")
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save plots
timestamp_plot = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = os.path.join("plots", f"textclassification_case{CASE}_{timestamp_plot}.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Plots saved to: {plot_path}")

# Final evaluation
if best_weights:
    model.load_state_dict(best_weights)
    final_pair_acc = evaluate_pairs(model, val_pair_dataloader)
    print(f"\nFinal pair comparison accuracy: {final_pair_acc:.3f}%")
