import datetime
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


import argparse
from build_dataset_dataloader import get_dataset
from CONFIG import BATCH_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_EPOCHS, LEARNING_RATE, SEQ_LENGTH, EARLY_STOPPING
from LSTM import PairClassifier, SiameseLSTM

parser = argparse.ArgumentParser()
parser.add_argument('--case', type=int, default=1, help='Case number (1 or 2)')
args = parser.parse_args()
CASE = args.case

train_dataset, val_dataset, vocabulary = get_dataset(case=CASE)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# model = SiameseLSTM(
#     vocab_size=len(vocabulary),
#     embedding_dim=EMBEDDING_DIM,
#     hidden_dim=HIDDEN_DIM,
#     output_dim=OUTPUT_DIM,
# )

model = PairClassifier(
    vocab_size=len(vocabulary),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, valid_dataloader, criterion):
    model.eval()
    total_loss = 0
    running_correct = 0
    total = 0
    with torch.no_grad():
        for seq1, seq2, labels in valid_dataloader:
            seq1, seq2, labels = (
                seq1.to(device),
                seq2.to(device),
                labels.to(device).float(),
            )
            outputs = model(seq1, seq2)
            loss = criterion(outputs.squeeze(1), labels)
            total_loss += loss.item()

            predicted = (torch.sigmoid(outputs.squeeze(1)) > 0.5).float()
            running_correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * running_correct / total
    total_loss = total_loss / total
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

    for epoch in range(max_epoch):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        epoch_start_time = time.time()

        for i, (seq1, seq2, labels) in enumerate(train_dataloader):
            seq1, seq2, labels = (
                seq1.to(device),
                seq2.to(device),
                labels.to(device).float(),
            )

            optimizer.zero_grad()
            outputs = model(seq1, seq2)
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

        if use_early_stopping and epochs_no_improve >= early_stopping_patience:
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


model.to(device)

history, best_weights = train(
    model,
    NUM_EPOCHS,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    scheduler,
    device,
    use_early_stopping=EARLY_STOPPING,
)

if best_weights:
    # Tạo thư mục 'models' nếu nó chưa tồn tại
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Đặt tên file cho model, có thể thêm cả case để phân biệt
    model_name = model._get_name()
    model_save_path = os.path.join("models", f"{model_name}_case{CASE}_{timestamp}.pth")

    # Lưu state_dict của model
    torch.save(best_weights, model_save_path)

    print(f"\nModel with best validation loss saved to: {model_save_path}")
else:
    print("\nTraining completed, but no best model was saved.")

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Create a figure with 2 subplots (1 row, 2 columns)
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

# Adjust layout
plt.tight_layout()

# Save the figure
timestamp_plot = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = os.path.join("plots", f"loss_accuracy_case{CASE}_{timestamp_plot}.png")
plt.savefig(plot_path)
plt.close()


print(f"Plots saved to: {plot_path}")

# ===================== MAKE SUBMISSION =====================
print("\n" + "="*60)
print("MAKING SUBMISSION")
print("="*60)

import pandas as pd
from load_data import read_texts_from_dir
from build_dataset_dataloader import TextComparisonDataset

# Load test data
print("Loading test data...")
df_test = read_texts_from_dir('/home/thangquang09/CODE/CTAI_MachineLearning/data/fake-or-real-the-impostor-hunt/data/test')
df_test['label'] = 3  # Dummy label

print(f"Test dataset size: {len(df_test)} samples")

# Create test dataset and dataloader
test_dataset = TextComparisonDataset(df_test, vocabulary)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load best model for prediction
if best_weights:
    model.load_state_dict(best_weights)
    model.to(device)
    model.eval()
    
    print("Making predictions...")
    full_predicted = []
    
    with torch.no_grad():
        for seq1, seq2, labels in test_dataloader:
            seq1, seq2 = seq1.to(device), seq2.to(device)
            outputs = model(seq1, seq2)
            predicted = (torch.sigmoid(outputs.squeeze(1)) > 0.5).float()
            full_predicted.append(predicted)
    
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
    submission_filename = f"submission_{model._get_name().lower()}_case{CASE}_{timestamp_sub}.csv"
    submission_path = os.path.join("submission", submission_filename)
    
    # Save submission
    submission.to_csv(submission_path, index=False)
    print(f"✅ Submission saved to: {submission_path}")
    print(f"   Predictions shape: {full_predicted.shape}")
    print(f"   Unique predictions: {sorted(submission['real_text_id'].unique())}")
    
else:
    print("❌ No trained model available for submission")

print("="*60)
