import os
import time

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from build_dataset_dataloader import get_dataset
from CONFIG import *
from LSTM import SiameseLSTM

train_dataset, val_dataset, vocabulary = get_dataset(case=CASE)


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = SiameseLSTM(
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
        epoch_loss = running_loss / len(
            train_dataloader
        )

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

    # Đặt tên file cho model, có thể thêm cả case để phân biệt
    model_save_path = os.path.join("models", f"Siamese_LSTM_case{CASE}.pth")

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
plot_path = os.path.join("plots", f"loss_accuracy_case{CASE}.png")
plt.savefig(plot_path)
plt.close()

print(f"Plots saved to: {plot_path}")
