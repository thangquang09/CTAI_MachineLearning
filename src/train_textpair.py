import argparse
import os
import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from build_dataset_dataloader import get_dataset
from CONFIG import LSTMConfig
from LSTM import PairClassifierLSTM

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_submission(model, test_dataset, vocabulary, device, case, model_name="PairClassifierLSTM"):
    """Create submission file for the test dataset"""
    from datasets import load_dataset
    
    print("🎯 Creating submission file...")
    
    # Load test dataset from HuggingFace
    dataset = load_dataset("thangquang09/fake-new-imposter-hunt-in-texts")
    
    if case == 1:
        test_df = pd.DataFrame(dataset["test_case1"])
    else:
        test_df = pd.DataFrame(dataset["test_case2"])
    
    # Get test dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=LSTMConfig.BATCH_SIZE, shuffle=False)
    
    # Predict
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for seq1, seq2, _ in test_dataloader:  # Labels are dummy for test set
            seq1, seq2 = seq1.to(device), seq2.to(device)
            outputs = model(seq1, seq2)
            probs = torch.sigmoid(outputs.squeeze(1))
            
            # Convert to predictions: prob > 0.5 means text2 is real (label=2), else text1 is real (label=1)
            batch_preds = (probs > 0.5).long() + 1  # Convert 0,1 to 1,2
            predictions.extend(batch_preds.cpu().tolist())
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'label': predictions
    })
    
    # Save submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f"submission_lstm_textpair_case{case}_{model_name}_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)
    
    print(f"📄 Submission saved to: {submission_path}")
    print(f"📊 Submission stats: {submission_df['label'].value_counts().to_dict()}")
    
    return submission_path

def evaluate(model, valid_dataloader, criterion, device):
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
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    best_weights = None
    best_test_loss = float("inf")
    epochs_no_improve = 0

    print("🚀 Starting training...")
    print("Epoch | Time  | Train Acc | Train Loss | Valid Acc | Valid Loss | LR")
    print("-" * 80)

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
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

        epoch_accuracy = 100 * running_correct / total
        epoch_loss = running_loss / len(train_dataloader)

        test_loss, test_accuracy = evaluate(model, valid_dataloader, criterion, device)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_weights = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        current_lr = optimizer.param_groups[0]['lr']
        
        print(
            f"{epoch + 1:5d} | {time.time() - epoch_start_time:5.2f}s | "
            f"{epoch_accuracy:8.3f}% | {epoch_loss:9.6f} | "
            f"{test_accuracy:8.3f}% | {test_loss:9.6f} | {current_lr:.2e}"
        )

        scheduler.step(test_loss)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        if use_early_stopping and epochs_no_improve >= early_stopping_patience:
            print(f"\n⏹️  Early stopping triggered after {early_stopping_patience} epochs with no improvement.")
            break

    values_dict = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies,
    }

    return values_dict, best_weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int, default=1, help='Case number (1 or 2)')
    parser.add_argument('--batchsize', type=int, default=None, help='Batch size (overrides config)')
    args = parser.parse_args()
    
    case = args.case
    batch_size = args.batchsize if args.batchsize else LSTMConfig.BATCH_SIZE
    
    # Set random seed
    set_random_seed(LSTMConfig.RANDOM_SEED)
    
    print("🔧 Configuration:")
    print(f"   Case: {case}")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {LSTMConfig.DEVICE}")
    print(f"   Random seed: {LSTMConfig.RANDOM_SEED}")
    
    # Check for multiple GPUs
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and LSTMConfig.USE_MULTI_GPU:
        print(f"🚀 Multi-GPU training enabled! Using {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Load dataset
    print("\n📊 Loading dataset...")
    train_dataset, val_dataset, vocabulary = get_dataset(case=case)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("📈 Dataset loaded:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    print(f"   Vocabulary: {len(vocabulary)} tokens")

    # Initialize model
    model = PairClassifierLSTM(
        vocab_size=len(vocabulary),
        embedding_dim=LSTMConfig.EMBEDDING_DIM,
        hidden_dim=LSTMConfig.HIDDEN_DIM,
        output_dim=LSTMConfig.OUTPUT_DIM,
        num_layers=LSTMConfig.NUM_LAYERS,
        num_residual_blocks=LSTMConfig.NUM_RESIDUAL_BLOCKS,
        dropout=LSTMConfig.LSTM_DROPOUT,
        embedding_dropout=LSTMConfig.EMBEDDING_DROPOUT,
        residual_dropout=LSTMConfig.RESIDUAL_DROPOUT
    )

    device = torch.device(LSTMConfig.DEVICE)
    
    # Multi-GPU setup
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and LSTMConfig.USE_MULTI_GPU:
        model = nn.DataParallel(model)
        print(f"🔥 Model wrapped with DataParallel for {torch.cuda.device_count()} GPUs")
    
    model.to(device)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🤖 Model: {model.module._get_name() if hasattr(model, 'module') else model._get_name()}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Training setup with improved regularization
    criterion = nn.BCEWithLogitsLoss()  # Standard BCE loss
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LSTMConfig.LEARNING_RATE,
        weight_decay=LSTMConfig.WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    scheduler = ReduceLROnPlateau(
        optimizer, "min", 
        factor=0.5,  # More aggressive reduction
        patience=3,  # Reduce patience
    )

    # Train model
    history, best_weights = train(
        model,
        LSTMConfig.NUM_EPOCHS,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        scheduler,
        device,
        early_stopping_patience=LSTMConfig.EARLY_STOPPING_PATIENCE,
        use_early_stopping=True,
    )

    # Save best model
    if best_weights:
        os.makedirs("models", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model.module._get_name() if hasattr(model, 'module') else model._get_name()
        model_save_path = os.path.join("models", f"{model_name}_case{case}_{timestamp}.pth")

        torch.save(best_weights, model_save_path)
        print(f"\n💾 Best model saved to: {model_save_path}")

        # Load best weights for submission
        if hasattr(model, 'module'):
            model.module.load_state_dict(best_weights)
        else:
            model.load_state_dict(best_weights)
        
        # Create submission
        from build_dataset_dataloader import get_test_dataset
        test_dataset = get_test_dataset(case=case, vocabulary=vocabulary)
        create_submission(model, test_dataset, vocabulary, device, case, model_name)
        
    else:
        print("\n❌ Training completed, but no best model was saved.")

    # Create plots
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(15, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history["train_losses"], label="Training Loss")
    plt.plot(history["test_losses"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy plot
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
    plot_path = os.path.join("plots", f"lstm_textpair_case{case}_{timestamp_plot}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Plots saved to: {plot_path}")

if __name__ == "__main__":
    main()