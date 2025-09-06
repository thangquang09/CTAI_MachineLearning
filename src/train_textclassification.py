import argparse
import datetime
import os
import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from build_dataset_dataloader import get_dataset_1, get_dataset
from CONFIG import LSTMConfig
from LSTM import TextClassificationLSTMWithAttention

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


def create_submission(model, test_dataset, vocabulary, device, case, model_name="TextClassificationLSTMWithAttention"):
    """Create submission file using confidence-based pair prediction"""
    from datasets import load_dataset
    
    print("🎯 Creating submission file...")
    
    # Load test dataset from HuggingFace
    dataset = load_dataset("thangquang09/fake-new-imposter-hunt-in-texts")
    
    if case == 1:
        test_df = pd.DataFrame(dataset["test_case1"])
    else:
        test_df = pd.DataFrame(dataset["test_case2"])
    
    # We need to process pairs of texts from test_df
    # Each row has file_1, file_2, and we predict which one is real
    model.eval()
    predictions = []
    
    # Process pairs from test_df
    from build_dataset_dataloader import text_to_sequence
    
    with torch.no_grad():
        for _, row in test_df.iterrows():
            # Convert texts to sequences
            seq1 = text_to_sequence(row['file_1'], vocabulary)
            seq2 = text_to_sequence(row['file_2'], vocabulary)
            
            # Pad sequences to same length
            max_len = max(len(seq1), len(seq2))
            seq1 = seq1 + [0] * (max_len - len(seq1))  # Pad with 0
            seq2 = seq2 + [0] * (max_len - len(seq2))
            
            # Convert to tensors
            seq1_tensor = torch.tensor([seq1], device=device)
            seq2_tensor = torch.tensor([seq2], device=device)
            
            # Get predictions for both texts
            logit1 = model(seq1_tensor)
            logit2 = model(seq2_tensor)
            
            # Convert to probabilities (probability of being REAL)
            prob1 = torch.sigmoid(logit1).item()
            prob2 = torch.sigmoid(logit2).item()
            
            # Determine prediction based on confidence
            if prob1 > 0.5 and prob2 > 0.5:
                # Both predicted as REAL -> choose higher confidence
                pred = 1 if prob1 > prob2 else 2
            elif prob1 < 0.5 and prob2 < 0.5:
                # Both predicted as FAKE -> choose less fake (higher prob)
                pred = 1 if prob1 > prob2 else 2
            else:
                # Normal case: one REAL, one FAKE
                pred = 1 if prob1 > 0.5 else 2
                
            predictions.append(pred)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'label': predictions
    })
    
    # Save submission
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f"submission_lstm_textclass_case{case}_{model_name}_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)
    
    print(f"📄 Submission saved to: {submission_path}")
    print(f"📊 Submission stats: {submission_df['label'].value_counts().to_dict()}")
    
    return submission_path


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


def evaluate_individual(model, valid_dataloader, criterion, device):
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


def evaluate_pairs(model, pair_dataloader, device):
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
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    pair_accuracies = []

    # Track both metrics to save 2 models
    best_val_loss = float("inf")
    best_pair_accuracy = 0.0
    best_weights_val_loss = None
    best_weights_pair_acc = None
    best_val_loss_epoch = 0
    best_pair_acc_epoch = 0

    print("🚀 Starting training...")
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
        val_loss, val_accuracy = evaluate_individual(model, valid_dataloader, criterion, device)
        
        # Evaluation on pairs (original task)
        pair_accuracy = evaluate_pairs(model, pair_dataloader, device)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Track best models for both metrics
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights_val_loss = model.module.state_dict().copy() if hasattr(model, 'module') else model.state_dict().copy()
            best_val_loss_epoch = epoch + 1

        if pair_accuracy > best_pair_accuracy:
            best_pair_accuracy = pair_accuracy
            best_weights_pair_acc = model.module.state_dict().copy() if hasattr(model, 'module') else model.state_dict().copy()
            best_pair_acc_epoch = epoch + 1

        # Print progress with best epoch indicators
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
    
    # Load both datasets
    print("\n📊 Loading datasets...")
    train_dataset, val_dataset, vocabulary = get_dataset_1(case=case)  # For training individual texts
    _, val_pair_dataset, _ = get_dataset(case=case)  # For pair evaluation

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_pair_dataloader = DataLoader(val_pair_dataset, batch_size=batch_size, shuffle=False)

    print("📈 Dataset loaded:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    print(f"   Pair validation: {len(val_pair_dataset)} pairs")
    print(f"   Vocabulary: {len(vocabulary)} tokens")

    # Initialize model
    model = TextClassificationLSTMWithAttention(
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

    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LSTMConfig.LEARNING_RATE,
        weight_decay=LSTMConfig.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, "min", 
        factor=LSTMConfig.SCHEDULER_FACTOR, 
        patience=LSTMConfig.SCHEDULER_PATIENCE
    )

    # Train model
    history, best_weights_val_loss, best_weights_pair_acc = train(
        model,
        LSTMConfig.NUM_EPOCHS,
        train_dataloader,
        val_dataloader,
        val_pair_dataloader,
        criterion,
        optimizer,
        scheduler,
        device,
    )

    # Save both models
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model.module._get_name() if hasattr(model, 'module') else model._get_name()

    # Model 1: Best validation loss
    if best_weights_val_loss:
        model_save_path_val = os.path.join("models", f"{model_name}_case{case}_best_val_loss_{timestamp}.pth")
        torch.save(best_weights_val_loss, model_save_path_val)
        print(f"\n🔥 Model with best validation loss (epoch {history['best_val_loss_epoch']}) saved to:")
        print(f"   {model_save_path_val}")
        print(f"   Best val loss: {history['best_val_loss']:.6f}")

    # Model 2: Best pair accuracy
    if best_weights_pair_acc:
        model_save_path_pair = os.path.join("models", f"{model_name}_case{case}_best_pair_acc_{timestamp}.pth")
        torch.save(best_weights_pair_acc, model_save_path_pair)
        print(f"\n🎯 Model with best pair accuracy (epoch {history['best_pair_acc_epoch']}) saved to:")
        print(f"   {model_save_path_pair}")
        print(f"   Best pair accuracy: {history['best_pair_accuracy']:.3f}%")

        # Load best weights for submission
        if hasattr(model, 'module'):
            model.module.load_state_dict(best_weights_pair_acc)
        else:
            model.load_state_dict(best_weights_pair_acc)
        
        # Create submission - Note: we can't use test_dataset from get_dataset_1 for this
        # So we'll create submission directly from test pairs
        create_submission(model, None, vocabulary, device, case, model_name)

    # Enhanced visualization
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(20, 8))

    # Plot 1: Loss with best epoch markers
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

    # Plot 3: Pair comparison accuracy with best epoch marker
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
    timestamp_plot = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join("plots", f"enhanced_lstm_textclass_case{case}_{timestamp_plot}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n📊 Enhanced plots saved to: {plot_path}")

    # Final evaluation with both models
    print("=" * 60)
    print("FINAL COMPARISON:")
    print("=" * 60)

    if best_weights_val_loss:
        if hasattr(model, 'module'):
            model.module.load_state_dict(best_weights_val_loss)
        else:
            model.load_state_dict(best_weights_val_loss)
        final_pair_acc_val = evaluate_pairs(model, val_pair_dataloader, device)
        print(f"🔥 Best Val Loss Model  (E{history['best_val_loss_epoch']:2d}): Pair Accuracy = {final_pair_acc_val:.3f}%")

    if best_weights_pair_acc:
        if hasattr(model, 'module'):
            model.module.load_state_dict(best_weights_pair_acc)
        else:
            model.load_state_dict(best_weights_pair_acc)
        final_pair_acc_pair = evaluate_pairs(model, val_pair_dataloader, device)
        print(f"🎯 Best Pair Acc Model (E{history['best_pair_acc_epoch']:2d}): Pair Accuracy = {final_pair_acc_pair:.3f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()
