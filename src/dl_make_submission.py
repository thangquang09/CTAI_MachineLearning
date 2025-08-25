import argparse
import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd

from build_dataset_dataloader import TextComparisonDataset, get_dataset_1, get_dataset
from load_data import read_texts_from_dir
from LSTM import TextClassificationLSTM, PairClassifier, SiameseLSTM
from CONFIG import BATCH_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, required=True, help="Path to model file")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract case number from model path
def extract_case_from_path(model_path):
    """Extract case number from model filename"""
    filename = os.path.basename(model_path).lower()
    if "case1" in filename:
        return 1
    elif "case2" in filename:
        return 2
    else:
        print(f"Warning: Could not detect case from filename '{filename}', defaulting to case 1")
        return 1

# Auto-detect case from model path
case = extract_case_from_path(args.path)
print(f"Auto-detected case: {case} from model path")

# Load test data
print("Loading test data...")
df_test = read_texts_from_dir("./data/fake-or-real-the-impostor-hunt/data/test")
df_test["label"] = 3  # Dummy label
print(f"Test dataset size: {len(df_test)} samples")

# Load vocabulary based on auto-detected case
print(f"Loading vocabulary for case {case}...")
if "textclassification" in args.path.lower():
    _, _, vocabulary = get_dataset_1(case=case)
else:
    _, _, vocabulary = get_dataset(case=case)

# Determine model type and load
model_path = args.path
print(f"Loading model from: {model_path}")

if 'textclassificationlstm' in model_path.lower():
    model_type = "TextClassificationLSTM"
    model = TextClassificationLSTM(
        vocab_size=len(vocabulary),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
    )
    use_confidence_prediction = True
elif 'pairclassifier' in model_path.lower():
    model_type = "PairClassifier"
    model = PairClassifier(
        vocab_size=len(vocabulary),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
    )
    use_confidence_prediction = False
elif 'siamese' in model_path.lower():
    model_type = "SiameseLSTM"
    model = SiameseLSTM(
        vocab_size=len(vocabulary),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
    )
    use_confidence_prediction = False
else:
    raise ValueError(f"Unknown model type in path: {model_path}")

# Load model weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print(f"✅ {model_type} model loaded successfully")

# Create test dataset
test_dataset_pair = TextComparisonDataset(df_test, vocabulary)
test_dataloader_pair = DataLoader(test_dataset_pair, batch_size=BATCH_SIZE, shuffle=False)


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


# Function to make submission
def make_submission():
    print(f"Making predictions with {model_type} model...")
    full_predicted = []

    with torch.no_grad():
        for seq1, seq2, labels in test_dataloader_pair:
            seq1, seq2 = seq1.to(device), seq2.to(device)

            if use_confidence_prediction:
                # Use confidence-based prediction for TextClassificationLSTM
                predictions, explanations = predict_pair_confidence(model, seq1, seq2)
            else:
                # Direct pair prediction for PairClassifier/SiameseLSTM
                outputs = model(seq1, seq2)
                predictions = (torch.sigmoid(outputs.squeeze(1)) > 0.5).float()
            
            full_predicted.append(predictions)

    # Concatenate all predictions
    full_predicted = torch.cat(full_predicted, dim=0)
    full_predicted = full_predicted.cpu().numpy()
    full_predicted = full_predicted + 1  # Convert 0,1 to 1,2

    # Create submission DataFrame
    submission = pd.DataFrame(
        {"id": df_test.index, "real_text_id": full_predicted.astype(int)}
    ).sort_values("id")

    # Create submission directory if it doesn't exist
    os.makedirs("submission", exist_ok=True)

    # Create submission filename with timestamp
    timestamp_sub = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_clean = os.path.basename(model_path).replace('.pth', '')
    submission_filename = f"submission_{model_name_clean}_{timestamp_sub}.csv"
    submission_path = os.path.join("submission", submission_filename)

    # Save submission
    submission.to_csv(submission_path, index=False)
    print(f"✅ Submission saved to: {submission_path}")
    print(f"   Predictions shape: {full_predicted.shape}")
    print(f"   Unique predictions: {sorted(submission['real_text_id'].unique())}")
    print(f"   Label 1: {(submission['real_text_id'] == 1).sum():3d}, Label 2: {(submission['real_text_id'] == 2).sum():3d}")

    return submission_path


# Make submission
print("="*60)
print("MAKING SUBMISSION")
print("="*60)
submission_path = make_submission()
print("="*60)
