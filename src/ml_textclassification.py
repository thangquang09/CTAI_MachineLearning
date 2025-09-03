import re
import string

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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

def build_text_classification_data(df):
    new_data = []
    new_label = []
    for _, row in df.iterrows():
        text1 = row['file_1']
        text2 = row['file_2']
        label = row['label']
        
        if label == 1:
            new_data.append(text1)
            new_label.append(1)
            new_data.append(text2)
            new_label.append(0)
            
        else:
            new_data.append(text1)
            new_label.append(0)
            new_data.append(text2)
            new_label.append(1)

    return pd.DataFrame({"text": new_data, "label": new_label})


def eval_text_pair(model, embedding_model, pair_df, threshold=0.5, make_submission=False):
    # if not all(col in pair_df.columns for col in ['file_1', 'file_2', 'label']):
    #     raise ValueError("pair_df must contain 'file_1', 'file_2', and 'label' columns")
    
    # if not pair_df['label'].isin([1, 2]).all():
    #     raise ValueError("Label column must contain only 1 or 2")
    
    # Encode all texts at once for efficiency
    texts = pair_df[['file_1', 'file_2']].values.flatten().tolist()
    embeddings = embedding_model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = embeddings.reshape(-1, 2, embeddings.shape[-1])  # Reshape to (n_pairs, 2, embedding_dim)
    
    correct_pairs = 0
    total_pairs = len(pair_df)
    pred_labels = []
    true_labels = pair_df['label'].tolist()
    
    for i in range(total_pairs):
        pair_embeddings = embeddings[i]
        probs = model.predict_proba(pair_embeddings)[:, 1]
        
        # Decision logic
        if probs[0] > threshold and probs[1] > threshold:
            pred_label = 1 if probs[0] > probs[1] else 2
        else:
            pred_label = 1 if probs[0] > probs[1] else 2
        
        pred_labels.append(pred_label)
        if pred_label == true_labels[i]:
            correct_pairs += 1
    
    pair_accuracy = correct_pairs / total_pairs
    
    print(f"Pair-wise Accuracy: {pair_accuracy:.4f}")
    
    if make_submission:
        return pred_labels
    return pair_accuracy


def train(model, embedding_model, train_df, valid_df, train_pair_df, valid_pair_df):
    # Generate Embedding
    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].values

    valid_texts = valid_df['text'].tolist()
    valid_labels = valid_df['label'].values
    
    # Encode
    train_embeddings = embedding_model.encode(train_texts, batch_size=32, show_progress_bar=True)
    valid_embeddings = embedding_model.encode(valid_texts, batch_size=32, show_progress_bar=True)
    
    # Train Model
    model.fit(train_embeddings, train_labels)

    # Evaluation - Text classification
    valid_predictions = model.predict(valid_embeddings)
    print("Validation Accuracy:", accuracy_score(valid_labels, valid_predictions))
    
    # Valuation - Text Pair classification
    train_pair_accuracy = eval_text_pair(model, embedding_model, train_pair_df)
    print("Train Pair Accuracy:", train_pair_accuracy)

    valid_pair_accuracy = eval_text_pair(model, embedding_model, valid_pair_df)
    print("Validation Pair Accuracy:", valid_pair_accuracy)

    return model

dataset = load_dataset("thangquang09/fake-new-imposter-hunt-in-texts")
case = 2

train_df = dataset[f'case{case}_train'].to_pandas()
valid_df = dataset[f'case{case}_validation'].to_pandas()
test_df = dataset['test'].to_pandas()

train_df['file_1'] = train_df['file_1'].apply(preprocessing)
train_df['file_2'] = train_df['file_2'].apply(preprocessing)

valid_df['file_1'] = valid_df['file_1'].apply(preprocessing)
valid_df['file_2'] = valid_df['file_2'].apply(preprocessing)

test_df['file_1'] = test_df['file_1'].apply(preprocessing)
test_df['file_2'] = test_df['file_2'].apply(preprocessing)

train_textclf_df = build_text_classification_data(train_df)
valid_textclf_df = build_text_classification_data(valid_df)

lr_model = LogisticRegression(max_iter=10000, random_state=42)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

trained_lr_model = train(lr_model, embedding_model, train_textclf_df, valid_textclf_df, train_df, valid_df)


y_preds_submission = eval_text_pair(trained_lr_model, embedding_model, test_df, make_submission=True)
submission = pd.DataFrame(
        {"id": test_df.index, "real_text_id": np.array(y_preds_submission).astype(int)}
    ).sort_values("id")

submission.to_csv("submission_ml_text_classification_case2.csv", index=False)

# print(train_df.head())
# print("="*50)
# print(train_textclf_df.head())