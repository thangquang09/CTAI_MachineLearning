import re
import string

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from ml_objective import cb_objective, lr_objective, rf_objective


# === MODEL CONFIGURATION MAPPING ===
MODEL_CONFIG = {
    LogisticRegression: {
        'objective_func': lr_objective,
        'n_trials': 30,
        'model_name': 'LogisticRegression'
    },
    RandomForestClassifier: {
        'objective_func': rf_objective,
        'n_trials': 25,
        'model_name': 'RandomForest'
    },
    CatBoostClassifier: {
        'objective_func': cb_objective,
        'n_trials': 20,
        'model_name': 'CatBoost'
    }
}


def get_model_config(model_class):
    """Get configuration for the specified model class."""
    if model_class not in MODEL_CONFIG:
        raise ValueError(f"Unsupported model class: {model_class}. "
                        f"Supported models: {list(MODEL_CONFIG.keys())}")
    return MODEL_CONFIG[model_class]


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


def eval_text_pair(
    model, embedding_model, pair_df, threshold=0.5, make_submission=False
):
    texts = pair_df[["file_1", "file_2"]].values.flatten().tolist()
    embeddings = embedding_model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = embeddings.reshape(
        -1, 2, embeddings.shape[-1]
    )  # Reshape to (n_pairs, 2, embedding_dim)

    correct_pairs = 0
    total_pairs = len(pair_df)
    pred_labels = []
    true_labels = pair_df["label"].tolist()

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

    if make_submission:
        return pred_labels
    return pair_accuracy



def train(model_class, embedding_model, train_df, valid_df, train_pair_df, valid_pair_df):
    """
    Auto train function with intelligent model selection.
    Chỉ cần đổi model_class là tự động fine tune đúng model!
    """
    
    # Get model configuration
    config = get_model_config(model_class)
    print(f"\n🎯 Training {config['model_name']} with {config['n_trials']} trials")
    
    # Generate Embedding
    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].values

    valid_texts = valid_df["text"].tolist()
    valid_labels = valid_df["label"].values

    # Encode
    print("Encoding training texts...")
    train_embeddings = embedding_model.encode(
        train_texts, batch_size=32, show_progress_bar=True
    )
    print("Encoding validation texts...")
    valid_embeddings = embedding_model.encode(
        valid_texts, batch_size=32, show_progress_bar=True
    )

    # Fine-tune Model with Optuna
    print(f"🔍 Starting Optuna hyperparameter optimization for {config['model_name']}...")
    study = optuna.create_study(direction='maximize')
    
    # Use the appropriate objective function
    def objective_func(trial):
        return config['objective_func'](
            trial,
            train_embeddings,
            train_labels,
            valid_embeddings,
            valid_labels
        )
    
    study.optimize(objective_func, n_trials=config['n_trials'])
    
    best_params = study.best_params
    print(f"\n🏆 Best params for {config['model_name']}:", best_params)
    print(f"🎯 Best validation score: {study.best_value:.4f}")
    
    # Create model with best parameters
    print(f"\n🔧 Creating {config['model_name']} with optimized parameters...")
    if model_class == RandomForestClassifier:
        # Ensure random_state and n_jobs are set
        best_params.update({'random_state': 42, 'n_jobs': -1})
    elif model_class == CatBoostClassifier:
        # Ensure random_state, verbose, and thread_count are set
        best_params.update({'random_state': 42, 'verbose': 0, 'thread_count': -1})
    elif model_class == LogisticRegression:
        # Ensure random_state and max_iter are set
        best_params.update({'random_state': 42, 'max_iter': 10000})
    
    model = model_class(**best_params)
    model.fit(train_embeddings, train_labels)

    # Evaluation - Text classification
    valid_predictions = model.predict(valid_embeddings)
    text_accuracy = accuracy_score(valid_labels, valid_predictions)
    print(f"📊 Text Classification Accuracy: {text_accuracy:.4f}")

    # Evaluation - Text Pair classification
    print("📝 Evaluating pair classification...")
    train_pair_accuracy = eval_text_pair(model, embedding_model, train_pair_df)
    print(f"🎯 Train Pair Accuracy: {train_pair_accuracy:.4f}")

    valid_pair_accuracy = eval_text_pair(model, embedding_model, valid_pair_df)
    print(f"🎯 Validation Pair Accuracy: {valid_pair_accuracy:.4f}")
    
    # Save optimization results
    results = {
        'model': model,
        'best_params': best_params,
        'best_score': study.best_value,
        'text_accuracy': text_accuracy,
        'train_pair_accuracy': train_pair_accuracy,
        'valid_pair_accuracy': valid_pair_accuracy,
        'model_name': config['model_name']
    }

    return model, results


def create_submission(model, embedding_model, test_df, case, results):
    """Create submission file with model info."""
    print("\n📤 Making predictions on test set...")
    y_preds_submission = eval_text_pair(
        model, embedding_model, test_df, make_submission=True
    )

    submission = pd.DataFrame(
        {"id": test_df.index, "real_text_id": np.array(y_preds_submission).astype(int)}
    ).sort_values("id")

    # Create informative filename
    model_name = results['model_name']
    score = results['valid_pair_accuracy']
    submission_filename = f"submission_case{case}_{model_name}_score{score:.4f}.csv"
    
    submission.to_csv(submission_filename, index=False)
    print(f"✅ Submission saved to: {submission_filename}")
    
    return submission_filename


# === MAIN EXECUTION ===
if __name__ == "__main__":
    # === CONFIGURATION ===
    case = 2
    merge_option = True
    
    # 🎯 CHỌN MODEL
    model_class = RandomForestClassifier      # 🌲 Random Forest
    # model_class = LogisticRegression        # 📈 Logistic Regression  
    # model_class = CatBoostClassifier        # 🚀 CatBoost
    
    print(f"🚀 Starting training pipeline with {get_model_config(model_class)['model_name']}")
    
    # === DATA LOADING & PREPROCESSING ===
    print("\n📊 Loading dataset...")
    dataset = load_dataset("thangquang09/fake-new-imposter-hunt-in-texts")
    
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

    # Merge option (as requested)
    if merge_option:
        print("🔄 Merging validation into training data...")
        train_df = pd.concat([train_df, valid_df], ignore_index=True)

    # Build text classification data
    print("🏗️ Building text classification datasets...")
    train_textclf_df = build_text_classification_data(train_df)
    valid_textclf_df = build_text_classification_data(valid_df)

    print(f"📏 Train samples: {len(train_textclf_df)}, Valid samples: {len(valid_textclf_df)}")

    # Initialize embedding model
    embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")
    print(f"🤖 Using embedding model: {embedding_model}")

    # === TRAINING ===
    trained_model, results = train(
        model_class, embedding_model, train_textclf_df, valid_textclf_df, train_df, valid_df
    )

    # === SUBMISSION ===
    submission_filename = create_submission(trained_model, embedding_model, test_df, case, results)
    
    # === SUMMARY ===
    print("\n" + "="*60)
    print(f"🎉 TRAINING COMPLETED - {results['model_name']}")
    print("="*60)
    print(f"📊 Text Classification Accuracy: {results['text_accuracy']:.4f}")
    print(f"🎯 Validation Pair Accuracy: {results['valid_pair_accuracy']:.4f}")
    print(f"💾 Best Parameters: {results['best_params']}")
    print(f"📤 Submission: {submission_filename}")
    print("="*60)