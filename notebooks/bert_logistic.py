# ----------------------------- IMPORTS ------------------------------------
import os
import re
import warnings
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from collections import Counter
import joblib

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

# BERT
import torch
from transformers import AutoTokenizer, AutoModel

# ML & utils
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------- DATA LOADING ------------------------------------
def read_texts_from_dir(dir_path):
    """
    Reads the texts from a given directory and saves them in the pd.DataFrame with columns ['id', 'file_1', 'file_2'].

    Params:
      dir_path (str): path to the directory with data
    """
    # Count number of directories in the provided path
    dir_count = sum(
        os.path.isdir(os.path.join(root, d))
        for root, dirs, _ in os.walk(dir_path)
        for d in dirs
    )
    data = [0 for _ in range(dir_count)]
    print(f"Number of directories: {dir_count}")

    # For each directory, read both file_1.txt and file_2.txt and save results to the list
    i = 0
    for folder_name in sorted(os.listdir(dir_path)):
        folder_path = os.path.join(dir_path, folder_name)
        if os.path.isdir(folder_path):
            try:
                with open(
                    os.path.join(folder_path, "file_1.txt"), "r", encoding="utf-8"
                ) as f1:
                    text1 = f1.read().strip()
                with open(
                    os.path.join(folder_path, "file_2.txt"), "r", encoding="utf-8"
                ) as f2:
                    text2 = f2.read().strip()
                index = int(folder_name[-4:])
                data[i] = (index, text1, text2)
                i += 1
            except Exception as e:
                print(f"Error reading directory {folder_name}: {e}")

    # Change list with results into pandas DataFrame
    df = pd.DataFrame(data, columns=["id", "file_1", "file_2"]).set_index("id")
    return df

# ----------------------- TEXT PRE-PROCESSING ------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """Làm sạch + chuẩn hoá một câu văn."""
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r"http\S+", " ", text)  # xoá URL
    text = re.sub(r"\d+", " NUM ", text)  # thay số = token NUM
    text = re.sub(r"[^\w\s]", " ", text)  # bỏ punctuation
    text = text.lower()
    tokens = [lemmatizer.lemmatize(tok)
              for tok in word_tokenize(text)
              if tok.isalpha() and tok not in stop_words and len(tok) > 2]
    return " ".join(tokens)

def preprocess_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Áp dụng clean_text cho 2 cột file_1 và file_2."""
    tqdm.pandas(desc="Cleaning text")
    df = df.copy()
    df["file_1_clean"] = df["file_1"].progress_apply(clean_text)
    df["file_2_clean"] = df["file_2"].progress_apply(clean_text)
    return df

# ----------------------- RULE-BASED FEATURES ------------------------------
def compute_rule_based_features(text: str) -> dict:
    """Tính toán các đặc trưng rule-based cho một văn bản."""
    if not isinstance(text, str):
        text = ""
    
    # Độ dài và cấu trúc
    char_count = len(text)
    word_count = len(word_tokenize(text))
    sentence_count = len(sent_tokenize(text))
    avg_sentence_length = word_count / max(sentence_count, 1)
    
    # Tính trang trọng: Số liệu, đơn vị, từ viết tắt
    number_count = len(re.findall(r'\d+', text))
    unit_count = len(re.findall(r'\b(?:km|cm|m|s|kg|g|Hz|K|A|deg|arcsec|dex|A|petabytes|terabytes)\b', text, re.I))
    acronym_count = len(re.findall(r'\b[A-Z]{2,}\b', text))
    uppercase_word_count = len(re.findall(r'\b[A-Z][A-Z]+\b', text))
    
    # Dấu chấm than và giọng văn không trang trọng
    exclamation_count = text.count('!')
    
    # Lặp từ (repetition)
    words = word_tokenize(text.lower())
    repetition_score = sum([count for word, count in Counter(words).items() if count > 3]) / max(word_count, 1)
    
    # Từ khóa bất thường (dấu hiệu FAKE)
    china_count = text.lower().count('china')
    dinosaur_count = text.lower().count('dinosaur')
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_sentence_length': avg_sentence_length,
        'number_count': number_count,
        'unit_count': unit_count,
        'acronym_count': acronym_count,
        'uppercase_word_count': uppercase_word_count,
        'exclamation_count': exclamation_count,
        'repetition_score': repetition_score,
        'china_count': china_count,
        'dinosaur_count': dinosaur_count
    }

def extract_rule_based_features(df: pd.DataFrame) -> np.ndarray:
    """Tạo ma trận đặc trưng rule-based, bao gồm diff giữa file_1 và file_2."""
    features = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting rule-based features"):
        f1 = compute_rule_based_features(row['file_1'])
        f2 = compute_rule_based_features(row['file_2'])
        
        # Tạo diff features
        diff = {k: f1[k] - f2[k] for k in f1}
        
        # Kết hợp f1, f2, diff thành một vector
        feature_vector = list(f1.values()) + list(f2.values()) + list(diff.values())
        features.append(feature_vector)
    
    return np.array(features).astype(np.float32)

# ----------------------- STATISTICAL FEATURES -----------------------------
def extract_statistical_features(df: pd.DataFrame) -> np.ndarray:
    """Sinh đặc trưng thống kê/đếm đơn giản ở dạng dense numpy array."""
    len_1 = df["file_1"].str.len()
    len_2 = df["file_2"].str.len()
    words_1 = df["file_1"].str.split().apply(len)
    words_2 = df["file_2"].str.split().apply(len)

    features = pd.DataFrame({
        "len_diff": (len_1 - len_2).abs(),
        "word_diff": (words_1 - words_2).abs(),
        "len_ratio": (len_1 + 1) / (len_2 + 1),
        "words_ratio": (words_1 + 1) / (words_2 + 1),
        "avg_word_len_1": df["file_1"].apply(lambda x: np.mean([len(w) for w in x.split()]) if x else 0),
        "avg_word_len_2": df["file_2"].apply(lambda x: np.mean([len(w) for w in x.split()]) if x else 0),
    })
    return features.values.astype(np.float32)

# ----------------------- BERT FEATURES ------------------------------------
class BERTFeatureExtractor:
    def __init__(self, model_name='bert-base-uncased', max_length=512, device=None):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading BERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def get_bert_embeddings(self, texts):
        """Extract BERT embeddings for a list of texts."""
        embeddings = []
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Extracting BERT features"):
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                outputs = self.model(**inputs)
                
                # Use [CLS] token embedding as sentence representation
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embedding.flatten())
        
        return np.array(embeddings)

def extract_bert_features(df: pd.DataFrame, bert_extractor: BERTFeatureExtractor) -> np.ndarray:
    """Extract BERT features for both file_1 and file_2, including diff features."""
    print("Extracting BERT features for file_1...")
    bert_f1 = bert_extractor.get_bert_embeddings(df['file_1'].tolist())
    
    print("Extracting BERT features for file_2...")
    bert_f2 = bert_extractor.get_bert_embeddings(df['file_2'].tolist())
    
    # Create difference features
    bert_diff = bert_f1 - bert_f2
    
    # Concatenate all BERT features
    bert_features = np.concatenate([bert_f1, bert_f2, bert_diff], axis=1)
    
    return bert_features.astype(np.float32)

# ----------------------- PREPARE DATA FOR MODEL ---------------------------
def prepare_data_for_model(df: pd.DataFrame, bert_extractor: BERTFeatureExtractor = None, fit_bert: bool = False):
    """
    Chuẩn bị dữ liệu cho model bằng cách áp dụng tất cả các bước tiền xử lý và trích xuất features.
    
    Args:
        df: DataFrame chứa dữ liệu thô
        bert_extractor: BERT feature extractor, nếu None sẽ tạo mới
        fit_bert: Có tạo BERT extractor mới hay không
    
    Returns:
        feature_matrix: Ma trận features đã kết hợp
        bert_extractor: BERT extractor (để dùng cho test set)
    """
    # 1. Preprocess text
    print("Step 1: Preprocessing text...")
    df_processed = preprocess_text_columns(df)
    
    # 2. Extract rule-based features
    print("Step 2: Extracting rule-based features...")
    rule_features = extract_rule_based_features(df_processed)
    
    # 3. Extract statistical features
    print("Step 3: Extracting statistical features...")
    stat_features = extract_statistical_features(df_processed)
    
    # 4. Extract BERT features
    print("Step 4: Extracting BERT features...")
    if bert_extractor is None or fit_bert:
        bert_extractor = BERTFeatureExtractor()
    
    bert_features = extract_bert_features(df, bert_extractor)
    
    # 5. Combine all features
    print("Step 5: Combining all features...")
    # Convert to sparse for efficient concatenation
    rule_sparse = csr_matrix(rule_features)
    stat_sparse = csr_matrix(stat_features)
    bert_sparse = csr_matrix(bert_features)
    
    feature_matrix = hstack([bert_sparse, rule_sparse, stat_sparse]).tocsr()
    
    print(f"Final feature matrix shape: {feature_matrix.shape}")
    return feature_matrix, bert_extractor

# ----------------------- MODEL TRAINING -----------------------------------
def train_and_evaluate(X, y, clf_model, param_grid):
    """Cross-validate + grid search, trả về best model."""
    clf = Pipeline([
        ("select", SelectKBest(f_classif, k=min(20_000, X.shape[1]))),
        ("clf", clf_model)
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring="accuracy",
        cv=skf,
        n_jobs=-1,
        verbose=2
    )
    grid.fit(X, y)
    print(f"Best CV accuracy: {grid.best_score_:.4f}")
    print(f"Best params: {grid.best_params_}")
    return grid.best_estimator_

def save_model_and_artifacts(model, bert_extractor, feature_names=None, save_dir="models", tune_dir="tune"):
    """Save trained model and related artifacts."""
    save_dir = Path(save_dir)
    tune_dir = Path(tune_dir)
    save_dir.mkdir(exist_ok=True)
    tune_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = save_dir / "logistic_regression_bert.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save BERT extractor
    bert_path = save_dir / "bert_extractor.pkl"
    joblib.dump(bert_extractor, bert_path)
    print(f"BERT extractor saved to: {bert_path}")
    
    # Save feature names if provided
    if feature_names:
        features_path = save_dir / "feature_names.pkl"
        joblib.dump(feature_names, features_path)
        print(f"Feature names saved to: {features_path}")
    
    # Save hyperparameter tuning results to tune directory
    if hasattr(model, 'best_params_'):
        tune_results = {
            'best_params': model.best_params_,
            'best_score': model.best_score_,
            'cv_results': model.cv_results_
        }
        tune_path = tune_dir / "hyperparameter_results.pkl"
        joblib.dump(tune_results, tune_path)
        print(f"Tuning results saved to: {tune_path}")

def main():
    """Main training pipeline."""
    # Paths
    train_path = "/home/thangquang09/CODE/CTAI_MachineLearning/data/fake-or-real-the-impostor-hunt/data/train"
    test_path = "/home/thangquang09/CODE/CTAI_MachineLearning/data/fake-or-real-the-impostor-hunt/data/test"
    gt_path = "/home/thangquang09/CODE/CTAI_MachineLearning/data/fake-or-real-the-impostor-hunt/data/train.csv"
    
    # Load data
    print("Loading data...")
    df_train = read_texts_from_dir(train_path)
    df_test = read_texts_from_dir(test_path)
    df_train_gt = pd.read_csv(gt_path)
    y_train = df_train_gt["real_text_id"].values
    
    # Prepare data for model
    print("Preparing training data...")
    X_train, bert_extractor = prepare_data_for_model(df_train, fit_bert=True)
    
    print("Preparing test data...")
    X_test, _ = prepare_data_for_model(df_test, bert_extractor=bert_extractor, fit_bert=False)
    
    # Define model and parameters
    clf_model = LogisticRegression(max_iter=5_000, solver="liblinear", n_jobs=-1, random_state=SEED)
    param_grid = {
        "select__k": [10_000, 15_000, 20_000],
        "clf__C": [0.1, 0.5, 1, 2, 5],
        "clf__penalty": ["l1", "l2"]
    }
    
    # Train model
    print("Training model...")
    model = train_and_evaluate(X_train, y_train, clf_model, param_grid)
    
    # Retrain on full data
    print("Retraining on full data...")
    model.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    test_pred = model.predict(X_test)
    
    # Save model and artifacts
    save_model_and_artifacts(model, bert_extractor, save_dir="models", tune_dir="tune")
    
    # Create submission
    submission = pd.DataFrame({
        "id": df_test.index,
        "real_text_id": test_pred.astype(int)
    }).sort_values("id")
    
    submission_path = Path("submission_bert_logistic.csv")
    submission.to_csv(submission_path, index=False)
    print(f"✅ Submission saved to {submission_path.resolve()}")
    
    return model, bert_extractor

if __name__ == "__main__":
    model, bert_extractor = main()