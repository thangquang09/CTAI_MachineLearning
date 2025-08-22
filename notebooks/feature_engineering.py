# ----------------------------- IMPORTS ------------------------------------
import math
import os
import random
import re
import warnings
from collections import Counter
from multiprocessing import Pool, cpu_count


# NLP
import nltk
import numpy as np
import pandas as pd
import textstat

# Deep Learning
import torch


from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# DetectorFactory.seed = SEED

#----------------------------- BASE FUNCTION ------------------------------

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

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # 2. Xóa các ký tự không mong muốn nhưng giữ lại ' và - nếu ở trong từ
    #   - Cho phép: chữ, số, khoảng trắng, ', -
    text = re.sub(r"[^a-z0-9\s'\-]", " ", text)
    # 3. Chuẩn hoá khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------------- FEATURE ENGINEERING ------------------------------

def compute_advanced_features(text: str) -> dict:
    """
    Tính toán các features đếm và tỉ lệ, ... cho real or fake detection.
    """
    if not isinstance(text, str) or not text.strip():
        return {f: 0 for f in [
            'unique_word_count_ratio', 'latin_ratio', 'digit_count', 
            'flesch_reading_ease', 'dale_chall_readability',
            'coleman_liau_index', 'short_word_count_ratio', 'uppercase_ratio',
            'english_ratio', 'perplexity_score', 'sentence_count',
            'word_count', 'avg_word_length'
        ]}
    
    # Basic text stats
    cleaned_text = clean_text(text)
    words = cleaned_text.split()
    word_count = len(words)
    unique_words = len(set(words))
    char_count = len(text)

    # 1. Unique word count ratio (top feature!)
    unique_word_count_ratio = unique_words / max(word_count, 1)
    
    # 2. Latin ratio (character-based)
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    latin_ratio = latin_chars / max(char_count, 1)
    
    # 3. Digit count
    digit_count = len(re.findall(r'\d', text))
    
    # 4. Readability scores using textstat
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    dale_chall_readability = textstat.dale_chall_readability_score(text)
    coleman_liau_index = textstat.coleman_liau_index(text)

    # 5. Short word ratio
    short_words = [w for w in words if len(w) <= 3]
    short_word_count_ratio = len(short_words) / max(word_count, 1)

    # 6. Uppercase ratio
    uppercase_chars = len(re.findall(r'[A-Z]', text))
    uppercase_ratio = uppercase_chars / max(char_count, 1)

    # 7. English ratio (approximate using common English patterns)
    english_words = len(re.findall(r'\b[a-zA-Z]+\b', cleaned_text))
    english_ratio = english_words / max(word_count, 1)

    # 8. Simple perplexity approximation (entropy-based)
    def calculate_perplexity(words):
        if not words:
            return 0
        
        word_freq = Counter(words)
        total_words = len(words)  # Dùng total occurrences thay vì unique
        
        # Calculate entropy
        entropy = 0
        for freq in word_freq.values():
            prob = freq / total_words
            entropy -= prob * math.log2(prob)
        
        return 2 ** entropy if entropy > 0 else 1
    perplexity_score = calculate_perplexity(words)


    # 9. Sentence count
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])

    # 10. Average word length
    avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
    
    return {
        'unique_word_count_ratio': unique_word_count_ratio,
        'latin_ratio': latin_ratio,
        'digit_count': digit_count,
        'flesch_reading_ease': flesch_reading_ease,
        'dale_chall_readability': dale_chall_readability,
        'coleman_liau_index': coleman_liau_index,
        'short_word_count_ratio': short_word_count_ratio,
        'uppercase_ratio': uppercase_ratio,
        'english_ratio': english_ratio,
        'perplexity_score': perplexity_score,
        'sentence_count': sentence_count,
        'word_count': word_count,
        'avg_word_length': avg_word_length
    }

def process_row_top_features(row_data):
    """Process single row for top features extraction (for multiprocessing)."""
    text1, text2 = row_data
    
    # Get features for both texts
    f1 = compute_advanced_features(text1)
    f2 = compute_advanced_features(text2)
    
    # Create difference and ratio features (theo pattern từ biểu đồ)
    feature_row = []
    
    # 1. unique_word_count_ratio (tỷ lệ giữa file1 và file2)
    unique_ratio = (f1['unique_word_count_ratio'] + 1e-8) / (f2['unique_word_count_ratio'] + 1e-8)
    unique_ratio = np.clip(unique_ratio, 0.1, 10.0)
    feature_row.append(unique_ratio)
    
    # 2. latin_ratio_diff (signed difference)
    latin_ratio_diff = f1['latin_ratio'] - f2['latin_ratio']
    feature_row.append(latin_ratio_diff)
    
    # 3. digit_count_diff (signed difference)
    digit_count_diff = f1['digit_count'] - f2['digit_count']
    feature_row.append(digit_count_diff)
    
    # 4. semantic_similarity 
    def cosine_similarity(text1, text2):
        words1 = Counter(text1.lower().split())
        words2 = Counter(text2.lower().split())
        # Get common words
        common_words = set(words1.keys()) & set(words2.keys())
        if not common_words:
            return 0.0
        # Calculate dot product and norms
        dot_product = sum(words1[word] * words2[word] for word in common_words)
        norm1 = math.sqrt(sum(count**2 for count in words1.values()))
        norm2 = math.sqrt(sum(count**2 for count in words2.values()))
        
        return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0
    semantic_similarity = cosine_similarity(text1, text2)
    feature_row.append(semantic_similarity)
    
    
    # 5. perplexity_diff (signed difference)
    perplexity_diff = f1['perplexity_score'] - f2['perplexity_score']
    feature_row.append(perplexity_diff)

    # 6. flesch_reading_ease_ratio
    flesch_ratio = (f1['flesch_reading_ease'] + 100) / (f2['flesch_reading_ease'] + 100)
    flesch_ratio = np.clip(flesch_ratio, 0.1, 10.0)
    feature_row.append(flesch_ratio)

    # 7. short_word_count_ratio
    short_ratio = (f1['short_word_count_ratio'] + 1e-8) / (f2['short_word_count_ratio'] + 1e-8)
    short_ratio = np.clip(short_ratio, 0.1, 10.0)
    feature_row.append(short_ratio)

    # 8. readability_avg_ratio
    readability_avg_1 = (f1['flesch_reading_ease'] + f1['dale_chall_readability']) / 2
    readability_avg_2 = (f2['flesch_reading_ease'] + f2['dale_chall_readability']) / 2
    readability_avg_ratio = (readability_avg_1 + 50) / (readability_avg_2 + 50)
    readability_avg_ratio = np.clip(readability_avg_ratio, 0.1, 10.0)
    feature_row.append(readability_avg_ratio)

    # 9. dale_chall_readability_score_diff (signed difference)
    dale_chall_diff = f1['dale_chall_readability'] - f2['dale_chall_readability']
    feature_row.append(dale_chall_diff)

    # 10. sentence_count_diff (signed difference)
    sentence_count_diff = f1['sentence_count'] - f2['sentence_count']
    feature_row.append(sentence_count_diff)

    # 11. perplexity_ratio
    perplexity_ratio = (f1['perplexity_score'] + 1e-8) / (f2['perplexity_score'] + 1e-8)
    perplexity_ratio = np.clip(perplexity_ratio, 0.1, 10.0)
    feature_row.append(perplexity_ratio)

    # 12. coleman_liau_index_diff (signed difference)
    coleman_diff = f1['coleman_liau_index'] - f2['coleman_liau_index']
    feature_row.append(coleman_diff)

    # 13. english_ratio_diff (signed difference)
    english_ratio_diff = f1['english_ratio'] - f2['english_ratio']
    feature_row.append(english_ratio_diff)

    # 14. word_count_diff (signed difference)
    word_count_diff = f1['word_count'] - f2['word_count']
    feature_row.append(word_count_diff)

    # 15. uppercase_ratio_diff (signed difference)
    uppercase_ratio_diff = f1['uppercase_ratio'] - f2['uppercase_ratio']
    feature_row.append(uppercase_ratio_diff)

    # 16. latin_ratio_ratio
    latin_ratio_ratio = (f1['latin_ratio'] + 1e-8) / (f2['latin_ratio'] + 1e-8)
    latin_ratio_ratio = np.clip(latin_ratio_ratio, 0.1, 10.0)
    feature_row.append(latin_ratio_ratio)

    # 17. english_ratio_ratio
    english_ratio_ratio = (f1['english_ratio'] + 1e-8) / (f2['english_ratio'] + 1e-8)
    english_ratio_ratio = np.clip(english_ratio_ratio, 0.1, 10.0)
    feature_row.append(english_ratio_ratio)

    # Add individual features as well
    feature_row.extend([
        f1['unique_word_count_ratio'], f2['unique_word_count_ratio'],
        f1['latin_ratio'], f2['latin_ratio'],
        f1['flesch_reading_ease'], f2['flesch_reading_ease'],
        f1['perplexity_score'], f2['perplexity_score']
    ])
    
    return feature_row

def extract_top_features(df: pd.DataFrame, n_jobs: int = None) -> np.ndarray:
    """Extract top features theo biểu đồ importance with multiprocessing."""
    if n_jobs is None:
        n_jobs = min(cpu_count(), 8)  # Limit to 8 cores max to avoid memory issues
    
    # Prepare data for multiprocessing
    row_data = [(row['file_1'], row['file_2']) for _, row in df.iterrows()]
    
    print(f"Using {n_jobs} cores for top features extraction...")
    
    if len(row_data) < 100 or n_jobs == 1:
        # For small datasets, use single process to avoid overhead
        features = []
        for data in tqdm(row_data, desc="Extracting top features (single-threaded)"):
            features.append(process_row_top_features(data))
    else:
        # Use multiprocessing for larger datasets
        with Pool(n_jobs) as pool:
            features = list(tqdm(
                pool.imap(process_row_top_features, row_data),
                total=len(row_data),
                desc="Extracting top features (multi-threaded)"
            ))
    
    return np.array(features).astype(np.float32)


# ----------------------- Extracting rule-based features ---------------------------
def compute_rule_based_features(text: str) -> dict:
    """Tính toán các đặc trưng rule-based cho một văn bản."""
    if not isinstance(text, str):
        text = ""

    # Existing features...
    cleaned_text = clean_text(text)
    word_count = len(cleaned_text.split())

    # === NEW FEATURES FOR FAKE DETECTION ===

    # 1. Multi-script detection (Garbage text pattern)
    cyrillic_count = len(re.findall(r"[\u0400-\u04FF]", text))  # Russian
    arabic_count = len(re.findall(r"[\u0600-\u06FF]", text))  # Arabic
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))  # Chinese
    emoji_count = len(
        re.findall(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]", text
        )
    )

    # Mixed script score
    script_diversity = sum(
        1 for count in [cyrillic_count, arabic_count, chinese_count] if count > 0
    )

    # 2. Storytelling pattern detection
    exclamation_ratio = text.count("!") / max(len(text), 1)
    question_ratio = text.count("?") / max(len(text), 1)

    # Informal language indicators
    informal_words = ["forget", "wow", "amazing", "incredible", "magic", "unicorn"]
    informal_count = sum(text.lower().count(word) for word in informal_words)

    # Bold/emphasis markers (markdown style)
    bold_count = text.count("**") + text.count("__")

    # 3. Tone analysis
    first_person_count = len(re.findall(r"\b(I|we|our|my|mine)\b", text, re.I))
    second_person_count = len(re.findall(r"\b(you|your|yours)\b", text, re.I))

    # Scientific vs casual tone
    scientific_terms = [
        "observation",
        "analysis",
        "telescope",
        "data",
        "measurement",
        "survey",
    ]
    scientific_count = sum(text.lower().count(term) for term in scientific_terms)

    # 4. Inconsistency detection
    # Sudden language change (character encoding issues)
    unicode_control_chars = len(re.findall(r"[\u0000-\u001F\u007F-\u009F]", text))

    # Suspicious name patterns (China relay network, etc.)
    suspicious_entities = ["china relay", "rainbow unicorn", "santa", "north pole"]
    suspicious_count = sum(text.lower().count(entity) for entity in suspicious_entities)


    return {
        # Multi-script features
        "cyrillic_count": cyrillic_count,
        "arabic_count": arabic_count,
        "chinese_count": chinese_count,
        "emoji_count": emoji_count,
        "script_diversity": script_diversity,
        # Storytelling features
        "exclamation_ratio": exclamation_ratio,
        "question_ratio": question_ratio,
        "informal_count": informal_count,
        "bold_count": bold_count,
        # Tone features
        "first_person_count": first_person_count,
        "second_person_count": second_person_count,
        "scientific_count": scientific_count,
        # Inconsistency features
        "unicode_control_chars": unicode_control_chars,
        "suspicious_count": suspicious_count,
        # Existing features
        "number_count": len(re.findall(r"\d+", text)),
        "unit_count": len(
            re.findall(
                r"\b(?:km|cm|m|s|kg|g|Hz|K|A|deg|arcsec|dex|A|petabytes|terabytes)\b",
                text,
                re.I,
            )
        ),
        "acronym_count": len(re.findall(r"\b[A-Z]{2,}\b", text)),
        "uppercase_word_count": len(re.findall(r"\b[A-Z][A-Z]+\b", text)),
        "exclamation_count": text.count("!"),
        "repetition_score": sum(
            [
                count
                for word, count in Counter(cleaned_text.lower().split()).items()
                if count > 3
            ]
        )
        / max(word_count, 1),
    }


def process_row_rule_based(row_data):
    """Process single row for rule-based features extraction (for multiprocessing)."""
    text1, text2 = row_data
    f1 = compute_rule_based_features(text1)
    f2 = compute_rule_based_features(text2)
    
    # Tạo diff features
    diff = {k: f1[k] - f2[k] for k in f1}
    
    # Kết hợp f1, f2, diff thành một vector
    feature_vector = list(f1.values()) + list(f2.values()) + list(diff.values())
    return feature_vector

def extract_rule_based_features(df: pd.DataFrame, n_jobs: int = None) -> np.ndarray:
    """Tạo ma trận đặc trưng rule-based với multiprocessing."""
    if n_jobs is None:
        n_jobs = min(cpu_count(), 8)
    
    # Prepare data for multiprocessing
    row_data = [(row['file_1'], row['file_2']) for _, row in df.iterrows()]
    
    print(f"Using {n_jobs} cores for rule-based features extraction...")
    
    if len(row_data) < 100 or n_jobs == 1:
        # For small datasets, use single process
        features = []
        for data in tqdm(row_data, desc="Extracting rule-based features (single-threaded)"):
            features.append(process_row_rule_based(data))
    else:
        # Use multiprocessing for larger datasets
        with Pool(n_jobs) as pool:
            features = list(tqdm(
                pool.imap(process_row_rule_based, row_data),
                total=len(row_data),
                desc="Extracting rule-based features (multi-threaded)"
            ))

    return np.array(features).astype(np.float32)

# ----------------------- PREPARE DATA FOR MODEL ---------------------------
class EmbeddingExtractor:
    def __init__(self, model_name: str, max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)

    def get_embedding(self, text : str ) -> np.ndarray:
        cleaned_text = clean_text(text)
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        with torch.no_grad():
            embeddings = self.model(inputs['input_ids'].to(self.device), inputs['attention_mask'].to(self.device)).last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings


def extract_embedding_features(df: pd.DataFrame, embedding_extractor: EmbeddingExtractor) -> np.ndarray:
    """Extract embedding features from the text data using the embedding extractor."""
    embeddings = []
    for text in tqdm(df['text'], desc="Extracting embeddings"):
        embedding = embedding_extractor.get_embedding(text)
        embeddings.append(embedding)
    return np.vstack(embeddings)

# ----------------------- PREPARE DATA FOR MODEL ---------------------------
def prepare_data_for_model(
    df: pd.DataFrame,
    embedding_extractor: EmbeddingExtractor = None,
    model_name: str = "bert-base-uncased",
    n_jobs: int = None,
):
    """
    Chuẩn bị dữ liệu cho model với focus vào top features theo importance chart.

    Args:
        df: DataFrame chứa dữ liệu thô
        embedding_extractor: Universal embedding extractor, nếu None sẽ tạo mới
        model_name: Tên model embedding để sử dụng
        n_jobs: Số lượng CPU cores để sử dụng cho multiprocessing

    Returns:
        feature_matrix: Ma trận features đã kết hợp
        embedding_extractor: Embedding extractor (để dùng cho test set)
    """
    if n_jobs is None:
        n_jobs = min(cpu_count(), 8)  # Default to 8 cores max
    
    print(f"Using {n_jobs} CPU cores for feature extraction...")
    
    # 0. clean text
    df['cleaned_file_1'] = df['file_1'].apply(clean_text)
    df['cleaned_file_2'] = df['file_2'].apply(clean_text)
    df['text'] = '[CLS] ' + df['cleaned_file_1'] + " [SEP] " + df['cleaned_file_2']

    # 1. Extract top features (most important) - with multiprocessing
    print("Step 1: Extracting top importance features...")
    top_features = extract_top_features(df, n_jobs=n_jobs)
    
    # 2. Extract rule-based features (existing) - with multiprocessing
    print("Step 2: Extracting rule-based features...")
    rule_features = extract_rule_based_features(df, n_jobs=n_jobs)
    
    # 4. Extract embedding features (lighter approach) - handled by embedding extractor
    print("Step 4: Extracting embedding features...")
    if embedding_extractor is None:
        embedding_extractor = EmbeddingExtractor(
            model_name=model_name,
            max_length=512,
        )
    
    embedding_features = extract_embedding_features(df, embedding_extractor)
    
    # 6. Combine all features with priority on top features
    print("Step 6: Combining features...")  
    # Priority: top features first, then others
    feature_matrix = np.hstack([top_features, rule_features, embedding_features])

    print(f"Final feature matrix shape: {feature_matrix.shape}")
    print(f"Top features: {top_features.shape[1]}, Rule: {rule_features.shape[1]}, Embedding: {embedding_features.shape[1]}")

    return feature_matrix

if __name__ == "__main__":
    print("Loading data...")
    df_train = read_texts_from_dir("./data/train")
    df_test = read_texts_from_dir("./data/test")
    df_train_gt = pd.read_csv("./data/train.csv")
    y_train = df_train_gt["real_text_id"].values
    df_train['label'] = df_train_gt["real_text_id"]
    print("Data loading complete.")
    print('Feature engineering...')
    embedding_extractor = EmbeddingExtractor(
        model_name="bert-base-uncased",
        max_length=512,
    )
    feature_matrix = extract_embedding_features(df_train, embedding_extractor=embedding_extractor)
    print(f"Feature matrix shape: {feature_matrix.shape}")