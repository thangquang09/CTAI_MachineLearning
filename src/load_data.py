import math
import os
import re
from collections import Counter

import pandas as pd

# Global variables for lazy initialization
_ENGLISH_WORDS = None
_NLTK_DOWNLOADED = False

def _ensure_nltk_downloads():
    """Lazy download of NLTK data."""
    global _NLTK_DOWNLOADED
    if not _NLTK_DOWNLOADED:
        import nltk
        nltk.download("words", quiet=True)
        nltk.download("punkt", quiet=True)
        _NLTK_DOWNLOADED = True

def _get_english_words():
    """Lazy initialization of English words set."""
    global _ENGLISH_WORDS
    if _ENGLISH_WORDS is None:
        _ensure_nltk_downloads()
        from nltk.corpus import words as nltk_words
        _ENGLISH_WORDS = set(nltk_words.words())
    return _ENGLISH_WORDS

TRAIN_DATA_FOLDER_PATH = "./data/fake-or-real-the-impostor-hunt/data/train"
TEST_DATA_FOLDER_PATH = "./data/fake-or-real-the-impostor-hunt/data/test"
GT_DATA_PATH = "./data/fake-or-real-the-impostor-hunt/data/train.csv"


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
    """Làm sạch + chuẩn hoá một câu văn."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+", " ", text)  # xoá URL
    text = re.sub(r"\d+", " NUM ", text)  # thay số = token NUM
    text = re.sub(r"[^\w\s]", " ", text)  # bỏ punctuation
    text = text.lower()
    return text

def preprocessing_df(df):
    df['file_1'] = df['file_1'].apply(clean_text)
    df['file_2'] = df['file_2'].apply(clean_text)
    return df

def calculate_english_ratio(text_words: list) -> float:
    """
    Tính toán tỷ lệ các từ tiếng Anh dựa trên danh sách từ đã được tokenize.
    Sử dụng một tập hợp từ vựng để tăng tốc độ.
    """
    if not text_words:
        return 0.0
    
    english_words = _get_english_words()
    english_word_count = sum(1 for word in text_words if word.lower() in english_words)
            
    return english_word_count / len(text_words)

def compute_rule_based_features(text: str) -> dict:
    """Tính toán các đặc trưng rule-based cho một văn bản."""
    if not isinstance(text, str):
        text = ""

    # Lazy import NLTK tokenizers
    from nltk.tokenize import sent_tokenize, word_tokenize
    
    # Tokenize một lần và tái sử dụng
    words = word_tokenize(text)
    sentences = sent_tokenize(text)

    # 1. Basic text metrics
    char_count = len(text)
    word_count = len(words)
    sentence_count = len(sentences)
    avg_sentence_length = word_count / max(sentence_count, 1)

    # === NEW FEATURES FOR FAKE DETECTION ===

    # 2. Multi-script detection (Binary classification)
    cyrillic_count = len(re.findall(r"[\u0400-\u04FF]", text))  # Russian
    arabic_count = len(re.findall(r"[\u0600-\u06FF]", text))  # Arabic
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))  # Chinese

    # 3. Script presence indicators (Binary features)
    has_non_english_script = 1 if (cyrillic_count > 0 or arabic_count > 0 or chinese_count > 0) else 0
    has_mixed_scripts = 1 if sum(1 for count in [cyrillic_count, arabic_count, chinese_count] if count > 0) > 1 else 0
    
    # 4. Inconsistency detection features
    # 4.1. Sudden language change (character encoding issues)
    unicode_control_chars = len(re.findall(r"[\u0000-\u001F\u007F-\u009F]", text))

    # 4.2. Suspicious name patterns detection
    # suspicious_entities = ["china relay", "rainbow unicorn", "santa", "north pole"]
    # suspicious_count = sum(text.lower().count(entity) for entity in suspicious_entities)

    # 5. Language detection feature
    english_word_ratio = calculate_english_ratio(words)

    # 6. Statistical text features
    word_freq = Counter(words)
    total_words = sum(word_freq.values())
    perplexity_score = 0
    if total_words > 0:
        entropy = -sum((freq/total_words) * math.log2(freq/total_words) 
                        for freq in word_freq.values())
        perplexity_score = 2 ** entropy
    
    # FIX: TTR (Type-Token Ratio) = Số từ duy nhất / Tổng số từ
    unique_words = len(set(words))  # Số từ duy nhất
    ttr_ratio = unique_words / max(word_count, 1)    
        
    return {
        # Basic text metrics
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        # Language detection
        "english_word_ratio": english_word_ratio,
        # Script presence features (Binary)
        "has_non_english_script": has_non_english_script,
        "has_mixed_scripts": has_mixed_scripts,
        # Inconsistency features
        "unicode_control_chars": unicode_control_chars,
        # "suspicious_count": suspicious_count,
        # Additional text characteristics
        "num_count": text.count("num"),
        # Content repetition pattern
        "repetition_score": sum(
            [
                count
                for word, count in Counter(word_tokenize(text.lower())).items()
                if count > 3
            ]
        )
        / max(word_count, 1),
        # Statistical complexity metrics
        "perplexity_score": perplexity_score,
        "ttr_ratio": ttr_ratio,
        # Trả về words và word_freq để tái sử dụng
        "tokenized_words": words,
        "word_freq": word_freq,
    }
    
def feature_engineering(df):
    # Extract features for each text file
    features_list1 = df['file_1'].apply(compute_rule_based_features).tolist()
    features_list2 = df['file_2'].apply(compute_rule_based_features).tolist()
    
    # Tách riêng các features cần cho DataFrame và các đối tượng cần để tái sử dụng
    df_features1 = pd.DataFrame([{k: v for k, v in f.items() if k not in ['tokenized_words', 'word_freq']} for f in features_list1])
    df_features2 = pd.DataFrame([{k: v for k, v in f.items() if k not in ['tokenized_words', 'word_freq']} for f in features_list2])

    # Lấy ra danh sách các từ đã được tokenize để tái sử dụng
    words1_list = [f['tokenized_words'] for f in features_list1]
    words2_list = [f['tokenized_words'] for f in features_list2]
    
    # Add prefix to column names to distinguish between files
    df_features1.columns = ['file1_' + col for col in df_features1.columns]
    df_features2.columns = ['file2_' + col for col in df_features2.columns]
    
    # Calculate differences between features
    diff_features = pd.DataFrame()
    base_cols = [col.replace('file1_', '') for col in df_features1.columns]
    for base_col in base_cols:
        col1 = 'file1_' + base_col
        col2 = 'file2_' + base_col
        
        diff_col = f'diff_{base_col}'
        diff_features[diff_col] = df_features1[col1].values - df_features2[col2].values
        
        # Add ratio features for meaningful metrics
        if base_col in ['word_count', 'char_count', 'sentence_count', 'perplexity_score', 'ttr_ratio']:
            ratio_col = f'ratio_{base_col}'
            # Avoid division by zero
            diff_features[ratio_col] = df_features1[col1].values / (df_features2[col2].values + 1e-6)
    
    # Tối ưu tính toán cosine similarity
    # Sử dụng lại kết quả tokenize thay vì tính toán lại
    cosine_sims = []
    for words1, words2 in zip(words1_list, words2_list):
        counter1 = Counter(words1)
        counter2 = Counter(words2)
        intersection_sum = sum((counter1 & counter2).values())
        min_len = min(len(words1), len(words2))
        sim = intersection_sum / max(1, min_len)
        cosine_sims.append(sim)
    
    diff_features['cosine_sim_word_counts'] = cosine_sims
    
    # Merge all features with original dataframe
    # Đảm bảo index của các DataFrame mới khớp với index của df gốc
    df_features1.index = df.index
    df_features2.index = df.index
    diff_features.index = df.index
    
    result_df = pd.concat([df, df_features1, df_features2, diff_features], axis=1)
    
    # Xóa các cột không cần thiết cho mô hình
    result_df = result_df.drop(columns=['file1_tokenized_words', 'file1_word_freq', 'file2_tokenized_words', 'file2_word_freq'], errors='ignore')

    return result_df


# ==============================================================================
# === CÁC HÀM TÍNH TOÁN FEATURES DỰA TRÊN VECTOR HÓA VÀ EMBEDDINGS ===
# ==============================================================================

# Lưu ý: Để dùng GloVe, bạn cần tải file pre-trained, ví dụ từ: https://nlp.stanford.edu/projects/glove/

def create_vectorizer_features(train_texts: pd.Series, val_texts: pd.Series, test_texts: pd.Series, vectorizer, svd_components: int, prefix: str) -> tuple:
    """
    Hàm chung để tạo features từ TF-IDF hoặc BoW và giảm chiều dữ liệu bằng SVD.
    FIX: Chỉ fit trên train data để tránh data leakage.
    """
    # Lazy import sklearn components
    from sklearn.decomposition import TruncatedSVD
    
    # FIX: Chỉ fit vectorizer trên train data
    vectorizer.fit(train_texts)
    
    # Transform dữ liệu
    train_vectors = vectorizer.transform(train_texts)
    val_vectors = vectorizer.transform(val_texts)
    test_vectors = vectorizer.transform(test_texts)
    
    # FIX: Chỉ fit SVD trên train data
    svd = TruncatedSVD(n_components=svd_components, random_state=42)
    svd.fit(train_vectors)
    
    train_svd = svd.transform(train_vectors)
    val_svd = svd.transform(val_vectors)
    test_svd = svd.transform(test_vectors)
    
    # Tạo DataFrame
    feature_names = [f"{prefix}{i}" for i in range(svd_components)]
    df_train_features = pd.DataFrame(train_svd, columns=feature_names, index=train_texts.index)
    df_val_features = pd.DataFrame(val_svd, columns=feature_names, index=val_texts.index)
    df_test_features = pd.DataFrame(test_svd, columns=feature_names, index=test_texts.index)
    
    return df_train_features, df_val_features, df_test_features

def create_embedding_features(texts: pd.Series, embedding_model, vector_size: int, prefix: str) -> pd.DataFrame:
    """
    Tạo features bằng cách lấy trung bình các vector embedding của từ trong văn bản.
    Áp dụng cho Word2Vec (CBOW/Skip-gram) hoặc GloVe.

    Args:
        texts (pd.Series): Dữ liệu văn bản.
        embedding_model: Model đã được train (Word2Vec) hoặc dict chứa word vectors (GloVe).
        vector_size (int): Kích thước của vector embedding.
        prefix (str): Tiền tố cho tên cột features.

    Returns:
        pd.DataFrame: DataFrame chứa các features embedding.
    """
    # Lazy import
    import numpy as np
    from nltk.tokenize import word_tokenize
    
    vectors = []
    is_gensim_model = hasattr(embedding_model, 'wv')
    
    for text in texts:
        tokens = word_tokenize(text)
        doc_vector = np.zeros(vector_size)
        count = 0
        for word in tokens:
            try:
                if is_gensim_model:
                    vec = embedding_model.wv[word]
                else: # Dành cho GloVe (dict)
                    vec = embedding_model[word]
                doc_vector += vec
                count += 1
            except KeyError:
                # Bỏ qua các từ không có trong từ điển
                continue
        
        if count > 0:
            doc_vector /= count
        vectors.append(doc_vector)
        
    feature_names = [f"{prefix}{i}" for i in range(vector_size)]
    return pd.DataFrame(vectors, columns=feature_names, index=texts.index)

# --- Các hàm cụ thể cho từng loại ---

def calculate_tfidf_features(train_texts: pd.Series, val_texts: pd.Series, test_texts: pd.Series, n_components=20):
    """Tạo TF-IDF features."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
    return create_vectorizer_features(train_texts, val_texts, test_texts, vectorizer, n_components, 'tfidf_')

def calculate_bow_features(train_texts: pd.Series, val_texts: pd.Series, test_texts: pd.Series, n_components=20):
    """Tạo Bag-of-Words features."""
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=20000)
    return create_vectorizer_features(train_texts, val_texts, test_texts, vectorizer, n_components, 'bow_')

def calculate_cbow_features(train_texts: pd.Series, val_texts: pd.Series, test_texts: pd.Series, vector_size=50):
    """Tạo CBOW (Word2Vec) features. FIX: Chỉ train trên train data."""
    # Lazy imports
    from gensim.models import Word2Vec
    from nltk.tokenize import word_tokenize
    
    # FIX: Chỉ tokenize train data để train model
    tokenized_corpus = [word_tokenize(text) for text in train_texts]
    
    # Train model Word2Vec (CBOW) chỉ trên train data
    w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=vector_size, window=5, min_count=1, workers=4, cbow_mean=1, sg=0)
    
    df_train_features = create_embedding_features(train_texts, w2v_model, vector_size, 'cbow_')
    df_val_features = create_embedding_features(val_texts, w2v_model, vector_size, 'cbow_')
    df_test_features = create_embedding_features(test_texts, w2v_model, vector_size, 'cbow_')
    
    return df_train_features, df_val_features, df_test_features

def calculate_glove_features(train_texts: pd.Series, val_texts: pd.Series, test_texts: pd.Series, glove_model, vector_size=50):
    """Tạo GloVe features từ model đã tải."""
    df_train_features = create_embedding_features(train_texts, glove_model, vector_size, 'glove_')
    df_val_features = create_embedding_features(val_texts, glove_model, vector_size, 'glove_')
    df_test_features = create_embedding_features(test_texts, glove_model, vector_size, 'glove_')
    return df_train_features, df_val_features, df_test_features

def load_glove_model(glove_file_path):
    """Tải pre-trained GloVe vectors từ file."""
    import numpy as np
    
    print("Loading GloVe Model...")
    model = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array([float(val) for val in parts[1:]])
            model[word] = vector
    print("GloVe Model Loaded.")
    return model

def main():
    # Lazy import for train_test_split
    from sklearn.model_selection import train_test_split
    
    # load data and split train and test
    df_train = read_texts_from_dir(TRAIN_DATA_FOLDER_PATH)
    df_test = read_texts_from_dir(TEST_DATA_FOLDER_PATH)
    df_gt = pd.read_csv(GT_DATA_PATH)
    
    df_train['label'] = df_gt['real_text_id']
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)

    # data augmented (swap file_1 and file_2)
    df_train_aug = df_train.copy()
    df_train_aug['file_1'], df_train_aug['file_2'] = df_train_aug['file_2'], df_train_aug['file_1']
    
    # Update labels after swapping: if label was 1, change to 2; if label was 2, change to 1
    df_train_aug['label'] = df_train_aug['label'].map({1: 2, 2: 1})
    
    df_train = pd.concat([df_train, df_train_aug], axis=0)

    # shuffle the data
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Training shape:", df_train.shape)
    print("Valid shape:", df_val.shape)
    print("Test shape:", df_test.shape)

    # Data Cleaning
    df_train = preprocessing_df(df_train)
    df_val = preprocessing_df(df_val)
    df_test = preprocessing_df(df_test)

    # Feature Engineering
    df_train = feature_engineering(df_train)
    df_val = feature_engineering(df_val)
    df_test = feature_engineering(df_test)

    # FIX: Calculate features properly separated
    print("Calculating TF-IDF features...")
    train_tfidf_features, val_tfidf_features, test_tfidf_features = calculate_tfidf_features(
        df_train['file_1'], df_val['file_1'], df_test['file_1'], n_components=50
    )
    
    print("Calculating BoW features...")
    train_bow_features, val_bow_features, test_bow_features = calculate_bow_features(
        df_train['file_1'], df_val['file_1'], df_test['file_1'], n_components=50
    )
    
    # Skip CBOW features to avoid gensim compatibility issues
    print("Skipping CBOW features...")
    # train_cbow_features, val_cbow_features, test_cbow_features = calculate_cbow_features(
    #     df_train['file_1'], df_val['file_1'], df_test['file_1'], vector_size=100
    # )
    train_cbow_features = val_cbow_features = test_cbow_features = None
    
    # Calculate GloVe features
    try:
        print("Calculating GloVe features...")
        glove_file_path = "./data/glove.6B.100d.txt"
        if os.path.exists(glove_file_path):
            glove_model = load_glove_model(glove_file_path)
            train_glove_features, val_glove_features, test_glove_features = calculate_glove_features(
                df_train['file_1'], df_val['file_1'], df_test['file_1'], glove_model, 100
            )
        else:
            print(f"GloVe file not found at {glove_file_path}. Skipping GloVe features.")
            train_glove_features = val_glove_features = test_glove_features = None
    except Exception as e:
        print(f"Error calculating GloVe features: {e}")
        train_glove_features = val_glove_features = test_glove_features = None
    
    # Save each feature set separately
    print("Saving TF-IDF features...")
    train_tfidf_features.to_csv("./data/train_tfidf_features.csv", index=False)
    val_tfidf_features.to_csv("./data/val_tfidf_features.csv", index=False)
    test_tfidf_features.to_csv("./data/test_tfidf_features.csv", index=False)
    
    print("Saving BoW features...")
    train_bow_features.to_csv("./data/train_bow_features.csv", index=False)
    val_bow_features.to_csv("./data/val_bow_features.csv", index=False)
    test_bow_features.to_csv("./data/test_bow_features.csv", index=False)
    
    # Skip saving CBOW features since they're disabled
    print("Skipping CBOW features save...")
    # print("Saving CBOW features...")
    # train_cbow_features.to_csv("./data/train_cbow_features.csv", index=False)
    # val_cbow_features.to_csv("./data/val_cbow_features.csv", index=False)
    # test_cbow_features.to_csv("./data/test_cbow_features.csv", index=False)
    
    
    if train_glove_features is not None:
        print("Saving GloVe features...")
        if train_glove_features is not None:
            train_glove_features.to_csv("./data/train_glove_features.csv", index=False)
            val_glove_features.to_csv("./data/val_glove_features.csv", index=False)
            test_glove_features.to_csv("./data/test_glove_features.csv", index=False)
            

    # df_train.drop(columns=['file_1', 'file_2'], inplace=True, errors='ignore')
    # df_val.drop(columns=['file_1', 'file_2'], inplace=True, errors='ignore')
    # df_test.drop(columns=['file_1', 'file_2'], inplace=True, errors='ignore')
    
    print(df_train.head())
    print("features:", df_train.columns)

    df_train.to_csv("./data/train_statistic_features.csv", index=False)
    df_val.to_csv("./data/val_statistic_features.csv", index=False)
    df_test.to_csv("./data/test_statistic_features.csv", index=False)

if __name__ == "__main__":
    main()