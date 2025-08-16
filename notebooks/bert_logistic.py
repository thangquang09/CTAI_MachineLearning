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
import math

# NLP
import nltk
import textstat
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

# Deep Learning
import torch
from transformers import AutoTokenizer, AutoModel

# ML & utils
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

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
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in word_tokenize(text)
        if tok.isalpha() and tok not in stop_words and len(tok) > 2
    ]
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

    # Existing features...
    char_count = len(text)
    word_count = len(word_tokenize(text))
    sentence_count = len(sent_tokenize(text))
    avg_sentence_length = word_count / max(sentence_count, 1)

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
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
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
                for word, count in Counter(word_tokenize(text.lower())).items()
                if count > 3
            ]
        )
        / max(word_count, 1),
        "china_count": text.lower().count("china"),
        "dinosaur_count": text.lower().count("dinosaur"),
    }


def compute_advanced_features(text: str) -> dict:
    """Tính toán các features nâng cao cho fake detection theo biểu đồ importance."""
    if not isinstance(text, str) or not text.strip():
        return {f: 0 for f in [
            'unique_word_count_ratio', 'latin_ratio', 'digit_count', 
            'ttr_ratio', 'flesch_reading_ease', 'dale_chall_readability',
            'coleman_liau_index', 'short_word_count_ratio', 'uppercase_ratio',
            'english_ratio', 'perplexity_score', 'sentence_count',
            'word_count', 'avg_word_length'
        ]}
    
    # Basic text stats
    words = text.split()
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
    
    # 4. Type-Token Ratio (TTR)
    ttr_ratio = unique_word_count_ratio  # Same as unique_word_count_ratio
    
    # 5. Readability scores using textstat
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    dale_chall_readability = textstat.dale_chall_readability_score(text)
    coleman_liau_index = textstat.coleman_liau_index(text)
    
    # 6. Short word ratio
    short_words = [w for w in words if len(w) <= 3]
    short_word_count_ratio = len(short_words) / max(word_count, 1)
    
    # 7. Uppercase ratio
    uppercase_chars = len(re.findall(r'[A-Z]', text))
    uppercase_ratio = uppercase_chars / max(char_count, 1)
    
    # 8. English ratio (approximate using common English patterns)
    english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
    english_ratio = english_words / max(word_count, 1)
    
    # 9. Simple perplexity approximation (entropy-based)
    word_freq = Counter(words)
    total_words = sum(word_freq.values())
    perplexity_score = 0
    if total_words > 0:
        entropy = -sum((freq/total_words) * math.log2(freq/total_words) 
                      for freq in word_freq.values())
        perplexity_score = 2 ** entropy
    
    # 10. Sentence count
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # 11. Average word length
    avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
    
    return {
        'unique_word_count_ratio': unique_word_count_ratio,
        'latin_ratio': latin_ratio,
        'digit_count': digit_count,
        'ttr_ratio': ttr_ratio,
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


def extract_top_features(df: pd.DataFrame) -> np.ndarray:
    """Extract top features theo biểu đồ importance."""
    features = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting top features"):
        text1, text2 = row['file_1'], row['file_2']
        
        # Get features for both texts
        f1 = compute_advanced_features(text1)
        f2 = compute_advanced_features(text2)
        
        # Create difference and ratio features (theo pattern từ biểu đồ)
        feature_row = []
        
        # 1. unique_word_count_ratio (tỷ lệ giữa file1 và file2)
        unique_ratio = (f1['unique_word_count_ratio'] + 1e-8) / (f2['unique_word_count_ratio'] + 1e-8)
        # Clip ratio to reasonable range
        unique_ratio = np.clip(unique_ratio, 0.1, 10.0)
        feature_row.append(unique_ratio)
        
        # 2. latin_ratio_diff
        latin_ratio_diff = abs(f1['latin_ratio'] - f2['latin_ratio'])
        feature_row.append(latin_ratio_diff)
        
        # 3. digit_count_diff
        digit_count_diff = abs(f1['digit_count'] - f2['digit_count'])
        feature_row.append(digit_count_diff)
        
        # 4. semantic_similarity (using simple overlap for now)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        semantic_similarity = len(words1 & words2) / max(len(words1 | words2), 1)
        feature_row.append(semantic_similarity)
        
        # 5. ttr_ratio
        ttr_ratio = (f1['ttr_ratio'] + 1e-8) / (f2['ttr_ratio'] + 1e-8)
        ttr_ratio = np.clip(ttr_ratio, 0.1, 10.0)
        feature_row.append(ttr_ratio)
        
        # 6. perplexity_diff
        perplexity_diff = abs(f1['perplexity_score'] - f2['perplexity_score'])
        feature_row.append(perplexity_diff)
        
        # 7. flesch_reading_ease_ratio
        flesch_ratio = (f1['flesch_reading_ease'] + 100) / (f2['flesch_reading_ease'] + 100)
        flesch_ratio = np.clip(flesch_ratio, 0.1, 10.0)
        feature_row.append(flesch_ratio)
        
        # 8. short_word_count_ratio
        short_ratio = (f1['short_word_count_ratio'] + 1e-8) / (f2['short_word_count_ratio'] + 1e-8)
        short_ratio = np.clip(short_ratio, 0.1, 10.0)
        feature_row.append(short_ratio)
        
        # 9. readability_avg_ratio
        readability_avg_1 = (f1['flesch_reading_ease'] + f1['dale_chall_readability']) / 2
        readability_avg_2 = (f2['flesch_reading_ease'] + f2['dale_chall_readability']) / 2
        readability_avg_ratio = (readability_avg_1 + 50) / (readability_avg_2 + 50)
        readability_avg_ratio = np.clip(readability_avg_ratio, 0.1, 10.0)
        feature_row.append(readability_avg_ratio)
        
        # 10. dale_chall_readability_score_diff
        dale_chall_diff = abs(f1['dale_chall_readability'] - f2['dale_chall_readability'])
        feature_row.append(dale_chall_diff)
        
        # 11. sentence_count_diff
        sentence_count_diff = abs(f1['sentence_count'] - f2['sentence_count'])
        feature_row.append(sentence_count_diff)
        
        # 12. perplexity_ratio
        perplexity_ratio = (f1['perplexity_score'] + 1e-8) / (f2['perplexity_score'] + 1e-8)
        perplexity_ratio = np.clip(perplexity_ratio, 0.1, 10.0)
        feature_row.append(perplexity_ratio)
        
        # 13. coleman_liau_index_diff
        coleman_diff = abs(f1['coleman_liau_index'] - f2['coleman_liau_index'])
        feature_row.append(coleman_diff)
        
        # 14. english_ratio_diff
        english_ratio_diff = abs(f1['english_ratio'] - f2['english_ratio'])
        feature_row.append(english_ratio_diff)
        
        # 15. word_count_diff
        word_count_diff = abs(f1['word_count'] - f2['word_count'])
        feature_row.append(word_count_diff)
        
        # 16. uppercase_ratio_diff
        uppercase_ratio_diff = abs(f1['uppercase_ratio'] - f2['uppercase_ratio'])
        feature_row.append(uppercase_ratio_diff)
        
        # 17. latin_ratio_ratio
        latin_ratio_ratio = (f1['latin_ratio'] + 1e-8) / (f2['latin_ratio'] + 1e-8)
        latin_ratio_ratio = np.clip(latin_ratio_ratio, 0.1, 10.0)
        feature_row.append(latin_ratio_ratio)
        
        # Add individual features as well
        feature_row.extend([
            f1['unique_word_count_ratio'], f2['unique_word_count_ratio'],
            f1['latin_ratio'], f2['latin_ratio'],
            f1['flesch_reading_ease'], f2['flesch_reading_ease'],
            f1['perplexity_score'], f2['perplexity_score']
        ])
        
        features.append(feature_row)
    
    return np.array(features).astype(np.float32)


def extract_pairwise_features(df: pd.DataFrame) -> np.ndarray:
    """Extract features comparing file_1 vs file_2 directly."""
    features = []

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Extracting pairwise features"
    ):
        text1, text2 = row["file_1"], row["file_2"]

        # Language consistency
        def detect_language_consistency(text):
            latin_ratio = len(re.findall(r"[a-zA-Z]", text)) / max(len(text), 1)
            return latin_ratio

        lang_consistency_1 = detect_language_consistency(text1)
        lang_consistency_2 = detect_language_consistency(text2)
        lang_consistency_diff = abs(lang_consistency_1 - lang_consistency_2)

        # Tone similarity
        def get_tone_score(text):
            formal_words = ["analysis", "observation", "measurement", "data", "survey"]
            informal_words = ["amazing", "incredible", "forget", "wow"]
            formal_count = sum(text.lower().count(w) for w in formal_words)
            informal_count = sum(text.lower().count(w) for w in informal_words)
            return formal_count - informal_count

        tone_1 = get_tone_score(text1)
        tone_2 = get_tone_score(text2)
        tone_diff = abs(tone_1 - tone_2)

        # Content overlap (simple)
        words_1 = set(word_tokenize(text1.lower()))
        words_2 = set(word_tokenize(text2.lower()))
        jaccard_similarity = len(words_1 & words_2) / max(len(words_1 | words_2), 1)

        features.append(
            [
                lang_consistency_1,
                lang_consistency_2,
                lang_consistency_diff,
                tone_1,
                tone_2,
                tone_diff,
                jaccard_similarity,
            ]
        )

    return np.array(features).astype(np.float32)


def extract_rule_based_features(df: pd.DataFrame) -> np.ndarray:
    """Tạo ma trận đặc trưng rule-based, bao gồm diff giữa file_1 và file_2."""
    features = []
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Extracting rule-based features"
    ):
        f1 = compute_rule_based_features(row["file_1"])
        f2 = compute_rule_based_features(row["file_2"])

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

    features = pd.DataFrame(
        {
            "len_diff": (len_1 - len_2).abs(),
            "word_diff": (words_1 - words_2).abs(),
            "len_ratio": (len_1 + 1) / (len_2 + 1),
            "words_ratio": (words_1 + 1) / (words_2 + 1),
            "avg_word_len_1": df["file_1"].apply(
                lambda x: np.mean([len(w) for w in x.split()]) if x else 0
            ),
            "avg_word_len_2": df["file_2"].apply(
                lambda x: np.mean([len(w) for w in x.split()]) if x else 0
            ),
        }
    )
    return features.values.astype(np.float32)


# ----------------------- UNIVERSAL EMBEDDING FEATURES ------------------------------------
class UniversalEmbeddingExtractor:
    def __init__(
        self, model_name="bert-base-uncased", max_length=512, device=None, batch_size=32
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        print(f"Loading embedding model: {model_name}")

        # Detect model type and load accordingly
        self._load_model()

    def _load_model(self):
        """Load model based on model name/type."""
        model_name_lower = self.model_name.lower()

        # Check if it's a sentence-transformers model
        if any(
            keyword in model_name_lower
            for keyword in [
                "sentence-transformers",
                "all-minilm",
                "all-mpnet",
                "bge-",
                "e5-",
            ]
        ):
            self._load_sentence_transformer()
        # Check if it's a Vietnamese model
        elif any(
            keyword in model_name_lower
            for keyword in ["vinai", "vietnamese", "phobert"]
        ):
            self._load_transformers_model()
        # Check if it's OpenAI model
        elif "openai" in model_name_lower or "text-embedding" in model_name_lower:
            self._load_openai_model()
        # Default to transformers for BERT, RoBERTa, etc.
        else:
            self._load_transformers_model()

    def _load_sentence_transformer(self):
        """Load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer

            self.model_type = "sentence_transformer"
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.tokenizer = None  # Not needed for sentence-transformers
            print(f"Loaded as SentenceTransformer model")
        except ImportError:
            print("sentence-transformers not installed, falling back to transformers")
            self._load_transformers_model()

    def _load_transformers_model(self):
        """Load standard transformers model (BERT, RoBERTa, etc.)."""
        self.model_type = "transformers"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded as Transformers model")

    def _load_openai_model(self):
        """Load OpenAI embedding model."""
        try:
            import openai

            self.model_type = "openai"
            self.model = None  # Will use API
            self.tokenizer = None
            print(f"Loaded as OpenAI model")
        except ImportError:
            print("openai package not installed, falling back to transformers")
            self._load_transformers_model()

    def get_embeddings(self, texts):
        """Extract embeddings for a list of texts using the appropriate method."""
        if self.model_type == "sentence_transformer":
            return self._get_sentence_transformer_embeddings(texts)
        elif self.model_type == "transformers":
            return self._get_transformers_embeddings(texts)
        elif self.model_type == "openai":
            return self._get_openai_embeddings(texts)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _get_sentence_transformer_embeddings(self, texts):
        """Extract embeddings using sentence-transformers."""
        embeddings = []

        # Process in batches
        for i in tqdm(
            range(0, len(texts), self.batch_size), desc="Extracting embeddings"
        ):
            batch_texts = texts[i : i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def _get_transformers_embeddings(self, texts):
        """Extract embeddings using transformers (BERT-style)."""
        embeddings = []

        with torch.no_grad():
            for i in tqdm(
                range(0, len(texts), self.batch_size), desc="Extracting embeddings"
            ):
                batch_texts = texts[i : i + self.batch_size]
                batch_embeddings = []

                for text in batch_texts:
                    # Tokenize
                    inputs = self.tokenizer(
                        text,
                        max_length=self.max_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    )

                    # Move to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Get embeddings
                    outputs = self.model(**inputs)

                    # Use [CLS] token embedding or mean pooling
                    if hasattr(outputs, "last_hidden_state"):
                        # For BERT-style models, use [CLS] token
                        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    elif hasattr(outputs, "pooler_output"):
                        # Some models have pooler output
                        cls_embedding = outputs.pooler_output.cpu().numpy()
                    else:
                        # Fallback: mean pooling
                        cls_embedding = (
                            outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                        )

                    batch_embeddings.append(cls_embedding.flatten())

                embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def _get_openai_embeddings(self, texts):
        """Extract embeddings using OpenAI API."""
        import openai

        embeddings = []

        for i in tqdm(
            range(0, len(texts), self.batch_size), desc="Extracting OpenAI embeddings"
        ):
            batch_texts = texts[i : i + self.batch_size]

            try:
                response = openai.Embedding.create(
                    model=self.model_name, input=batch_texts
                )
                batch_embeddings = [item["embedding"] for item in response["data"]]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error with OpenAI API: {e}")
                # Fallback to zero embeddings
                embeddings.extend(
                    [np.zeros(1536) for _ in batch_texts]
                )  # OpenAI default dim

        return np.array(embeddings)


def extract_embedding_features(
    df: pd.DataFrame, embedding_extractor: UniversalEmbeddingExtractor
) -> np.ndarray:
    """Extract embedding features for both file_1 and file_2, including diff features."""
    print("Extracting embedding features for file_1...")
    emb_f1 = embedding_extractor.get_embeddings(df["file_1"].tolist())

    print("Extracting embedding features for file_2...")
    emb_f2 = embedding_extractor.get_embeddings(df["file_2"].tolist())

    # Create difference and similarity features
    emb_diff = emb_f1 - emb_f2
    emb_abs_diff = np.abs(emb_diff)

    # Cosine similarity
    cosine_sim = np.sum(emb_f1 * emb_f2, axis=1, keepdims=True) / (
        np.linalg.norm(emb_f1, axis=1, keepdims=True)
        * np.linalg.norm(emb_f2, axis=1, keepdims=True)
        + 1e-8
    )

    # Euclidean distance
    euclidean_dist = np.linalg.norm(emb_diff, axis=1, keepdims=True)

    # Concatenate all embedding features
    embedding_features = np.concatenate(
        [emb_f1, emb_f2, emb_diff, emb_abs_diff, cosine_sim, euclidean_dist], axis=1
    )

    return embedding_features.astype(np.float32)


# ----------------------- PREPARE DATA FOR MODEL ---------------------------
def prepare_data_for_model(
    df: pd.DataFrame,
    embedding_extractor: UniversalEmbeddingExtractor = None,
    fit_embedding: bool = False,
    model_name: str = "bert-base-uncased",
):
    """
    Chuẩn bị dữ liệu cho model với focus vào top features theo importance chart.

    Args:
        df: DataFrame chứa dữ liệu thô
        embedding_extractor: Universal embedding extractor, nếu None sẽ tạo mới
        fit_embedding: Có tạo embedding extractor mới hay không
        model_name: Tên model embedding để sử dụng

    Returns:
        feature_matrix: Ma trận features đã kết hợp
        embedding_extractor: Embedding extractor (để dùng cho test set)
    """
    # 1. Extract top features (most important)
    print("Step 1: Extracting top importance features...")
    top_features = extract_top_features(df)
    
    # 2. Extract rule-based features (existing)
    print("Step 2: Extracting rule-based features...")
    rule_features = extract_rule_based_features(df)
    
    # 3. Extract statistical features (existing)
    print("Step 3: Extracting statistical features...")
    stat_features = extract_statistical_features(df)
    
    # 4. Extract embedding features (lighter approach)
    print("Step 4: Extracting embedding features...")
    if embedding_extractor is None or fit_embedding:
        embedding_extractor = UniversalEmbeddingExtractor(
            model_name=model_name,
            max_length=256,  # Shorter for speed
            batch_size=64
        )
    
    embedding_features = extract_embedding_features(df, embedding_extractor)
    
    # 5. Extract pairwise features (existing)
    print("Step 5: Extracting pairwise features...")
    pairwise_features = extract_pairwise_features(df)
    
    # 6. Combine all features with priority on top features
    print("Step 6: Combining features...")
    top_sparse = csr_matrix(top_features)
    rule_sparse = csr_matrix(rule_features)
    stat_sparse = csr_matrix(stat_features)
    embedding_sparse = csr_matrix(embedding_features)
    pairwise_sparse = csr_matrix(pairwise_features)
    
    # Priority: top features first, then others
    feature_matrix = hstack([top_sparse, rule_sparse, stat_sparse, embedding_sparse, pairwise_sparse]).tocsr()
    
    print(f"Final feature matrix shape: {feature_matrix.shape}")
    print(f"Top features: {top_features.shape[1]}, Rule: {rule_features.shape[1]}, Stat: {stat_features.shape[1]}, Embedding: {embedding_features.shape[1]}, Pairwise: {pairwise_features.shape[1]}")
    
    return feature_matrix, embedding_extractor


# ----------------------- MODEL TRAINING -----------------------------------
def train_and_evaluate(X, y, clf_model, param_grid):
    """Cross-validate + grid search, trả về best model."""
    clf = Pipeline(
        [
            ("select", SelectKBest(f_classif, k=min(20_000, X.shape[1]))),
            ("clf", clf_model),
        ]
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring="accuracy",
        cv=skf,
        n_jobs=-1,
        verbose=2,
    )
    grid.fit(X, y)
    print(f"Best CV accuracy: {grid.best_score_:.4f}")
    print(f"Best params: {grid.best_params_}")
    return grid.best_estimator_


def save_model_and_artifacts(
    model, embedding_extractor, feature_names=None, save_dir="models", tune_dir="tune"
):
    """Save trained model and related artifacts."""
    save_dir = Path(save_dir)
    tune_dir = Path(tune_dir)
    save_dir.mkdir(exist_ok=True)
    tune_dir.mkdir(exist_ok=True)

    # Save model
    model_path = save_dir / "logistic_regression_universal.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Save embedding extractor
    embedding_path = save_dir / "embedding_extractor.pkl"
    joblib.dump(embedding_extractor, embedding_path)
    print(f"Embedding extractor saved to: {embedding_path}")

    # Save feature names if provided
    if feature_names:
        features_path = save_dir / "feature_names.pkl"
        joblib.dump(feature_names, features_path)
        print(f"Feature names saved to: {features_path}")

    # Save hyperparameter tuning results to tune directory
    if hasattr(model, "best_params_"):
        tune_results = {
            "best_params": model.best_params_,
            "best_score": model.best_score_,
            "cv_results": model.cv_results_,
        }
        tune_path = tune_dir / "hyperparameter_results.pkl"
        joblib.dump(tune_results, tune_path)
        print(f"Tuning results saved to: {tune_path}")


def main(embedding_model_name: str = "bert-base-uncased"):
    """Main training pipeline with configurable embedding model."""
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
    X_train, embedding_extractor = prepare_data_for_model(
        df_train, fit_embedding=True, model_name=embedding_model_name
    )

    print("Preparing test data...")
    X_test, _ = prepare_data_for_model(
        df_test, embedding_extractor=embedding_extractor, fit_embedding=False
    )

    # Define model and parameters
    clf_model = LogisticRegression(
        max_iter=5_000, solver="liblinear", n_jobs=-1, random_state=SEED
    )
    param_grid = {
        "select__k": [10_000, 15_000, 20_000],
        "clf__C": [0.1, 0.5, 1, 2, 5],
        "clf__penalty": ["l1", "l2"],
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
    save_model_and_artifacts(
        model, embedding_extractor, save_dir="models", tune_dir="tune"
    )

    # Create submission
    submission = pd.DataFrame(
        {"id": df_test.index, "real_text_id": test_pred.astype(int)}
    ).sort_values("id")

    submission_path = Path(
        f"submission_universal_{embedding_model_name.replace('/', '_')}.csv"
    )
    submission.to_csv(submission_path, index=False)
    print(f"✅ Submission saved to {submission_path.resolve()}")

    return model, embedding_extractor


if __name__ == "__main__":
    pass
