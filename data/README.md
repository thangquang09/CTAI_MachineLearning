# Dataset Information

Dữ liệu của dự án này được lưu trữ trên Google Drive và **không** được đính kèm trực tiếp trong repository này do dung lượng lớn.

## Link tải dữ liệu
Bạn có thể truy cập và tải dữ liệu tại đây:  
[Google Drive Dataset Link](https://drive.google.com/drive/folders/1D533zdt4em35IA3Cfjfpyv1JuywNYIjO?usp=sharing)

## Hướng dẫn sử dụng
1. Mở liên kết Google Drive ở trên.
2. Tải toàn bộ thư mục dữ liệu về máy.
3. Giải nén (nếu có).
4. Đặt thư mục dữ liệu vào đúng đường dẫn mà mã nguồn yêu cầu (ví dụ: `data/`).

---

**Lưu ý:**  
- Bạn cần quyền truy cập vào Google Drive để tải dữ liệu.  
- Nếu gặp sự cố khi tải, hãy kiểm tra xem liên kết có bị giới hạn quyền truy cập hay không.

# Processed Features Information

Sau khi thực hiện Feature Engineering, dữ liệu được biến đổi thành các features sau:
- **Training set**: 152 samples × 52 features
- **Validation set**: 19 samples × 52 features

## Statistical Features (Rule-based Features)

### Basic Text Metrics

| Feature | Mô tả |
|---------|-------|
| `file1_char_count` / `file2_char_count` | Số lượng ký tự trong văn bản |
| `file1_word_count` / `file2_word_count` | Số lượng từ trong văn bản |
| `file1_sentence_count` / `file2_sentence_count` | Số lượng câu trong văn bản |
| `file1_avg_sentence_length` / `file2_avg_sentence_length` | Độ dài trung bình của câu (số từ/câu) |

### Language Detection Features

| Feature | Mô tả |
|---------|-------|
| `file1_english_word_ratio` / `file2_english_word_ratio` | Tỷ lệ từ tiếng Anh trong văn bản (0-1) |
| `file1_cyrillic_count` / `file2_cyrillic_count` | Số lượng ký tự Cyrillic (tiếng Nga) |
| `file1_arabic_count` / `file2_arabic_count` | Số lượng ký tự Arabic (tiếng Ả Rập) |
| `file1_chinese_count` / `file2_chinese_count` | Số lượng ký tự Chinese (tiếng Trung) |
| `file1_script_diversity` / `file2_script_diversity` | Mức độ đa dạng ngôn ngữ (số loại script khác nhau) |

### Text Quality & Consistency Features

| Feature | Mô tả |
|---------|-------|
| `file1_unicode_control_chars` / `file2_unicode_control_chars` | Số ký tự điều khiển Unicode (phát hiện encoding lỗi) |
| `file1_number_count` / `file2_number_count` | Số lượng chuỗi số trong văn bản |
| `file1_uppercase_word_count` / `file2_uppercase_word_count` | Số từ viết hoa toàn bộ |
| `file1_repetition_score` / `file2_repetition_score` | Điểm số lặp lại từ (phát hiện spam/fake content) |

### Statistical Complexity Metrics

| Feature | Mô tả |
|---------|-------|
| `file1_perplexity_score` / `file2_perplexity_score` | Điểm perplexity (độ phức tạp ngôn ngữ) |
| `file1_ttr_ratio` / `file2_ttr_ratio` | Type-Token Ratio (tỷ lệ từ vựng độc đáo) |

### Comparison Features (Differences & Ratios)

| Feature | Mô tả |
|---------|-------|
| `diff_*` | Sự khác biệt giữa file1 và file2 (file1 - file2) cho mỗi metric |
| `ratio_char_count` | Tỷ lệ số ký tự file1/file2 |
| `ratio_word_count` | Tỷ lệ số từ file1/file2 |
| `ratio_sentence_count` | Tỷ lệ số câu file1/file2 |
| `ratio_perplexity_score` | Tỷ lệ perplexity score file1/file2 |
| `ratio_ttr_ratio` | Tỷ lệ TTR file1/file2 |

### Similarity Features

| Feature | Mô tả |
|---------|-------|
| `cosine_sim_word_counts` | Độ tương đồng cosine dựa trên số lượng từ chung |

## Vectorized Features

### TF-IDF Features
- **File**: `train_tfidf_features.csv`, `val_tfidf_features.csv`, `test_tfidf_features.csv`
- **Mô tả**: Term Frequency-Inverse Document Frequency features
- **Số features**: 50 components (sau khi giảm chiều bằng SVD)
- **Cách tính**: 
  - Sử dụng n-gram (1,2) với tối đa 20,000 features
  - Áp dụng TruncatedSVD để giảm xuống 50 dimensions
  - TF-IDF đo tầm quan trọng của từ trong document so với toàn bộ corpus

### Bag of Words (BoW) Features  
- **File**: `train_bow_features.csv`, `val_bow_features.csv`, `test_bow_features.csv`
- **Mô tả**: Count-based word frequency features
- **Số features**: 50 components (sau khi giảm chiều bằng SVD)
- **Cách tính**:
  - Đếm tần suất xuất hiện của từ/n-gram
  - Sử dụng n-gram (1,2) với tối đa 20,000 features
  - Áp dụng TruncatedSVD để giảm xuống 50 dimensions

### Features được lưu trữ

| File | Mô tả | Kích thước |
|------|-------|------------|
| `train_statistic_features.csv` | Rule-based features cho training set | 152 × 52 |
| `val_statistic_features.csv` | Rule-based features cho validation set | 19 × 52 |
| `test_statistic_features.csv` | Rule-based features cho test set | N × 52 |
| `train_tfidf_features.csv` | TF-IDF features cho training set | 152 × 50 |
| `val_tfidf_features.csv` | TF-IDF features cho validation set | 19 × 50 |
| `test_tfidf_features.csv` | TF-IDF features cho test set | N × 50 |
| `train_bow_features.csv` | BoW features cho training set | 152 × 50 |
| `val_bow_features.csv` | BoW features cho validation set | 19 × 50 |
| `test_bow_features.csv` | BoW features cho test set | N × 50 |

**Lưu ý**: Tất cả vectorizer và SVD chỉ được fit trên training data để tránh data leakage.

