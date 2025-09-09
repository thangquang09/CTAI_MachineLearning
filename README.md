# CTAI_MachineLearning

Repository của nhóm 7 Chương trình Đào tạo Kỹ sư AI

## Hướng dẫn cài đặt

Đầu tiên cài đặt `uv`

```
# MacOS or Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Đồng bộ môi trường của project

```
uv sync --quiet
```

## Các đường dẫn quan trọng

- [Notebook quá trình EDA của bài toán](notebooks/EDA.ipynb)
- [Tiếp cận bằng Thống kê + Machine Learning](notebooks/ML_Approach.ipynb)
- [Cấu hình của Pretrained Model](src/pretrained_model.py)
- [Code chính train Machine Learning text classification](src/ml_textclassification.py)
- [Code chính để fine-tune pretrained model textpair](src/train_pretrained_model.py)
- [Code chính để fine-tune pretrained model text classification](src/train_pretrained_model_textclassification.py)


## Bảng kết quả

Nhấp vào tên model sẽ dẫn tới file csv kết quả, để kiểm tra vui lòng lên [trang web cuộc thi](https://www.kaggle.com/competitions/fake-or-real-the-impostor-hunt/submissions#) để submit

**TextPair**

| Approach          | Model                  | Accuracy (Public Test) |
|-------------------|------------------------|----------|
| Statistic         | [statistics_svm](new_submission/submission_svm.csv)         | 75.933   |
|                   | [statistics_catboost](new_submission/submission_catboost.csv)    | 73.858   |
| Embedding + ML    | [e5_logistic](new_submission/logistic_model_test_submission.csv)            | 74.900   |
|                   | [e5_catboost](new_submission/submission_e5_catboost.csv)            | 85.062   |
| Pretrained Model  | [deberta_v3_large](new_submission/submission_SiamesePretrainedModel_microsoft_deberta_v3_large.csv)       | 93.775   | 
|                   | [bert_large](new_submission/submission_SiamesePretrainedModel_bert_large.csv)             | 89.419   | 


**Text Classification**

| Model                               | Accuracy (Public Test) |
|-------------------------------------|----------|
| [logistic_textclassification](new_submission/submission_ml_text_classification_LogisticRegression.csv)         | 87.344   | 
| [randomforest_textclassification](new_submission/submission_ml_text_classification_RandomForestClassifier.csv)     | 85.269   | 
| [deberta_v3_large_classification](new_submission/submission_TextClassificationPretrainedModel_microsoft_deberta_v3_large.csv)     | 93.586   | 
| [google_electra](new_submission/submission_TextClassificationPretrainedModel_google_electra_base_discriminator.csv)         | 90.456   | 
