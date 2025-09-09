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
