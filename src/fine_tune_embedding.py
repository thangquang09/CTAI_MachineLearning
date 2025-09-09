import argparse
import os
import random
import sys
from itertools import combinations

import numpy as np
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    losses,
)
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

from ml_textclassification import build_text_classification_data, preprocessing


class CustomEvaluator:
    """Custom evaluator để đánh giá embedding model trên task phân loại văn bản"""

    def __init__(self, sentences1, sentences2, labels, name="custom"):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels  # 1.0 cho similar, 0.0 cho dissimilar
        self.name = name

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        """Evaluate model và return accuracy score"""
        # Encode sentences
        embeddings1 = model.encode(self.sentences1)
        embeddings2 = model.encode(self.sentences2)

        # Compute cosine similarity
        cosine_scores = []
        for emb1, emb2 in zip(embeddings1, embeddings2):
            cosine_sim = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )
            cosine_scores.append(cosine_sim)

        # Convert similarity scores to predictions (threshold = 0.5)
        predictions = [1.0 if score > 0.5 else 0.0 for score in cosine_scores]

        # Calculate accuracy
        accuracy = sum(
            1 for pred, true in zip(predictions, self.labels) if pred == true
        ) / len(self.labels)

        print(f"{self.name} accuracy: {accuracy:.4f}")
        return accuracy


def create_training_examples(
    textclf_df, max_positive_pairs=None, use_all_combinations=False
):
    """
    Tạo training examples từ DataFrame với nhiều tùy chọn tối ưu hóa

    Args:
        textclf_df: DataFrame chứa text và label
        max_positive_pairs: Giới hạn số positive pairs (None = không giới hạn)
        use_all_combinations: Có dùng tất cả combinations cho positive pairs không
    """
    train_examples = []
    real_texts = textclf_df[textclf_df["label"] == 1]["text"].tolist()
    fake_texts = textclf_df[textclf_df["label"] == 0]["text"].tolist()

    # Validate input data
    if len(real_texts) < 2:
        raise ValueError(
            f"Cần ít nhất 2 real texts để tạo positive pairs, nhưng chỉ có {len(real_texts)}"
        )
    if len(fake_texts) < 1:
        raise ValueError(
            f"Cần ít nhất 1 fake text để tạo negative pairs, nhưng chỉ có {len(fake_texts)}"
        )

    random.seed(42)
    random.shuffle(real_texts)
    random.shuffle(fake_texts)

    # Tạo positive pairs từ real texts
    if use_all_combinations:
        # Sử dụng tất cả combinations (tốt hơn nhưng nhiều data hơn)
        positive_pairs = list(combinations(real_texts, 2))
        if max_positive_pairs and len(positive_pairs) > max_positive_pairs:
            positive_pairs = random.sample(positive_pairs, max_positive_pairs)

        for text1, text2 in positive_pairs:
            train_examples.append(InputExample(texts=[text1, text2], label=1.0))
    else:
        # Ghép đôi tuần tự (như cũ)
        num_positive_pairs = len(real_texts) // 2
        if max_positive_pairs:
            num_positive_pairs = min(num_positive_pairs, max_positive_pairs)

        for i in range(0, num_positive_pairs * 2, 2):
            train_examples.append(
                InputExample(texts=[real_texts[i], real_texts[i + 1]], label=1.0)
            )

    # Tạo negative pairs (real-fake)
    min_len = min(len(real_texts), len(fake_texts))
    num_negative_pairs = len(
        [ex for ex in train_examples if ex.label == 1.0]
    )  # Balance với positive pairs
    num_negative_pairs = min(num_negative_pairs, min_len)

    for i in range(num_negative_pairs):
        train_examples.append(
            InputExample(
                texts=[
                    real_texts[i % len(real_texts)],
                    fake_texts[i % len(fake_texts)],
                ],
                label=0.0,
            )
        )

    print(f"Created {len(train_examples)} training examples:")
    positive_count = sum(1 for ex in train_examples if ex.label == 1.0)
    print(f"  - Positive pairs: {positive_count}")
    print(f"  - Negative pairs: {len(train_examples) - positive_count}")

    return train_examples


def fine_tune_embedding(
    base_model_name,
    train_textclf_df,
    valid_textclf_df=None,
    epochs=3,
    batch_size=16,
    output_path="fine_tune_embedding_model",
    patience=3,
    min_delta=0.001,
    warmup_steps=None,
    use_all_combinations=False,
    max_positive_pairs=1000,
):
    """
    Fine-tune embedding model với validation set và early stopping được cải thiện

    Args:
        base_model_name: Tên model base để fine-tune
        train_textclf_df: DataFrame training data
        valid_textclf_df: DataFrame validation data (optional)
        epochs: Số epochs tối đa
        batch_size: Batch size
        output_path: Đường dẫn lưu model
        patience: Số epochs chờ trước khi early stop
        min_delta: Threshold cải thiện tối thiểu
        warmup_steps: Số warmup steps (None = auto calculate)
        use_all_combinations: Dùng tất cả combinations cho positive pairs
        max_positive_pairs: Giới hạn số positive pairs
    """
    print("=== PREPARING DATA ===")

    # Tạo training examples
    train_examples = create_training_examples(
        train_textclf_df,
        max_positive_pairs=max_positive_pairs,
        use_all_combinations=use_all_combinations,
    )

    # Tạo evaluators
    train_evaluator = None
    valid_evaluator = None

    if valid_textclf_df is not None:
        print("\n=== CREATING VALIDATION SET ===")
        valid_examples = create_training_examples(
            valid_textclf_df,
            max_positive_pairs=max_positive_pairs // 2,  # Ít hơn để nhanh hơn
            use_all_combinations=use_all_combinations,
        )

        # Tạo validation evaluator
        valid_sentences1 = [example.texts[0] for example in valid_examples]
        valid_sentences2 = [example.texts[1] for example in valid_examples]
        valid_labels = [example.label for example in valid_examples]

        valid_evaluator = CustomEvaluator(
            sentences1=valid_sentences1,
            sentences2=valid_sentences2,
            labels=valid_labels,
            name="validation",
        )

    # Tạo train evaluator để monitor training
    train_sentences1 = [
        example.texts[0] for example in train_examples[:200]
    ]  # Sample để nhanh
    train_sentences2 = [example.texts[1] for example in train_examples[:200]]
    train_labels = [example.label for example in train_examples[:200]]

    train_evaluator = CustomEvaluator(
        sentences1=train_sentences1,
        sentences2=train_sentences2,
        labels=train_labels,
        name="training",
    )

    print("\n=== INITIALIZING MODEL ===")
    # Load Model
    model = SentenceTransformer(base_model_name)

    # Shuffle training examples trước khi tạo dataset
    random.shuffle(train_examples)

    train_dataset = SentenceLabelDataset(train_examples)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Calculate warmup steps
    if warmup_steps is None:
        warmup_steps = int(len(train_dataloader) * 0.1)  # 10% của total steps

    print(f"Total training steps per epoch: {len(train_dataloader)}")
    print(f"Warmup steps: {warmup_steps}")

    # Training với early stopping được cải thiện
    if valid_evaluator is not None:
        print("\n=== TRAINING WITH VALIDATION ===")
        best_score = -1
        patience_counter = 0
        best_model_path = f"{output_path}_best"

        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

            # Shuffle data lại cho mỗi epoch để đảm bảo randomness
            random.shuffle(train_examples)
            train_dataset = SentenceLabelDataset(train_examples)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

            # Training 1 epoch với warmup (chỉ epoch đầu)
            current_warmup = warmup_steps if epoch == 0 else 0

            # Manual training loop để control được từng epoch
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=1,
                warmup_steps=current_warmup,
                show_progress_bar=True,
            )

            # Evaluate trên train set (sample)
            _ = train_evaluator(model)  # Chỉ print, không cần lưu

            # Evaluate trên validation set
            valid_score = valid_evaluator(model)

            # Check for improvement
            if valid_score > best_score + min_delta:
                best_score = valid_score
                patience_counter = 0
                # Lưu best model
                model.save(best_model_path)
                print(f"✅ New best validation score: {best_score:.4f} - Model saved")
            else:
                patience_counter += 1
                print(f"⏳ No improvement. Patience: {patience_counter}/{patience}")

            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                break

        # Load best model
        print("=== LOADING BEST MODEL ===")
        model = SentenceTransformer(best_model_path)
        model.save(output_path)
        print(f"✅ Best model loaded and saved to {output_path}")

        # Final evaluation
        print("=== FINAL EVALUATION ===")
        final_train_score = train_evaluator(model)
        final_valid_score = valid_evaluator(model)
        print(f"Final train accuracy: {final_train_score:.4f}")
        print(f"Final validation accuracy: {final_valid_score:.4f}")

    else:
        # Training không có validation
        print("\n=== TRAINING WITHOUT VALIDATION ===")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=True,
        )
        model.save(output_path)

        # Evaluate trên train set
        final_train_score = train_evaluator(model)
        print(f"Final train accuracy: {final_train_score:.4f}")

    print(f"\n🎉 Fine-tuned embedding model saved to {output_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=1, help="Case number (1 or 2)")
    parser.add_argument(
        "--model", type=str, default="all-MiniLM-L6-v2", help="Embedding Model Name"
    )

    args = parser.parse_args()

    case = args.case
    base_model_name = args.model
    print("Loading data with case:", case)
    dataset = load_dataset("thangquang09/fake-new-imposter-hunt-in-texts")
    train_df = dataset[f"case{case}_train"].to_pandas()
    valid_df = dataset[f"case{case}_validation"].to_pandas()

    train_df["file_1"] = train_df["file_1"].apply(preprocessing)
    train_df["file_2"] = train_df["file_2"].apply(preprocessing)
    valid_df["file_1"] = valid_df["file_1"].apply(preprocessing)
    valid_df["file_2"] = valid_df["file_2"].apply(preprocessing)

    train_textclf_df = build_text_classification_data(train_df)
    valid_textclf_df = build_text_classification_data(valid_df)

    print(f"Training examples: {len(train_textclf_df)}")
    print(f"Validation examples: {len(valid_textclf_df)}")

    normalize_model_name = base_model_name.split("/")[-1].replace("-", "_")

    model = fine_tune_embedding(
        base_model_name=base_model_name,
        train_textclf_df=train_textclf_df,
        valid_textclf_df=valid_textclf_df,  # Validation set
        epochs=10,  # Tăng epochs vì có early stopping
        batch_size=16,
        output_path=f"{normalize_model_name}_case{case}_fine_tuned",
        patience=3,  # Chờ 3 epochs không cải thiện
        min_delta=0.001,  # Cải thiện tối thiểu 0.1%
        warmup_steps=None,  # Auto calculate
        use_all_combinations=True,  # Dùng tất cả combinations
        max_positive_pairs=1000,  # Giới hạn để không quá nhiều
    )

    print("Fine-tuning completed!")
