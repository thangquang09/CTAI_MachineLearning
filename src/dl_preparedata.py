import os

import pandas as pd
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from load_data import read_texts_from_dir

load_dotenv()

# --- READ AND PREPARE INITIAL DATA ---
print("Reading and preparing initial data...")
train_texts_df = read_texts_from_dir("./data/fake-or-real-the-impostor-hunt/data/train")
train_labels_df = pd.read_csv("./data/fake-or-real-the-impostor-hunt/data/train.csv")
test_df = pd.read_csv("./data/X_test_ground_truth.csv")

# An toàn hơn khi merge bằng 'id'
train_df = pd.merge(train_texts_df, train_labels_df, on='id')
train_df.rename(columns={'real_text_id': 'label'}, inplace=True)

test_df.rename(columns={'ground_truth_guess': 'label'}, inplace=True)
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# --- CASE 1: ONLY TRAIN DATA ---
print("\n--- Processing Case 1: Train Data Only ---")
train_data_c1, valid_data_c1 = train_test_split(train_df, test_size=20, random_state=42, shuffle=True)

# Augmentation
augmented_data = train_data_c1.copy()
augmented_data['file_1'], augmented_data['file_2'] = augmented_data['file_2'], augmented_data['file_1']
augmented_data['label'] = augmented_data['label'].map({1: 2, 2: 1})
train_data_c1 = pd.concat([train_data_c1, augmented_data], ignore_index=True)

print("Train shape (augmented):", train_data_c1.shape)
print("Validation shape:", valid_data_c1.shape)

# Giữ lại các cột cần thiết và lưu
columns_to_keep = ['file_1', 'file_2', 'label']
train_data_c1[columns_to_keep].to_csv("./data/data_for_dl/case1_train.csv", index=False)
valid_data_c1[columns_to_keep].to_csv("./data/data_for_dl/case1_valid.csv", index=False)
print("Saved case1_train.csv and case1_valid.csv")

# --- CASE 2: TRAIN + TEST DATA ---
print("\n--- Processing Case 2: Merged Train + Test Data ---")
merged_df = pd.concat([train_df, test_df], ignore_index=True)
merged_df = merged_df.dropna().reset_index(drop=True)

train_data_c2, valid_data_c2 = train_test_split(merged_df, test_size=100, random_state=42, shuffle=True)

print("Train shape:", train_data_c2.shape)
print("Validation shape:", valid_data_c2.shape)

# Giữ lại các cột cần thiết và lưu
train_data_c2[columns_to_keep].to_csv("./data/data_for_dl/case2_train.csv", index=False)
valid_data_c2[columns_to_keep].to_csv("./data/data_for_dl/case2_valid.csv", index=False)
print("Saved case2_train.csv and case2_valid.csv")


def create_dataset_dict():
    """Tạo DatasetDict từ các file CSV đã được chuẩn hóa."""
    print("\n--- Creating Hugging Face DatasetDict ---")
    
    # Đọc các file CSV đã được làm sạch
    case1_train = pd.read_csv("./data/data_for_dl/case1_train.csv")
    case1_valid = pd.read_csv("./data/data_for_dl/case1_valid.csv")
    case2_train = pd.read_csv("./data/data_for_dl/case2_train.csv")
    case2_valid = pd.read_csv("./data/data_for_dl/case2_valid.csv")
    
    # Tạo DatasetDict
    dataset_dict = DatasetDict({
        'case1_train': Dataset.from_pandas(case1_train),
        'case1_validation': Dataset.from_pandas(case1_valid),
        'case2_train': Dataset.from_pandas(case2_train),
        'case2_validation': Dataset.from_pandas(case2_valid)
    })
    
    print("DatasetDict created successfully. Features are consistent.")
    print(dataset_dict)
    return dataset_dict

def upload_dataset():
    """Upload dataset lên Hugging Face Hub."""
    print("\n--- Uploading to Hugging Face Hub ---")
    dataset = create_dataset_dict()
    
    dataset.push_to_hub(
        "thangquang09/fake-new-imposter-hunt-in-texts",
        token=os.getenv("HF_TOKEN")
    )
    
    print("\nDataset uploaded successfully!")

# Chạy quá trình upload
upload_dataset()