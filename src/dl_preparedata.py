import os

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from load_data import read_texts_from_dir

load_dotenv()

# READ DATA

train_df = read_texts_from_dir("./data/fake-or-real-the-impostor-hunt/data/train")
train_df_label = pd.read_csv("./data/fake-or-real-the-impostor-hunt/data/train.csv")
test_df = pd.read_csv("./data/X_test_ground_truth.csv")

train_df['label'] = train_df_label['real_text_id']
test_df.rename(columns={'ground_truth_guess': 'label'}, inplace=True)
train_df.dropna(inplace=True)

# Case1: Only Train
# choose 20 samples from train_df for valid
valid_data = train_df.sample(n=20, random_state=42)
train_data = train_df.drop(valid_data.index)
# augmented train_data by swap file_1, file_2 and swap label from 1, 2 to 2, 1
augmented_data = train_data.copy()
augmented_data['file_1'], augmented_data['file_2'] = train_data['file_2'], train_data['file_1']
augmented_data['label'] = augmented_data['label'].replace({1: 2, 2: 1})
train_data = pd.concat([train_data, augmented_data], ignore_index=True)

print("Case 1: Only Train")
print("Train shape:", train_data.shape)
print("Validation shape:", valid_data.shape)

# shuffle train_data va valid_data

train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
valid_data = valid_data.sample(frac=1, random_state=42).reset_index(drop=True)

train_data.to_csv("./data/data_for_dl/case1_train.csv")
valid_data.to_csv("./data/data_for_dl/case1_valid.csv")


# MERGE DATA
merged_df = pd.concat([train_df, test_df], ignore_index=True)
merged_df = merged_df.dropna().reset_index(drop=True)
print(len(merged_df))


# Split Data, shuffle
train_df, val_df = train_test_split(merged_df, test_size=100, random_state=42, shuffle=True)

print("Case 2: Train merge test")
print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)

# shuffle train_df va val_df
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)

train_df.drop(columns=['id'], inplace=True)
val_df.drop(columns=['id'], inplace=True)

train_df.to_csv("./data/data_for_dl/case2_train.csv", index=False)
val_df.to_csv("./data/data_for_dl/case2_valid.csv", index=False)

def create_dataset_dict():
    """
    Tạo DatasetDict với cấu trúc phẳng:
    {
        'case1_train': Dataset,
        'case1_validation': Dataset,
        'case2_train': Dataset,
        'case2_validation': Dataset
    }
    """
    
    # Đọc các file CSV
    case1_train = pd.read_csv("./data/data_for_dl/case1_train.csv")
    case1_valid = pd.read_csv("./data/data_for_dl/case1_valid.csv")
    case2_train = pd.read_csv("./data/data_for_dl/case2_train.csv")
    case2_valid = pd.read_csv("./data/data_for_dl/case2_valid.csv")
    
    # Xóa cột index nếu có
    for df in [case1_train, case1_valid, case2_train, case2_valid]:
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)
    
    # Tạo Dataset từ DataFrame
    case1_train_dataset = Dataset.from_pandas(case1_train)
    case1_valid_dataset = Dataset.from_pandas(case1_valid)
    case2_train_dataset = Dataset.from_pandas(case2_train)
    case2_valid_dataset = Dataset.from_pandas(case2_valid)
    
    # Tạo DatasetDict với cấu trúc phẳng
    dataset_dict = DatasetDict({
        'case1_train': case1_train_dataset,
        'case1_validation': case1_valid_dataset,
        'case2_train': case2_train_dataset,
        'case2_validation': case2_valid_dataset
    })
    
    return dataset_dict

def upload_dataset():
    """Upload dataset lên Hugging Face Hub"""
    from huggingface_hub import login
    
    # Đăng nhập (cần HF token)
    # login()
    
    # Tạo dataset
    dataset = create_dataset_dict()
    
    # Upload lên Hub
    dataset.push_to_hub(
        "thangquang09/fake-new-imposter-hunt-in-texts",
        token=os.getenv("HF_TOKEN")
    )
    
    print("Dataset uploaded successfully!")
    return dataset


upload_dataset()