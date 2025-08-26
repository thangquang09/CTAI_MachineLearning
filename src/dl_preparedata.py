import os
import pickle

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

# # Augmentation
# augmented_data = train_data_c1.copy()
# augmented_data['file_1'], augmented_data['file_2'] = augmented_data['file_2'], augmented_data['file_1']
# augmented_data['label'] = augmented_data['label'].map({1: 2, 2: 1})
# train_data_c1 = pd.concat([train_data_c1, augmented_data], ignore_index=True)

print("Train shape (augmented):", train_data_c1.shape)
print("Validation shape:", valid_data_c1.shape)

# Giữ lại các cột cần thiết và lưu với encoding UTF-8
columns_to_keep = ['file_1', 'file_2', 'label']
train_data_c1[columns_to_keep].to_csv("./data/data_for_dl/case1_train.csv", index=False, encoding='utf-8')
valid_data_c1[columns_to_keep].to_csv("./data/data_for_dl/case1_valid.csv", index=False, encoding='utf-8')
print("Saved case1_train.csv and case1_valid.csv")

# Tạo dataset ngay lập tức để tránh lỗi encoding
print("Creating Case 1 datasets...")
case1_train_dataset = Dataset.from_pandas(train_data_c1[columns_to_keep].reset_index(drop=True))
case1_valid_dataset = Dataset.from_pandas(valid_data_c1[columns_to_keep].reset_index(drop=True))
print("Case 1 datasets created successfully")

# --- CASE 2: TRAIN + TEST DATA ---
print("\n--- Processing Case 2: Merged Train + Test Data ---")
merged_df = pd.concat([train_df, test_df], ignore_index=True)
merged_df = merged_df.dropna().reset_index(drop=True)

train_data_c2, valid_data_c2 = train_test_split(merged_df, test_size=100, random_state=42, shuffle=True)

print("Train shape:", train_data_c2.shape)
print("Validation shape:", valid_data_c2.shape)

# Giữ lại các cột cần thiết và lưu với encoding UTF-8
train_data_c2[columns_to_keep].to_csv("./data/data_for_dl/case2_train.csv", index=False, encoding='utf-8')
valid_data_c2[columns_to_keep].to_csv("./data/data_for_dl/case2_valid.csv", index=False, encoding='utf-8')
print("Saved case2_train.csv and case2_valid.csv")

# Tạo dataset ngay lập tức để tránh lỗi encoding
print("Creating Case 2 datasets...")
case2_train_dataset = Dataset.from_pandas(train_data_c2[columns_to_keep].reset_index(drop=True))
case2_valid_dataset = Dataset.from_pandas(valid_data_c2[columns_to_keep].reset_index(drop=True))
print("Case 2 datasets created successfully")


def create_dataset_dict(case1_train_ds=None, case1_valid_ds=None, case2_train_ds=None, case2_valid_ds=None):
    """Tạo DatasetDict từ các dataset đã được tạo hoặc đọc từ file CSV."""
    print("\n--- Creating Hugging Face DatasetDict ---")
    
    # Nếu có dataset sẵn, sử dụng chúng. Không thì đọc từ CSV
    if all([case1_train_ds, case1_valid_ds, case2_train_ds, case2_valid_ds]):
        print("Using pre-created datasets...")
        dataset_dict = DatasetDict({
            'case1_train': case1_train_ds,
            'case1_validation': case1_valid_ds,
            'case2_train': case2_train_ds,
            'case2_validation': case2_valid_ds
        })
    else:
        print("Reading from CSV files with UTF-8 encoding...")
        # Đọc các file CSV với encoding UTF-8 để tránh lỗi
        case1_train = pd.read_csv("./data/data_for_dl/case1_train.csv", encoding='utf-8')
        case1_valid = pd.read_csv("./data/data_for_dl/case1_valid.csv", encoding='utf-8')
        case2_train = pd.read_csv("./data/data_for_dl/case2_train.csv", encoding='utf-8')
        case2_valid = pd.read_csv("./data/data_for_dl/case2_valid.csv", encoding='utf-8')
        
        # Tạo DatasetDict với reset_index để tránh cột __index_level_0__
        dataset_dict = DatasetDict({
            'case1_train': Dataset.from_pandas(case1_train.reset_index(drop=True)),
            'case1_validation': Dataset.from_pandas(case1_valid.reset_index(drop=True)),
            'case2_train': Dataset.from_pandas(case2_train.reset_index(drop=True)),
            'case2_validation': Dataset.from_pandas(case2_valid.reset_index(drop=True))
        })
    
    print("DatasetDict created successfully. Features are consistent.")
    print(dataset_dict)
    return dataset_dict

def save_datasets_as_pickle(case1_train_ds, case1_valid_ds, case2_train_ds, case2_valid_ds):
    """Lưu các dataset dưới dạng pickle để backup an toàn."""
    print("\n--- Saving datasets as pickle files ---")
    
    datasets = {
        'case1_train': case1_train_ds,
        'case1_validation': case1_valid_ds,
        'case2_train': case2_train_ds,
        'case2_validation': case2_valid_ds
    }
    
    # Tạo thư mục nếu chưa có
    os.makedirs("./data/data_for_dl/pickle_datasets", exist_ok=True)
    
    # Lưu từng dataset
    for name, dataset in datasets.items():
        pickle_path = f"./data/data_for_dl/pickle_datasets/{name}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Saved {name} to {pickle_path}")
    
    print("All datasets saved as pickle files successfully!")

def load_datasets_from_pickle():
    """Tải các dataset từ pickle files."""
    print("\n--- Loading datasets from pickle files ---")
    
    datasets = {}
    names = ['case1_train', 'case1_validation', 'case2_train', 'case2_validation']
    
    for name in names:
        pickle_path = f"./data/data_for_dl/pickle_datasets/{name}.pkl"
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                datasets[name] = pickle.load(f)
            print(f"Loaded {name} from {pickle_path}")
        else:
            print(f"Warning: {pickle_path} not found!")
            return None
    
    return datasets

def upload_dataset(case1_train_ds=None, case1_valid_ds=None, case2_train_ds=None, case2_valid_ds=None):
    """Upload dataset lên Hugging Face Hub."""
    print("\n--- Uploading to Hugging Face Hub ---")
    dataset = create_dataset_dict(case1_train_ds, case1_valid_ds, case2_train_ds, case2_valid_ds)
    
    dataset.push_to_hub(
        "thangquang09/fake-new-imposter-hunt-in-texts",
        token=os.getenv("HF_TOKEN")
    )
    
    print("\nDataset uploaded successfully!")

# Chạy quá trình upload với các dataset đã tạo
print("\n--- Final Dataset Creation and Upload ---")

# Lưu dataset dưới dạng pickle để backup
save_datasets_as_pickle(case1_train_dataset, case1_valid_dataset, case2_train_dataset, case2_valid_dataset)

# Upload lên Hugging Face Hub
upload_dataset(case1_train_dataset, case1_valid_dataset, case2_train_dataset, case2_valid_dataset)