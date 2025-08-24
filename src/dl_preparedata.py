import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from load_data import read_texts_from_dir

# READ DATA

train_df = read_texts_from_dir("./data/fake-or-real-the-impostor-hunt/data/train")
train_df_label = pd.read_csv("./data/fake-or-real-the-impostor-hunt/data/train.csv")
test_df = pd.read_csv("./data/X_test_ground_truth.csv")

train_df['label'] = train_df_label['real_text_id']
test_df.rename(columns={'ground_truth_guess': 'label'}, inplace=True)

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

train_data.to_csv("./data/data_for_dl/case1_train.csv")
valid_data.to_csv("./data/data_for_dl/case1_valid.csv")


# MERGE DATA
merged_df = pd.concat([train_df, test_df], ignore_index=True)
print(len(merged_df))


# Split Data, shuffle
train_df, val_df = train_test_split(merged_df, test_size=100, random_state=42, shuffle=True)

print("Case 2: Train merge test")
print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)

train_df.to_csv("./data/data_for_dl/case2_train.csv", index=False)
val_df.to_csv("./data/data_for_dl/case2_valid.csv", index=False)
