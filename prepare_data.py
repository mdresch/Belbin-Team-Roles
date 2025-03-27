# prepare_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import numpy as np  # Import numpy

def prepare_data(input_file="sentences.csv", train_file="train.csv", test_file="test.csv"):
    """Loads, cleans, splits, and oversamples data."""

    # Load data
    df = pd.read_csv(input_file, header=None)
    if len(df.columns) == 3:
        df.columns = ['sentence', 'belbin_role', 'label']
    else:
        print("Error: Input CSV must have 3 columns: sentence, belbin_role, label")
        return

    # --- Data Cleaning and Preprocessing ---
    df['belbin_role'].fillna('UnknownRole', inplace=True)
    df['label'].fillna('UnknownSentiment', inplace=True)
    df['combined_label'] = df['belbin_role'] + "_" + df['label']
    df.drop_duplicates(subset=['sentence'], inplace=True)
    df['sentence'] = df['sentence'].str.strip()
    df['belbin_role'] = df['belbin_role'].str.strip()
    df['label'] = df['label'].str.strip()

    # --- Numerical Encoding ---
    label_encoder_role = LabelEncoder()
    df['belbin_role_encoded'] = label_encoder_role.fit_transform(df['belbin_role'])
    label_encoder_sentiment = LabelEncoder()
    df['label_encoded'] = label_encoder_sentiment.fit_transform(df['label'])

    # --- Data Splitting ---
    single_instance_classes = df['combined_label'].value_counts()
    single_instance_classes = single_instance_classes[single_instance_classes == 1].index.tolist()
    df_for_splitting = df[~df['combined_label'].isin(single_instance_classes)]

    train_df, test_df = train_test_split(
        df_for_splitting, test_size=0.2, random_state=42, stratify=df_for_splitting['combined_label']
    )

    if single_instance_classes:
        single_instance_df = df[df['combined_label'].isin(single_instance_classes)]
        train_df = pd.concat([train_df, single_instance_df])
        print(f"Warning: {len(single_instance_classes)} classes with only one instance were added to the training set.")

    # --- SMOTE (AFTER Splitting, on Encoded Data) ---

    # 1. Prepare data for SMOTE
    X_train = train_df[['belbin_role_encoded', 'label_encoded']]
    y_train = train_df['combined_label']

    # 2. Determine the appropriate k_neighbors value
    min_samples = y_train.value_counts().min()
    k_neighbors = min(min_samples - 1, 5)  # Use min_samples -1, but no more than 5
    if k_neighbors < 1:
        print("Warning: Some classes have only one instance, even after handling single-instance classes. Skipping SMOTE for those.")
        # Option 1: Skip SMOTE entirely (simplest)
        # In this case, just don't apply SMOTE.
        # train_df_resampled = train_df.copy()  # Keep original training data
        # Option 2: Use RandomOverSampler (see commented-out code below)
        # Option 3: Do nothing and proceed
        #
        #Let's go with Option 1, and skip the resampling.
        train_df_resampled = train_df

    else:
        # 3. Apply SMOTE with the determined k_neighbors
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    #Option 2:
    # from imblearn.over_sampling import RandomOverSampler
    # ros = RandomOverSampler(random_state=42)
    # X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
         # 4. Combine back with the original sentence data
        train_df_resampled = pd.DataFrame(X_train_resampled, columns=['belbin_role_encoded', 'label_encoded'])
        train_df_resampled['combined_label'] = y_train_resampled
        train_df_resampled = train_df_resampled.merge(train_df[['sentence','belbin_role_encoded', 'label_encoded', 'belbin_role', 'label']].drop_duplicates(), on=['belbin_role_encoded', 'label_encoded'], how='left')


    # --- Save to CSV ---
    train_df_resampled[['sentence', 'belbin_role', 'label', 'combined_label']].to_csv(train_file, index=False, header=None)
    test_df[['sentence', 'belbin_role', 'label', 'combined_label']].to_csv(test_file, index=False, header=None)

    print(f"Prepared data: {len(train_df_resampled)} training samples, {len(test_df)} testing samples")
    print(f"Train data saved to {train_file}")
    print(f"Test data saved to {test_file}")

if __name__ == "__main__":
    if not os.path.exists('.'):
        os.makedirs('.')
    prepare_data()