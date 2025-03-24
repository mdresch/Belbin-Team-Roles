# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
import json  # Import the json module
import os

from config import (
    BASE_MODEL_NAME,
    NUM_LABELS,
    LEARNING_RATE,
    NUM_EPOCHS,
    BATCH_SIZE,
    TRAIN_DATA_PATH,
    FINE_TUNED_MODEL_PATH,
    VALIDATION_DATA_PATH,
    TEST_DATA_PATH
)

# Load data.  We only need train and test now.
def load_data(train_path, test_path):  # Removed validation_path
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        #val_df = pd.read_csv(validation_path) # No longer needed

        # Check if dataframes are empty
        if train_df.empty or test_df.empty:
            raise ValueError("One or more dataframes are empty. Check CSV files.")

        # Check for necessary columns, including 'belbin_role'
        if not all(col in train_df.columns for col in ['sentence', 'belbin_role', 'label']):
            raise ValueError("Train CSV must have 'sentence', 'belbin_role', and 'label' columns.")
        if not all(col in test_df.columns for col in ['sentence', 'belbin_role', 'label']):
            raise ValueError("Test CSV must have 'sentence', 'belbin_role', and 'label' columns.")

        return train_df, test_df  # Return only train and test DataFrames
    except FileNotFoundError as e:
        print(f"Error: File not found. {e}")
        exit(1)
    except pd.errors.EmptyDataError as e:
        print(f"Error: CSV file is empty. {e}")
        exit(1)
    except pd.errors.ParserError as e:
        print(f"Error: CSV parsing error. {e}")
        exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)


def prepare_data(train_df, test_df):
    """Prepares the data for training: combines role and sentiment, converts to numerical."""

    # Combine belbin_role and label into a single string label
    train_df['combined_label'] = train_df['belbin_role'] + "_" + train_df['label']
    test_df['combined_label'] = test_df['belbin_role'] + "_" + test_df['label']

    # Create a mapping from the combined label to a unique integer ID
    label_map = {label: i for i, label in enumerate(train_df['combined_label'].unique())}
    print("DEBUG: label_map:", label_map) # Print the label map
    # Important: Update config.NUM_LABELS with the number of unique labels.
    global NUM_LABELS  # Access the global variable
    NUM_LABELS = len(label_map)
    print (f"DEBUG: setting NUM_LABELS to {NUM_LABELS}")

    # Convert combined labels to numerical values
    train_df['label'] = train_df['combined_label'].map(label_map)
    test_df['label'] = test_df['combined_label'].map(label_map)

    # Drop the combined_label and belbin_role columns
    train_df = train_df.drop(columns=['combined_label', 'belbin_role'])
    test_df = test_df.drop(columns=['combined_label', 'belbin_role'])

    # Check for missing values AFTER mapping (in case some labels weren't mapped)
    if train_df.isnull().values.any() or test_df.isnull().values.any():
        raise ValueError("Missing values found in data after label mapping.  Check your CSV for inconsistent labels.")

    return train_df, test_df, label_map  # Return the label_map!


def tokenize_data(train_df, test_df, tokenizer):
    """Tokenizes the data using the provided tokenizer."""
    train_encodings = tokenizer(train_df['sentence'].tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(test_df['sentence'].tolist(), truncation=True, padding=True)

    return train_encodings, test_encodings


class Dataset(torch.utils.data.Dataset):
    """Custom Dataset class for PyTorch."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    """Computes accuracy.  You can add other metrics (precision, recall, F1) here."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = (preds == labels).mean()
    return {'accuracy': acc}

def main():
     # Load and prepare data.  Now only uses train and test.
    train_df, test_df = load_data(TRAIN_DATA_PATH, TEST_DATA_PATH) #Removed , VALIDATION_DATA_PATH
    train_df, test_df, label_map = prepare_data(train_df, test_df) #Removed, val_df

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=NUM_LABELS) # Use updated NUM_LABELS

    # Tokenize data.  Now only uses train and test.
    train_encodings, test_encodings = tokenize_data(train_df, test_df, tokenizer) # Removed val_encodings

    # Create datasets.  Now only train and test.
    train_dataset = Dataset(train_encodings, train_df['label'].tolist())
    # val_dataset = Dataset(val_encodings, val_df['label'].tolist()) # No longer needed
    test_dataset = Dataset(test_encodings, test_df['label'].tolist())


    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,  # Larger batch size for evaluation
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,  # Add some weight decay for regularization
        logging_dir='./logs',
        logging_steps=10,   # Log more frequently
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch", # Save at the end of each epoch
        load_best_model_at_end=True,  # Load the best model (based on validation) at the end
        metric_for_best_model="accuracy",  # Use accuracy to determine the best model
        greater_is_better=True, #For accuracy, greater is better
    )

    # Create Trainer.  Use training data for validation, since we combined val/test.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,  # Use the *training* dataset for validation during training.
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model (on the *test* set)
    eval_results = trainer.evaluate(test_dataset)
    print(f"Evaluation Results on Test Set: {eval_results}")

    # Save the fine-tuned model and tokenizer
    trainer.save_model(FINE_TUNED_MODEL_PATH)
    tokenizer.save_pretrained(FINE_TUNED_MODEL_PATH)
    # Save the label_map to a JSON file
    with open(os.path.join(FINE_TUNED_MODEL_PATH, "label_map.json"), "w") as f:
        json.dump(label_map, f)
    print(f"Fine-tuned model, tokenizer, and label map saved to {FINE_TUNED_MODEL_PATH}")



if __name__ == "__main__":
    main()