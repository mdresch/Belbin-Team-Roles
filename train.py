# train.py
import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import logging
import os
import json
import sys

# Set up logging (to both console and file)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to console
        logging.FileHandler("train.log"),  # Log to a file
    ],
)

def load_data(train_file="train.csv", test_file="test.csv"):
    try:
        train_df = pd.read_csv(train_file, header=None)
        test_df = pd.read_csv(test_file, header=None)

        # --- ROBUST COLUMN CHECK ---
        if len(train_df.columns) != 4:
            raise ValueError(f"train.csv has {len(train_df.columns)} columns, expected 4.")
        if len(test_df.columns) != 4:
            raise ValueError(f"test.csv has {len(test_df.columns)} columns, expected 4.")

        train_df.columns = ['sentence', 'belbin_role', 'label', 'combined_label']
        test_df.columns = ['sentence', 'belbin_role', 'label', 'combined_label']

        return train_df, test_df
    except Exception as e:
        logging.exception("Error loading data: %s", e)
        raise

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def main():
    try:
        train_df, test_df = load_data()

        # Create label map
        label_map = {label: i for i, label in enumerate(train_df['combined_label'].unique())}
        logging.debug(f"label_map: {label_map}")

        # Map combined labels to integers
        train_labels = train_df['combined_label'].map(label_map).tolist()
        test_labels = test_df['combined_label'].map(label_map).tolist()
        test_labels = [label if label is not None else 0 for label in test_labels]

        # --- Compute Class Weights ---
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        # Convert class weights to tensor and move to device
        class_weights = torch.tensor(class_weights, dtype=torch.float)

        # Load tokenizer
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

        # Tokenize data
        train_encodings = tokenizer(
            train_df['sentence'].tolist(), truncation=True, padding=True
        )
        test_encodings = tokenizer(
            test_df['sentence'].tolist(), truncation=True, padding=True
        )

        # Create datasets
        train_dataset = Dataset(train_encodings, train_labels)
        test_dataset = Dataset(test_encodings, test_labels)

        # --- Model Training ---
        num_labels = len(label_map)
        logging.debug(f"setting NUM_LABELS to {num_labels}")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=num_labels
        )
        # Check CUDA availability and set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
        model.to(device)  # Move model to the selected device
        class_weights = class_weights.to(device)  # Move class_weights to the same device as the model

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,  # Increased epochs
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True
        )

        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            acc = accuracy_score(labels, preds)
            return {"accuracy": acc}

        # --- Custom Trainer with Class Weights ---
        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # <--- Add **kwargs
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )

        trainer.train()

        # --- Evaluation ---
        eval_results = trainer.evaluate(test_dataset)
        logging.info("Evaluation Results on Test Set: %s", eval_results)

        # Get predictions
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=-1)

        # --- Classification Report ---
        report = classification_report(test_dataset.labels, preds, target_names=[key for key, value in sorted(label_map.items(), key=lambda item: item[1])], labels=list(label_map.values()))
        logging.info("Classification Report:\n%s", report)

        # --- Save Model, Tokenizer, and Label Map ---
        model_save_path = "./fine_tuned_model/"
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        label_map_save_path = os.path.join(model_save_path, "label_map.json")
        with open(label_map_save_path, "w") as f:
            json.dump(label_map, f)

        logging.info(f"Model, tokenizer, and label map saved to {model_save_path}")

    except Exception as e:
        logging.exception("An error occurred: %s", e)

if __name__ == "__main__":
    main()