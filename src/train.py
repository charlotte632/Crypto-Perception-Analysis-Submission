#Fine-tune BERT to classify crypto tweets as negative, neutral, or positive.

# import python libraries for argument parsing, file handling, and time measurement

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# import Numpy, PyTorch, and scikit-learn libraries for data manipulation, model training, and evaluation.
# import Hugging Face Transformers libraries required for BERT model

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# import dataset.py module to load the crypto tweet data, split it into train, validation, 
# and test sets, and define the sentiment labels and project directory.

from dataset import DEFAULT_DATA_PATH, LABELS, PROJECT_DIR, load_sentiment_data, split_data


# Define constants for the BERT model name, label-to-ID mapping, and ID-to-label mapping.

MODEL_NAME = "google-bert/bert-base-uncased"
LABEL_TO_ID = {label: number for number, label in enumerate(LABELS)}
ID_TO_LABEL = {number: label for label, number in LABEL_TO_ID.items()}

# Helper function to format elapsed time in a human-readable format,
# returning a string in the format "HH:MM:SS".

def format_duration(seconds: float) -> str:
    """Return a human-readable elapsed time."""
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# Define a custom dataset class for handling tweet data, inheriting from the PyTorch Dataset class.
# The class takes a DataFrame, a tokenizer, and a maximum sequence length as input, and prepares 
# the data for training and evaluation.

# The __init__ method tokenizes the tweet text and converts sentiment labels to numerical IDs.

# The __len__ method returns the number of samples in the dataset, and the __getitem__ method 
# retrieves a single sample (input tensors and label) by index.

# The _getitem__ method returns a dictionary containing the input tensors and the corresponding 
# label tensor for the specified index.

class TweetDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length: int):
        self.encodings = tokenizer(
            frame["text"].tolist(), truncation=True, max_length=max_length
        )
        self.labels = [LABEL_TO_ID[label] for label in frame["label"]]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

# Define a function to compute evaluation metrics (accuracy and macro F1 score) 
# for the model's predictions.

def metrics(prediction):
    predicted = np.argmax(prediction.predictions, axis=1)
    actual = prediction.label_ids
    return {
        "accuracy": accuracy_score(actual, predicted),
        "f1_macro": f1_score(actual, predicted, average="macro"),
    }

# Define a function to parse command-line arguments for the training script 
# and provide default values if not specified.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--max-rows", type=int, default=5000, help="Use 0 for every row.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Save this run in separate model/output experiment folders.",
    )
    return parser.parse_args()

# Define the main function to execute the training and evaluation process.

# Reads command-line settings and checks for validity, sets the seed.
# records start time and creates the model and results folders. 
# Loads and cleans the labelled tweets and uses the split_data function to divide the tweets. 
# It loads the BERT model and tokenizer and converts each dataset into the format required by BERT.

def main() -> None:
    args = parse_args()
    if args.max_rows < 0:
        raise ValueError("--max-rows cannot be negative")
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 1:
        raise ValueError("Epochs, batch size and maximum length must be positive")
    set_seed(args.seed)
    run_started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_started = time.perf_counter()
    max_rows = None if args.max_rows == 0 else args.max_rows
    output_dir = PROJECT_DIR / "models" / "bert-crypto-sentiment"
    results_dir = PROJECT_DIR / "outputs" / "bert-crypto-sentiment"
    if args.experiment_name:
        output_dir = output_dir / "experiments" / args.experiment_name
        results_dir = results_dir / "experiments" / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.data}")
    frame = load_sentiment_data(args.data, max_rows=max_rows)
    print(f"Usable tweets: {len(frame):,}")
    print("Labels:", frame["label"].value_counts().to_dict())
    train_frame, validation_frame, test_frame = split_data(frame, seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=ID_TO_LABEL, label2id=LABEL_TO_ID
    )
    train_set = TweetDataset(train_frame, tokenizer, args.max_length)
    validation_set = TweetDataset(validation_frame, tokenizer, args.max_length)
    test_set = TweetDataset(test_frame, tokenizer, args.max_length)

# Sets the learning rate, batch size, number of epochs and weight decay. 
# It evaluates and saves the model after each epoch. 
# Uses validation macro-F1 to identify the best checkpoint. 
# Creates the Hugging Face Trainer and fineturnes BERT using the training treats. 
# Saves the best model and its tokeniser and records the length of training.

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=validation_set,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=metrics,
    )
    training_started = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - training_started
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)

# Uses the trained model to give sentiment predictions.
# choses the highest-scoring sentiment for each tweet. 
# Compares the predictions with the correct labels and calculates precison, recall. 
# F1 and accuracy for negative, neutral and positivie sentiment.

    prediction = trainer.predict(test_set)
    predicted = np.argmax(prediction.predictions, axis=1)
    report = classification_report(
        prediction.label_ids, predicted, target_names=LABELS, output_dict=True, zero_division=0
    )

# saves test results, records the dataset sizes, training settings and running time.
# saves it in a separate run-summary JSON file and then returns info.
    with (results_dir / "test_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    total_seconds = time.perf_counter() - total_started
    summary = {
        "started_at": run_started_at,
        "model": args.model,
        "usable_tweets": len(frame),
        "train_tweets": len(train_frame),
        "validation_tweets": len(validation_frame),
        "test_tweets": len(test_frame),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length_tokens": args.max_length,
        "training_seconds": round(training_seconds, 2),
        "total_seconds": round(total_seconds, 2),
    }
    summary_path = results_dir / f"run_summary_{run_id}.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print("\nTest results")
    print(
        classification_report(
            prediction.label_ids,
            predicted,
            target_names=LABELS,
            zero_division=0,
        )
    )
    print(f"Training time: {format_duration(training_seconds)}")
    print(f"Total run time: {format_duration(total_seconds)}")
    print(f"Saved model: {output_dir}")
    print(f"Saved report: {results_dir / 'test_report.json'}")
    print(f"Saved timing summary: {summary_path}")

# Checks script was run directly and calls main function.

if __name__ == "__main__":
    main()
