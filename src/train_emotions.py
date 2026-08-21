#Fine-tune BERT to detect several emotions in one crypto tweet.

# import tools for command-line setting, saving results, and measuring time
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# import Numpy, PyTorch, and scikit-learn libraries for data manipulation, model training, and evaluation.
# import Hugging Face Transformers libraries required for BERT model    

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
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
# and test sets, and define the emotion labels and project directory.

from dataset import DEFAULT_DATA_PATH, EMOTIONS, PROJECT_DIR, load_emotion_data


MODEL_NAME = "google-bert/bert-base-uncased"


def format_duration(seconds: float) -> str:
    """Return a human-readable elapsed time."""
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# Dataset class for handling emotion-labelled tweets.
# The class takes a DataFrame, a tokenizer, and a maximum sequence length as input, and prepares
# the data for training and evaluation.

class EmotionDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length: int):
        self.encodings = tokenizer(frame["text"].tolist(), truncation=True, max_length=max_length)
        self.labels = frame[EMOTIONS].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)
        return item

# Trainer that makes rare emotion labels matter more during learning.

class WeightedEmotionTrainer(Trainer):

    def __init__(self, *args, pos_weight: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs)
        loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(outputs.logits.device)
        )(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# measure validation performance using micro-F1, macro-F1 and per-emotion F1 scores.

def metrics(prediction):
    # A score at least 0.5 means the emotion is present.
    predicted = (1 / (1 + np.exp(-prediction.predictions)) >= 0.5).astype(int)
    actual = (prediction.label_ids >= 0.5).astype(int)
    scores = {
        "f1_micro": f1_score(actual, predicted, average="micro", zero_division=0),
        "f1_macro": f1_score(actual, predicted, average="macro", zero_division=0),
    }
    for emotion, score in zip(EMOTIONS, f1_score(actual, predicted, average=None, zero_division=0)):
        scores[f"f1_{emotion.lower()}"] = score
    return scores

# convert BERT's raw output scores into probabilities between 0 and 1.

def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-logits))

# use the validation set to find the best threshold for each emotion,
# so that the F1 score is maximized for each emotion.

def tune_thresholds(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probabilities = sigmoid(logits)
    actual = (labels >= 0.5).astype(int)
    candidates = np.arange(0.10, 0.91, 0.05)
    thresholds = []
    for column in range(len(EMOTIONS)):
        scores = [
            f1_score(
                actual[:, column],
                probabilities[:, column] >= threshold,
                zero_division=0,
            )
            for threshold in candidates
        ]
        thresholds.append(float(candidates[int(np.argmax(scores))]))
    return np.asarray(thresholds)

# calculate overall and per-emotion precision, recall and F1 scores for the model's predictions,
# using the specified thresholds to determine whether each emotion is present or absent in each tweet.

def multilabel_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, float]:
    probabilities = sigmoid(logits)
    actual = (labels >= 0.5).astype(int)
    predicted = (probabilities >= thresholds).astype(int)
    output = {
        "f1_micro": f1_score(actual, predicted, average="micro", zero_division=0),
        "f1_macro": f1_score(actual, predicted, average="macro", zero_division=0),
        "precision_micro": precision_score(actual, predicted, average="micro", zero_division=0),
        "recall_micro": recall_score(actual, predicted, average="micro", zero_division=0),
    }
    for index, emotion in enumerate(EMOTIONS):
        key = emotion.lower()
        output[f"precision_{key}"] = precision_score(
            actual[:, index],
            predicted[:, index],
            zero_division=0,
        )
        output[f"recall_{key}"] = recall_score(
            actual[:, index],
            predicted[:, index],
            zero_division=0,
        )
        output[f"f1_{key}"] = f1_score(
            actual[:, index],
            predicted[:, index],
            zero_division=0,
        )
    return {key: float(value) for key, value in output.items()}

# define the training settings.

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
        "--model-dir",
        type=Path,
        default=PROJECT_DIR / "models" / "bert-crypto-emotions",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "bert-crypto-emotions",
    )
    return parser.parse_args()

# run the complete training and evaluation process, saving the model and results in the specified directories.

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

# creates the model and results folders.

    model_dir = args.model_dir
    results_dir = args.results_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

# loads the emotion-labelled tweets and uses the split_data function to divide the tweets into train, validation and test sets.

    frame = load_emotion_data(args.data, max_rows=None)
    if args.max_rows > 0 and len(frame) > args.max_rows:
        frame = frame.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
    print(f"Usable tweets: {len(frame):,}")
    print("Emotion labels (scores above zero):", (frame[EMOTIONS] > 0).sum().to_dict())

# Splits the data into train, validation, and test sets with a 70/15/15 ratio,
# ensuring reproducibility with a random seed defaulted to 42.

    train_frame, remainder = train_test_split(frame, test_size=0.30, random_state=args.seed)
    validation_frame, test_frame = train_test_split(
        remainder,
        test_size=0.50,
        random_state=args.seed,
    )
# Rare emotions receive a larger penalty when the model misses them.
    positive_mass = train_frame[EMOTIONS].sum()
    pos_weight = ((len(train_frame) - positive_mass) / positive_mass).clip(lower=1, upper=8)
    print("Class weights:", pos_weight.round(2).to_dict())

# Loads the BERT model and tokenizer and converts each dataset into the format required by BERT.

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(EMOTIONS),
        problem_type="multi_label_classification",
        id2label={index: emotion for index, emotion in enumerate(EMOTIONS)},
        label2id={emotion: index for index, emotion in enumerate(EMOTIONS)},
    )

# configure trainer including learning rate, batch size, number of epochs and weight decay.
# evaluates and saves the model after each epoch.

    trainer = WeightedEmotionTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(model_dir / "checkpoints"),
            learning_rate=2e-5,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            report_to="none",
        ),

        # supply the training and validation datasets, tokenizer, data collator, metrics function, 
        # and class weights to the trainer

        train_dataset=EmotionDataset(train_frame, tokenizer, args.max_length),
        eval_dataset=EmotionDataset(validation_frame, tokenizer, args.max_length),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=metrics,
        pos_weight=torch.tensor(pos_weight.to_numpy(), dtype=torch.float32),
    )

# fine tunes BERT, records the training time and saves model.

    training_started = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - training_started
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)

# use validation predicctions to select thresholds without using test set.

    validation_set = EmotionDataset(validation_frame, tokenizer, args.max_length)
    test_set = EmotionDataset(test_frame, tokenizer, args.max_length)
    validation_prediction = trainer.predict(validation_set)
    tuned_thresholds = tune_thresholds(
        validation_prediction.predictions,
        validation_prediction.label_ids,
    )

# Uses the trained model to give emotion predictions on the test set.

    test_prediction = trainer.predict(test_set)
    test_metrics = multilabel_metrics(
        test_prediction.predictions,
        test_prediction.label_ids,
        tuned_thresholds,
    )
    threshold_values = {
        emotion: round(float(value), 2)
        for emotion, value in zip(EMOTIONS, tuned_thresholds)
    }

# saves the tuned thresholds and test metrics in JSON files.

    with (model_dir / "emotion_thresholds.json").open("w", encoding="utf-8") as file:
        json.dump(threshold_values, file, indent=2)
    with (results_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(test_metrics, file, indent=2)

# saves test results, records the dataset sizes, training settings and running time.

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
        "emotion_thresholds": threshold_values,
        "training_seconds": round(training_seconds, 2),
        "total_seconds": round(total_seconds, 2),
    }
    summary_path = results_dir / f"run_summary_{run_id}.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

# display final results, including the tuned thresholds, test metrics, 
# training time, total run time, and locations of saved model and summary.

    print("\nValidation-tuned thresholds:", threshold_values)
    print("Test results:", test_metrics)
    print(f"Training time: {format_duration(training_seconds)}")
    print(f"Total run time: {format_duration(total_seconds)}")
    print(f"Saved model: {model_dir}")
    print(f"Saved timing summary: {summary_path}")


# starts training if the script is run directly.
if __name__ == "__main__":
    main()
