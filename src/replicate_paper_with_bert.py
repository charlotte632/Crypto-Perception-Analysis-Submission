#Reproduce the CVC 2025 perception profiles with fine-tuned BERT models.

#The source repository supplies three datasets for each attack event:
#W_i (whole event window), A_i (attack-specific subset), and B_i (a normal-time
#benchmark). This script replaces the paper's lexicon inference with the local
#single-label sentiment BERT and multi-label emotion BERT, while retaining the
#paper's aggregation definitions and visual comparisons.

# import standard libraries for argument parsing, JSON handling, regular expressions, and data manipulation

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# import libraries to load datasets, run both models and creat graphs.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define the locations of the datasets, trained models and replication outputs.

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "SM_perceptions_51-Attacks"
    / "Datasets"
    / "cleaned datasets"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "paper-replication-bert"
SENTIMENT_MODEL_DIR = PROJECT_DIR / "models" / "bert-crypto-sentiment"
EMOTION_MODEL_DIR = PROJECT_DIR / "models" / "bert-crypto-emotions"
SENTIMENT_ORDER = ["negative", "neutral", "positive"]
EMOTION_ORDER = ["Happy", "Angry", "Surprise", "Sad", "Fear"]
DATASET_ORDER = ["W", "A", "B"]
DATASET_NAMES = {
    "W": "whole event window",
    "A": "attack subset",
    "B": "benchmark",
}
CRYPTOCURRENCY_ALIASES = {
    # The supplied files use the ticker for one Verge benchmark and the full
    # name for its corresponding whole/attack datasets.
    "xvg": "verge",
}


@dataclass(frozen=True)
# Store the identifying information for one event dataset file.
class DatasetFile:
    path: Path
    kind: str
    cryptocurrency: str
    date: datetime
    event_id: str = ""

# Define the analysis settings that can be changed from the command line.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sentiment-model", type=Path, default=SENTIMENT_MODEL_DIR)
    parser.add_argument("--emotion-model", type=Path, default=EMOTION_MODEL_DIR)
    parser.add_argument("--text-column", default="clean_content")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--emotion-threshold", type=float, default=0.5)
    parser.add_argument(
        "--emotion-thresholds",
        type=Path,
        help="Optional JSON file containing one validation-tuned threshold per emotion.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached per-dataset aggregates.",
    )
    parser.add_argument("--limit-rows", type=int, help="Diagnostic row limit per CSV.")
    return parser.parse_args()

# Select the requested processor, or automatically use CUDA, MPS or the CPU.
def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Read a dataset filename and identify its type, cryptocurrency and collection date.

def parse_dataset_filename(path: Path) -> DatasetFile | None:
    match = re.fullmatch(
        r"Dataset_(?:(benchmark|sub)_)?(.+?)_(\d{2})_(\d{2})_(\d{4})\.csv",
        path.name,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    prefix, crypto, day, month, year = match.groups()
    kind = {None: "W", "sub": "A", "benchmark": "B"}[prefix.lower() if prefix else None]
    cryptocurrency = CRYPTOCURRENCY_ALIASES.get(crypto.lower(), crypto.lower())
    return DatasetFile(
        path=path,
        kind=kind,
        cryptocurrency=cryptocurrency,
        date=datetime(int(year), int(month), int(day)),
    )

# Match the 51 source files into 17 W/A/B attack-event groups.

def build_manifest(data_dir: Path) -> list[DatasetFile]:
    parsed = [
        item
        for path in data_dir.glob("Dataset_*.csv")
        if (item := parse_dataset_filename(path))
    ]
    whole = sorted(
        (item for item in parsed if item.kind == "W"),
        key=lambda item: item.date,
    )
    attacks = {(item.cryptocurrency, item.date): item for item in parsed if item.kind == "A"}
    benchmarks_by_crypto: dict[str, list[DatasetFile]] = {}
    for item in parsed:
        if item.kind == "B":
            benchmarks_by_crypto.setdefault(item.cryptocurrency, []).append(item)

    manifest: list[DatasetFile] = []
    used_benchmarks: set[Path] = set()
    for index, w_item in enumerate(whole, start=1):
        event_id = f"E{index}"
        a_item = attacks.get((w_item.cryptocurrency, w_item.date))
        candidates = [
            item
            for item in benchmarks_by_crypto.get(w_item.cryptocurrency, [])
            if item.path not in used_benchmarks
        ]
        # Benchmark filenames normally retain the event window's day and move
        # it to a neighbouring month. Prefer that match before absolute date
        # distance; nearest-date matching alone swaps repeated Verge/ETC events.
        b_item = (
            min(
                candidates,
                key=lambda item: (
                    abs(item.date.day - w_item.date.day),
                    abs((item.date - w_item.date).days),
                ),
            )
            if candidates
            else None
        )
        if not a_item or not b_item:
            raise ValueError(f"Could not form W/A/B triplet for {w_item.path.name}")
        used_benchmarks.add(b_item.path)
        manifest.extend(
            # Keep one canonical event date across W_i, A_i, and B_i. The
            # benchmark collection date remains recoverable from its filename.
            DatasetFile(item.path, item.kind, item.cryptocurrency, w_item.date, event_id)
            for item in (w_item, a_item, b_item)
        )
    # Confirm that all 17 events contain one W, one A and one B dataset.
    if len(manifest) != 51:
        raise ValueError(f"Expected 17 W/A/B triplets (51 files), found {len(manifest)} files")
    return manifest

# Check the model's label names and convert them to the standard spelling used in the results.

def normalise_labels(model, required: list[str]) -> list[str]:
    labels = [model.config.id2label[index] for index in range(model.config.num_labels)]
    canonical = {label.lower(): label for label in required}
    try:
        return [canonical[label.lower()] for label in labels]
    except KeyError as exc:
        raise ValueError(f"Unexpected model label {exc.args[0]!r}; expected {required}") from exc

# Process tweets in smaller batches and return sentiment or emotion probability scores.
def batched_scores(
    texts,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    multilabel: bool,
):
    texts = list(texts)
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits
            scores = torch.sigmoid(logits) if multilabel else torch.softmax(logits, dim=1)
        yield scores.cpu().numpy()

# Apply both trained BERT models to one dataset and aggregate its perception profile.

def analyse_file(
    item: DatasetFile,
    args: argparse.Namespace,
    sentiment_tokenizer,
    sentiment_model,
    emotion_tokenizer,
    emotion_model,
    device,
    sentiment_labels,
    emotion_labels,
) -> dict:
    sentiment_counts = Counter({label: 0 for label in SENTIMENT_ORDER})
    emotion_counts = Counter({label: 0 for label in EMOTION_ORDER})
    emotion_sums = Counter({label: 0.0 for label in EMOTION_ORDER})
    combinations: Counter[str] = Counter()
    total = 0
    rows_remaining = args.limit_rows

    reader = pd.read_csv(
        item.path,
        usecols=[args.text_column],
        chunksize=args.chunk_size,
        encoding_errors="replace",
        on_bad_lines="skip",
    )
    for frame in reader:
        texts = frame[args.text_column].fillna("").astype(str)
        texts = texts[texts.str.strip().ne("")]
        if rows_remaining is not None:
            texts = texts.iloc[:rows_remaining]
            rows_remaining -= len(texts)
        if texts.empty:
            if rows_remaining == 0:
                break
            continue

        sentiment_batches = batched_scores(
            texts,
            sentiment_tokenizer,
            sentiment_model,
            device,
            args.batch_size,
            args.max_length,
            False,
        )
        emotion_batches = batched_scores(
            texts, emotion_tokenizer, emotion_model, device, args.batch_size, args.max_length, True
        )
        for sentiment_scores, emotion_scores in zip(sentiment_batches, emotion_batches):
            for index in sentiment_scores.argmax(axis=1):
                sentiment_counts[sentiment_labels[int(index)]] += 1
            detected = emotion_scores >= args.emotion_threshold_values
            for column, label in enumerate(emotion_labels):
                emotion_counts[label] += int(detected[:, column].sum())
                emotion_sums[label] += float(emotion_scores[:, column].sum())
            for row in detected:
                active = [emotion_labels[i] for i, value in enumerate(row) if value]
                combination = (
                    "+".join(sorted(active, key=EMOTION_ORDER.index))
                    if active
                    else "None"
                )
                combinations[combination] += 1
            total += len(sentiment_scores)
        print(f"  {item.event_id}{item.kind}: {total:,} tweets", flush=True)
        if rows_remaining == 0:
            break

    if total == 0:
        raise ValueError(f"No usable text found in {item.path}")
    return {
        "event_id": item.event_id,
        "dataset": item.kind,
        "cryptocurrency": item.cryptocurrency,
        "event_date": item.date.date().isoformat(),
        "source_file": item.path.name,
        "n_tweets": total,
        "sentiment_counts": dict(sentiment_counts),
        "emotion_counts": dict(emotion_counts),
        "emotion_probability_sums": dict(emotion_sums),
        "emotion_combinations": dict(combinations),
    }


def cache_path(output_dir: Path, item: DatasetFile, threshold_key: str, limited: bool) -> Path:
    suffix = "_limited" if limited else ""
    return output_dir / "cache" / f"{item.event_id}_{item.kind}_t{threshold_key}{suffix}.json"

# Convert the aggregated results into tables for sentiment, emotion and dataset information.
def make_tables(results: list[dict], output_dir: Path) -> tuple[pd.DataFrame, ...]:
    manifest_rows = []
    sentiment_rows = []
    volume_rows = []
    intensity_rows = []
    combination_rows = []
    for result in results:
        common = {
            key: result[key]
            for key in (
                "event_id",
                "dataset",
                "cryptocurrency",
                "event_date",
                "n_tweets",
            )
        }
        manifest_rows.append({**common, "source_file": result["source_file"]})
        for label in SENTIMENT_ORDER:
            count = result["sentiment_counts"][label]
            sentiment_rows.append(
                {
                    **common,
                    "sentiment": label,
                    "count": count,
                    "percentage": 100 * count / result["n_tweets"],
                }
            )
        for label in EMOTION_ORDER:
            count = result["emotion_counts"][label]
            volume_rows.append(
                {
                    **common,
                    "emotion": label,
                    "count": count,
                    "percentage": 100 * count / result["n_tweets"],
                }
            )
            intensity_rows.append(
                {
                    **common,
                    "emotion": label,
                    "mean_probability": (
                        result["emotion_probability_sums"][label]
                        / result["n_tweets"]
                    ),
                }
            )
        for combination, count in result["emotion_combinations"].items():
            combination_rows.append(
                {
                    **common,
                    "combination": combination,
                    "count": count,
                    "percentage": 100 * count / result["n_tweets"],
                }
            )

    frames = tuple(
        map(
            pd.DataFrame,
            (
                manifest_rows,
                sentiment_rows,
                volume_rows,
                intensity_rows,
                combination_rows,
            ),
        )
    )
    names = (
        "event_manifest",
        "sentiment_profiles",
        "emotion_volume",
        "emotion_intensity",
        "emotion_combinations",
    )
    for name, frame in zip(names, frames):
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    return frames

# Convert event names such as E10 into numbers so they are sorted correctly.
def event_sort_key(value: str) -> int:
    return int(value[1:])

# Create stacked bar charts comparing sentiment across W, A and B datasets.
def plot_sentiment(frame: pd.DataFrame, output_dir: Path) -> None:
    events = sorted(frame.event_id.unique(), key=event_sort_key)
    colors = {"negative": "#c0392b", "neutral": "#95a5a6", "positive": "#27ae60"}
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, sharey=True)
    for axis, kind in zip(axes, DATASET_ORDER):
        subset = frame[frame.dataset == kind]
        bottom = np.zeros(len(events))
        for label in SENTIMENT_ORDER:
            values = (
                subset[subset.sentiment == label]
                .set_index("event_id")
                .reindex(events)
                .percentage.to_numpy()
            )
            axis.bar(events, values, bottom=bottom, label=label.title(), color=colors[label])
            bottom += values
        axis.set_title(f"{kind}: {DATASET_NAMES[kind]}")
        axis.set_ylabel("Tweets (%)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(ncol=3, loc="upper center")
    axes[-1].set_xlabel("Attack event")
    fig.suptitle("BERT sentiment profiles SP(D)", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "sentiment_profiles.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# Create heatmaps showing the emotion combinations detected for every attack event.
def plot_emotion_heatmaps(frame: pd.DataFrame, output_dir: Path) -> None:
    events = sorted(frame.event_id.unique(), key=event_sort_key)
    combinations = ["None"] + [
        "+".join(combo)
        for mask in range(1, 1 << len(EMOTION_ORDER))
        for combo in [[EMOTION_ORDER[i] for i in range(len(EMOTION_ORDER)) if mask & (1 << i)]]
    ]
    fig, axes = plt.subplots(3, 1, figsize=(15, 16), sharex=True)
    maximum = max(1.0, frame.percentage.max())
    for axis, kind in zip(axes, DATASET_ORDER):
        pivot = (
            frame[frame.dataset == kind]
            .pivot_table(index="combination", columns="event_id", values="percentage", fill_value=0)
            .reindex(index=combinations, columns=events, fill_value=0)
        )
        image = axis.imshow(
            pivot.to_numpy(),
            aspect="auto",
            cmap="Blues",
            vmin=0,
            vmax=maximum,
        )
        axis.set_yticks(range(len(combinations)), combinations, fontsize=7)
        axis.set_title(f"{kind}: emotion combinations")
        axis.set_ylabel("Detected emotions")
    axes[-1].set_xticks(range(len(events)), events)
    axes[-1].set_xlabel("Attack event")
    fig.colorbar(image, ax=axes, label="Tweets (%)", shrink=0.6)
    fig.suptitle("BERT emotion-volume profiles EV(D), thresholded combinations", fontsize=15)
    fig.subplots_adjust(left=0.18, right=0.9, top=0.94, bottom=0.05, hspace=0.18)
    fig.savefig(output_dir / "emotion_combination_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# Plot the average BERT probability for each emotion across W, A and B datasets.
def plot_emotion_intensity(frame: pd.DataFrame, output_dir: Path) -> None:
    events = sorted(frame.event_id.unique(), key=event_sort_key)
    fig, axes = plt.subplots(len(EMOTION_ORDER), 1, figsize=(14, 15), sharex=True, sharey=True)
    x = np.arange(len(events))
    for axis, emotion in zip(axes, EMOTION_ORDER):
        subset = frame[frame.emotion == emotion]
        for kind, marker in zip(DATASET_ORDER, ("o", "s", "^")):
            values = (
                subset[subset.dataset == kind]
                .set_index("event_id")
                .reindex(events)
                .mean_probability
            )
            axis.plot(x, values, marker=marker, linewidth=1.5, label=kind)
        axis.set_title(emotion)
        axis.set_ylabel("Mean probability")
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=3)
    axes[-1].set_xticks(x, events)
    axes[-1].set_xlabel("Attack event")
    fig.suptitle("BERT emotion-intensity profiles EI(D)", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "emotion_intensity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# Run the complete BERT replication, from loading files to saving tables and graphs.
def main() -> None:
    args = parse_args()
    if not 0 < args.emotion_threshold < 1:
        raise ValueError("--emotion-threshold must be between 0 and 1")
    if args.batch_size < 1 or args.chunk_size < 1 or args.max_length < 1:
        raise ValueError("Batch size, chunk size and maximum length must be positive")
    threshold_path = args.emotion_thresholds
    if threshold_path is None:
        candidate = args.emotion_model / "emotion_thresholds.json"
        threshold_path = candidate if candidate.is_file() else None
    if threshold_path:
        threshold_data = json.loads(threshold_path.read_text(encoding="utf-8"))
        args.emotion_threshold_values = np.asarray(
            [float(threshold_data[label]) for label in EMOTION_ORDER]
        )
        if np.any((args.emotion_threshold_values <= 0) | (args.emotion_threshold_values >= 1)):
            raise ValueError("Every tuned emotion threshold must be between 0 and 1")
        threshold_key = "tuned_" + "_".join(f"{value:g}" for value in args.emotion_threshold_values)
    else:
        args.emotion_threshold_values = np.full(len(EMOTION_ORDER), args.emotion_threshold)
        threshold_key = f"{args.emotion_threshold:g}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cache").mkdir(exist_ok=True)
    manifest = build_manifest(args.data_dir)
    device = choose_device(args.device)
    print(f"Using device: {device}")

    sentiment_tokenizer = AutoTokenizer.from_pretrained(
        args.sentiment_model,
        local_files_only=True,
    )
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        args.sentiment_model,
        local_files_only=True,
    ).to(device).eval()
    emotion_tokenizer = AutoTokenizer.from_pretrained(
        args.emotion_model,
        local_files_only=True,
    )
    emotion_model = AutoModelForSequenceClassification.from_pretrained(
        args.emotion_model,
        local_files_only=True,
    ).to(device).eval()
    sentiment_labels = normalise_labels(sentiment_model, SENTIMENT_ORDER)
    emotion_labels = normalise_labels(emotion_model, EMOTION_ORDER)

    results = []
    for item in manifest:
        cached = cache_path(args.output_dir, item, threshold_key, args.limit_rows is not None)
        if cached.is_file() and not args.force:
            results.append(json.loads(cached.read_text(encoding="utf-8")))
            print(f"Loaded cache: {item.event_id}{item.kind}")
            continue
        print(f"Analysing {item.path.name}")
        result = analyse_file(
            item,
            args,
            sentiment_tokenizer,
            sentiment_model,
            emotion_tokenizer,
            emotion_model,
            device,
            sentiment_labels,
            emotion_labels,
        )
        cached.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)

    manifest_frame, sentiment, volume, intensity, combinations = make_tables(
        results,
        args.output_dir,
    )
    plot_sentiment(sentiment, args.output_dir)
    plot_emotion_heatmaps(combinations, args.output_dir)
    plot_emotion_intensity(intensity, args.output_dir)
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": "Fine-tuned BERT inference replacing the paper's lexicon inference",
        "emotion_thresholds": {
            label: float(value)
            for label, value in zip(
                EMOTION_ORDER,
                args.emotion_threshold_values,
            )
        },
        "max_length": args.max_length,
        "device": str(device),
        "limited_rows_per_file": args.limit_rows,
        "events": manifest_frame.event_id.nunique(),
        "datasets": len(manifest_frame),
        "tweets": int(manifest_frame.n_tweets.sum()),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Saved BERT replication to {args.output_dir}")

if __name__ == "__main__":
    main()
