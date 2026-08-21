# cluster cryptocurrency attack discussions using pretrained BERT embeddings.

# import libraries for argument parsing, file handling, data manipulation, machine learning, and plotting.

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from transformers import AutoModel, AutoTokenizer

from replicate_paper_with_bert import DATA_DIR, EMOTION_ORDER, build_manifest, choose_device


# Define the default project folders, pretrained encoder and source-data columns.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "paper-replication-unsupervised"
DEFAULT_MODEL = "google-bert/bert-base-uncased"
TEXT_COLUMN = "clean_content"
SENTIMENT_COLUMN = "sentiment_cat"


def parse_args() -> argparse.Namespace:
    # Define the clustering settings that can be changed from the command line.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fit-samples-per-dataset", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stability-runs", type=int, default=5)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face download if the encoder is not cached.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute the fit and all dataset aggregates.",
    )
    parser.add_argument("--limit-rows", type=int, help="Diagnostic limit per CSV.")
    return parser.parse_args()


def mean_pool(last_hidden_state, attention_mask):
    # Average the real token vectors while ignoring padding added to shorter tweets.
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def embed_texts(texts, tokenizer, model, device, batch_size: int, max_length: int):
    # Convert batches of tweet text into fixed-length BERT embedding vectors.
    texts = list(texts)
    batches = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            output = model(**encoded).last_hidden_state
            vectors = mean_pool(output, encoded["attention_mask"])
            # Normalisation prevents vector size from dominating the cluster distances.
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
        batches.append(vectors.cpu().numpy().astype("float32"))
    if batches:
        return np.concatenate(batches)
    return np.empty((0, model.config.hidden_size), dtype="float32")


def balanced_fit_sample(manifest, per_dataset: int, seed: int) -> pd.DataFrame:
    # Take a balanced sample from every W, A and B file so large datasets do not dominate.
    frames = []
    usecols = [TEXT_COLUMN, SENTIMENT_COLUMN, *EMOTION_ORDER]
    for offset, item in enumerate(manifest):
        frame = pd.read_csv(
            item.path,
            usecols=usecols,
            encoding_errors="replace",
            on_bad_lines="skip",
        )
        frame[TEXT_COLUMN] = frame[TEXT_COLUMN].fillna("").astype(str).str.strip()
        frame = frame[frame[TEXT_COLUMN].ne("")]
        if len(frame) > per_dataset:
            frame = frame.sample(per_dataset, random_state=seed + offset)
        frames.append(frame)
    # Remove repeated tweets before fitting the clusters.
    sample = pd.concat(frames, ignore_index=True).drop_duplicates(subset=[TEXT_COLUMN])
    return sample.reset_index(drop=True)


def fit_clusterers(embeddings: np.ndarray, seed: int, stability_runs: int):
    # Fit label-free K-means solutions with three and five clusters.
    models, quality = {}, {}
    # Use at most 5,000 embeddings to keep the quality calculations manageable.
    metric_rng = np.random.default_rng(seed)
    metric_indices = metric_rng.choice(
        len(embeddings),
        size=min(5000, len(embeddings)),
        replace=False,
    )
    metric_vectors = embeddings[metric_indices]
    for k in (3, 5):
        model = MiniBatchKMeans(
            n_clusters=k,
            batch_size=1024,
            n_init=10,
            random_state=seed,
        )
        labels = model.fit_predict(embeddings)
        metric_labels = model.predict(metric_vectors)
        stability_labels = [labels]
        # Repeat clustering with different seeds to check whether similar groups reappear.
        for offset in range(1, stability_runs):
            repeat = MiniBatchKMeans(
                n_clusters=k, batch_size=1024, n_init=10, random_state=seed + offset
            ).fit(embeddings)
            stability_labels.append(repeat.predict(embeddings))
        stability_scores = [
            adjusted_rand_score(left, right)
            for left, right in combinations(stability_labels, 2)
        ]
        # Record cluster separation, compactness, stability and cluster sizes.
        quality[f"k{k}"] = {
            "silhouette": float(
                silhouette_score(metric_vectors, metric_labels, metric="cosine")
            ),
            "davies_bouldin": float(davies_bouldin_score(metric_vectors, metric_labels)),
            "calinski_harabasz": float(calinski_harabasz_score(metric_vectors, metric_labels)),
            "stability_runs": stability_runs,
            "seed_stability_adjusted_rand_mean": (
                float(np.mean(stability_scores)) if stability_scores else 1.0
            ),
            "seed_stability_adjusted_rand_min": (
                float(np.min(stability_scores)) if stability_scores else 1.0
            ),
            "seed_stability_adjusted_rand_max": (
                float(np.max(stability_scores)) if stability_scores else 1.0
            ),
            "cluster_sizes": np.bincount(labels, minlength=k).astype(int).tolist(),
        }
        models[k] = model
    return models, quality


def representative_tweets(
    sample: pd.DataFrame,
    embeddings: np.ndarray,
    models: dict,
    output_dir: Path,
    per_cluster: int = 10,
) -> None:
    #Save tweets nearest each fitted centre as label-free interpretive examples
    # Tweets closest to a centre are the clearest examples of what that cluster contains.
    rows = []
    for k, model in models.items():
        assigned = model.predict(embeddings)
        distances = model.transform(embeddings)
        for cluster in range(k):
            candidates = np.flatnonzero(assigned == cluster)
            nearest = candidates[np.argsort(distances[candidates, cluster])[:per_cluster]]
            for rank, index in enumerate(nearest, start=1):
                rows.append(
                    {
                        "k": k,
                        "cluster": f"C{cluster}",
                        "rank": rank,
                        "distance_to_centre": float(distances[index, cluster]),
                        "tweet": sample.iloc[index][TEXT_COLUMN],
                    }
                )
    pd.DataFrame(rows).to_csv(output_dir / "representative_cluster_tweets.csv", index=False)


def posthoc_descriptions(
    sample: pd.DataFrame,
    embeddings: np.ndarray,
    models: dict,
    output_dir: Path,
) -> None:
    # Describe completed clusters using existing labels without using them to fit K-means.
    sentiment_rows, emotion_rows = [], []
    sentiment = sample[SENTIMENT_COLUMN].fillna("unknown").astype(str).str.lower()
    emotions = sample[EMOTION_ORDER].apply(pd.to_numeric, errors="coerce").fillna(0)
    for k, model in models.items():
        clusters = model.predict(embeddings)
        for cluster in range(k):
            mask = clusters == cluster
            distribution = sentiment[mask].value_counts(normalize=True)
            sentiment_rows.append(
                {
                    "k": k,
                    "cluster": f"C{cluster}",
                    "n": int(mask.sum()),
                    "dominant_lexicon_sentiment": (
                        distribution.index[0] if len(distribution) else "unknown"
                    ),
                    "dominant_share": (
                        float(distribution.iloc[0]) if len(distribution) else 0
                    ),
                    **{
                        f"share_{label}": float(distribution.get(label, 0))
                        for label in ("negative", "neutral", "positive")
                    },
                }
            )
            prevalence = (
                (emotions.loc[mask] >= 0.5).mean()
                if mask.any()
                else pd.Series(0, index=EMOTION_ORDER)
            )
            emotion_rows.append(
                {
                    "k": k,
                    "cluster": f"C{cluster}",
                    "n": int(mask.sum()),
                    "dominant_lexicon_emotion": prevalence.idxmax(),
                    **{
                        f"prevalence_{label}": float(prevalence[label])
                        for label in EMOTION_ORDER
                    },
                }
            )
    pd.DataFrame(sentiment_rows).to_csv(
        output_dir / "posthoc_sentiment_descriptions.csv",
        index=False,
    )
    pd.DataFrame(emotion_rows).to_csv(output_dir / "posthoc_emotion_descriptions.csv", index=False)


def analyse_dataset(item, args, tokenizer, encoder, device, models) -> dict:
    # Embed every usable tweet in one event dataset and count its assigned clusters.
    counts = {k: Counter() for k in models}
    total = 0
    remaining = args.limit_rows
    # Read large CSV files in chunks to limit memory use.
    reader = pd.read_csv(
        item.path,
        usecols=[TEXT_COLUMN],
        chunksize=args.chunk_size,
        encoding_errors="replace",
        on_bad_lines="skip",
    )
    for frame in reader:
        texts = frame[TEXT_COLUMN].fillna("").astype(str)
        texts = texts[texts.str.strip().ne("")]
        if remaining is not None:
            texts = texts.iloc[:remaining]
            remaining -= len(texts)
        if len(texts):
            # Only the tweet text is embedded and passed to the fitted cluster models.
            vectors = embed_texts(
                texts,
                tokenizer,
                encoder,
                device,
                args.batch_size,
                args.max_length,
            )
            for k, model in models.items():
                counts[k].update(map(int, model.predict(vectors)))
            total += len(texts)
            print(f"  {item.event_id}{item.kind}: {total:,} tweets", flush=True)
        if remaining == 0:
            break
    return {
        "event_id": item.event_id,
        "dataset": item.kind,
        "cryptocurrency": item.cryptocurrency,
        "event_date": item.date.date().isoformat(),
        "source_file": item.path.name,
        "n_tweets": total,
        "counts": {str(k): dict(value) for k, value in counts.items()},
    }


def write_profiles(results: list[dict], output_dir: Path) -> dict[int, pd.DataFrame]:
    # Convert cluster counts into event-level percentages and comparison tables.
    frames = {}
    for k in (3, 5):
        rows = []
        for result in results:
            for cluster in range(k):
                cluster_counts = result["counts"][str(k)]
                count = int(
                    cluster_counts.get(str(cluster), cluster_counts.get(cluster, 0))
                )
                rows.append(
                    {
                        **{
                            name: result[name]
                            for name in (
                                "event_id",
                                "dataset",
                                "cryptocurrency",
                                "event_date",
                                "n_tweets",
                            )
                        },
                        "cluster": f"C{cluster}",
                        "count": count,
                        "percentage": (
                            100 * count / result["n_tweets"]
                            if result["n_tweets"]
                            else 0
                        ),
                    }
                )
        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / f"cluster_profiles_k{k}.csv", index=False)
        # Summarise how common each cluster is across W, A and B datasets.
        summary = (
            frame.groupby(["dataset", "cluster"])["percentage"]
            .agg(mean="mean", std="std", min="min", max="max")
            .reset_index()
        )
        summary.to_csv(output_dir / f"cluster_event_summary_k{k}.csv", index=False)
        # Calculate the percentage-point difference between attack and benchmark data.
        paired = frame.pivot(
            index=["event_id", "cluster"],
            columns="dataset",
            values="percentage",
        ).reset_index()
        paired["attack_minus_benchmark_pp"] = paired["A"] - paired["B"]
        paired.to_csv(output_dir / f"cluster_attack_benchmark_differences_k{k}.csv", index=False)
        frames[k] = frame
    return frames


def plot_profiles(frame: pd.DataFrame, k: int, output_dir: Path) -> None:
    # Plot stacked cluster percentages for every event and dataset type.
    events = sorted(frame.event_id.unique(), key=lambda value: int(value[1:]))
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, sharey=True)
    colors = plt.get_cmap("tab10").colors
    for axis, kind in zip(axes, ("W", "A", "B")):
        subset = frame[frame.dataset == kind]
        bottom = np.zeros(len(events))
        for cluster in range(k):
            label = f"C{cluster}"
            values = (
                subset[subset.cluster == label]
                .set_index("event_id")
                .reindex(events)
                .percentage.to_numpy()
            )
            axis.bar(events, values, bottom=bottom, label=label, color=colors[cluster])
            bottom += values
        titles = {
            "W": "Whole event window",
            "A": "Attack subset",
            "B": "Benchmark",
        }
        axis.set_title(titles[kind])
        axis.set_ylabel("Tweets (%)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(ncol=k)
    axes[-1].set_xlabel("Attack event")
    fig.suptitle(f"Unsupervised BERT cluster profiles (k={k})")
    fig.tight_layout()
    fig.savefig(output_dir / f"cluster_profiles_k{k}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # Run the complete unsupervised embedding, clustering and profiling workflow.
    args = parse_args()
    # Stop early if any numerical setting is invalid.
    if args.stability_runs < 2:
        raise ValueError("--stability-runs must be at least 2")
    if args.fit_samples_per_dataset < 1:
        raise ValueError("--fit-samples-per-dataset must be positive")
    if args.batch_size < 1 or args.chunk_size < 1 or args.max_length < 1:
        raise ValueError("Batch size, chunk size and maximum length must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.data_dir)
    device = choose_device(args.device)
    # Create a fingerprint so results from different settings use different caches.
    run_settings = {
        "model": args.model,
        "sample": args.fit_samples_per_dataset,
        "max_length": args.max_length,
        "seed": args.seed,
        "stability_runs": args.stability_runs,
    }
    fingerprint = hashlib.sha256(
        json.dumps(run_settings, sort_keys=True).encode()
    ).hexdigest()[:12]
    cache_dir = args.output_dir / "cache" / fingerprint
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Load cached Hugging Face files unless downloading has been explicitly allowed.
    local_only = not args.allow_download
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=local_only)
    encoder = AutoModel.from_pretrained(
        args.model,
        local_files_only=local_only,
    ).to(device).eval()
    print(f"Encoder: {args.model}; device: {device}; run: {fingerprint}")

    fit_path = cache_dir / "fit_sample.pkl"
    vectors_path = cache_dir / "fit_embeddings.npy"
    models_path = cache_dir / "clusterers.joblib"
    cache_is_complete = fit_path.exists() and vectors_path.exists() and models_path.exists()
    # Reuse a completed clustering fit unless --force requests a fresh run.
    if not args.force and cache_is_complete:
        sample = pd.read_pickle(fit_path)
        vectors = np.load(vectors_path)
        payload = joblib.load(models_path)
        models, quality = payload["models"], payload["quality"]
        print(f"Loaded fitted clusters from {cache_dir}")
    else:
        print("Building balanced, label-free fit sample...")
        sample = balanced_fit_sample(manifest, args.fit_samples_per_dataset, args.seed)
        print(f"Embedding {len(sample):,} unique sampled tweets...")
        # K-means receives only BERT text embeddings; no sentiment or emotion labels.
        vectors = embed_texts(
            sample[TEXT_COLUMN],
            tokenizer,
            encoder,
            device,
            args.batch_size,
            args.max_length,
        )
        models, quality = fit_clusterers(vectors, args.seed, args.stability_runs)
        sample.to_pickle(fit_path)
        np.save(vectors_path, vectors)
        joblib.dump({"models": models, "quality": quality}, models_path)
    # Produce examples and optional label-based descriptions after clustering is complete.
    representative_tweets(sample, vectors, models, args.output_dir)
    posthoc_descriptions(sample, vectors, models, args.output_dir)

    # Apply the fitted cluster models to all 51 W/A/B datasets.
    results = []
    for item in manifest:
        limit_suffix = "_limited" if args.limit_rows else ""
        cache = cache_dir / f"{item.event_id}_{item.kind}{limit_suffix}.json"
        if cache.exists() and not args.force:
            results.append(json.loads(cache.read_text(encoding="utf-8")))
            print(f"Loaded cache: {item.event_id}{item.kind}")
        else:
            print(f"Analysing {item.path.name}")
            result = analyse_dataset(item, args, tokenizer, encoder, device, models)
            cache.write_text(json.dumps(result, indent=2), encoding="utf-8")
            results.append(result)
    # Save the numerical profiles and create their visual comparisons.
    frames = write_profiles(results, args.output_dir)
    for k, frame in frames.items():
        plot_profiles(frame, k, args.output_dir)
    # Record the method, settings and cluster-quality results for reproducibility.
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": (
            "Unsupervised clustering of mean-pooled, L2-normalised pretrained "
            "BERT embeddings"
        ),
        "encoder": args.model,
        "labels_used_for_fitting": False,
        "posthoc_descriptions_use_lexicon_labels": True,
        "fit_sample_unique_tweets": len(sample),
        "fit_samples_per_dataset_requested": args.fit_samples_per_dataset,
        "max_length": args.max_length,
        "seed": args.seed,
        "device": str(device),
        "stability_runs": args.stability_runs,
        "limited_rows_per_file": args.limit_rows,
        "quality": quality,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Saved unsupervised analysis to {args.output_dir}")


# Start the workflow only when this script is run directly.
if __name__ == "__main__":
    main()
