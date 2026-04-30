import argparse
import json
import os

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def normalize_text(text: str) -> str:
    text = str(text).strip().lower().replace("\n", " ")
    text = " ".join(text.split())
    return text


def load_banking77():
    """Load BANKING77 with a compatibility fallback for older/newer datasets versions."""
    try:
        return load_dataset("banking77")
    except Exception:
        return load_dataset("PolyAI/banking77")


def load_cached_raw_data(raw_dir: str):
    train_path = os.path.join(raw_dir, "train.csv")
    test_path = os.path.join(raw_dir, "test.csv")
    category_path = os.path.join(raw_dir, "category.json")
    categories_path = os.path.join(raw_dir, "categories.json")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        return None

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if os.path.exists(category_path):
        with open(category_path, "r", encoding="utf-8") as f:
            category = json.load(f)
        label2id = {str(label): int(idx) for label, idx in category["label2id"].items()}
        id2label = {int(idx): str(label) for idx, label in category["id2label"].items()}
    elif os.path.exists(categories_path):
        with open(categories_path, "r", encoding="utf-8") as f:
            categories = json.load(f)
        label2id = {str(label): idx for idx, label in enumerate(categories)}
        id2label = {idx: str(label) for label, idx in label2id.items()}
    else:
        all_categories = pd.concat([train_df["category"], test_df["category"]]).dropna().astype(str)
        categories = sorted(all_categories.unique().tolist())
        label2id = {label: idx for idx, label in enumerate(categories)}
        id2label = {idx: label for label, idx in label2id.items()}

    return train_df, test_df, label2id, id2label


def validate_split_args(val_size: float):
    if not 0 < val_size < 1:
        raise ValueError("val_size must be in (0, 1).")


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_category_statistics(train_df, val_df, test_df):
    train_counts = train_df["label_name"].value_counts().rename("train_count")
    val_counts = val_df["label_name"].value_counts().rename("val_count")
    test_counts = test_df["label_name"].value_counts().rename("test_count")

    stats_df = pd.concat([train_counts, val_counts, test_counts], axis=1).fillna(0).astype(int)
    stats_df["total_count"] = (
        stats_df["train_count"] + stats_df["val_count"] + stats_df["test_count"]
    )

    stats_df = stats_df.reset_index().rename(columns={"index": "label_name"})
    stats_df = stats_df.sort_values("label_name").reset_index(drop=True)
    return stats_df


def sample_subset(full_df: pd.DataFrame, samples_per_label: int | None, seed: int) -> pd.DataFrame:
    if samples_per_label is None or samples_per_label <= 0:
        return full_df.reset_index(drop=True)

    sampled_parts = (
        group.sample(n=min(len(group), samples_per_label), random_state=seed)
        for _, group in full_df.groupby("label")
    )
    return pd.concat(sampled_parts, ignore_index=True)


def clean_model_df(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df[["text", "label", "label_name"]].copy()
    clean_df["text"] = clean_df["text"].apply(normalize_text)
    clean_df["label"] = clean_df["label"].astype(int)
    clean_df["label_name"] = clean_df["label_name"].astype(str)
    return clean_df


def normalize_raw_schema(df: pd.DataFrame, label2id: dict, id2label: dict) -> pd.DataFrame:
    normalized_df = df.copy()

    if "category" in normalized_df.columns:
        normalized_df["label_name"] = normalized_df["category"].astype(str)
        normalized_df["label"] = normalized_df["label_name"].map(label2id)
    elif "label" in normalized_df.columns:
        normalized_df["label"] = normalized_df["label"].astype(int)
        if "label_name" not in normalized_df.columns:
            normalized_df["label_name"] = normalized_df["label"].map(id2label)
    else:
        raise ValueError(
            "Raw data must contain either a category column or a label column. "
            f"Found columns: {list(normalized_df.columns)}"
        )

    if normalized_df["label"].isna().any():
        missing_labels = sorted(normalized_df.loc[normalized_df["label"].isna(), "label_name"].unique().tolist())
        raise ValueError(f"Found labels not present in category mapping: {missing_labels}")

    normalized_df["label"] = normalized_df["label"].astype(int)
    normalized_df["label_name"] = normalized_df["label_name"].astype(str)
    return normalized_df[["text", "label", "label_name"]]


def summarize_counts(series: pd.Series):
    return {
        "min": int(series.min()),
        "max": int(series.max()),
        "avg": round(float(series.mean()), 2),
    }


def print_sample_statistics(raw_train_df, raw_test_df, train_df, val_df, test_df, num_labels):
    total_raw = len(raw_train_df) + len(raw_test_df)
    total_clean = len(train_df) + len(val_df) + len(test_df)

    print("===== SAMPLE STATISTICS =====")
    print(f"Raw train samples: {len(raw_train_df)}")
    print(f"Raw test samples:  {len(raw_test_df)}")
    print(f"Raw total samples: {total_raw}")
    print(f"Clean train samples: {len(train_df)}")
    print(f"Clean val samples:   {len(val_df)}")
    print(f"Clean test samples:  {len(test_df)}")
    print(f"Clean total samples: {total_clean}")
    print(f"Number of labels:    {num_labels}")


def print_category_statistics(category_stats_df: pd.DataFrame):
    train_summary = summarize_counts(category_stats_df["train_count"])
    val_summary = summarize_counts(category_stats_df["val_count"])
    test_summary = summarize_counts(category_stats_df["test_count"])
    total_summary = summarize_counts(category_stats_df["total_count"])

    print("===== CATEGORY STATISTICS =====")
    print(f"Categories: {len(category_stats_df)}")
    print(
        "Train per-category (min/max/avg): "
        f"{train_summary['min']}/{train_summary['max']}/{train_summary['avg']}"
    )
    print(
        "Val per-category (min/max/avg):   "
        f"{val_summary['min']}/{val_summary['max']}/{val_summary['avg']}"
    )
    print(
        "Test per-category (min/max/avg):  "
        f"{test_summary['min']}/{test_summary['max']}/{test_summary['avg']}"
    )
    print(
        "Total per-category (min/max/avg): "
        f"{total_summary['min']}/{total_summary['max']}/{total_summary['avg']}"
    )


def main(args):
    validate_split_args(val_size=args.val_size)

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.clean_dir, exist_ok=True)
    os.makedirs(args.mapping_dir, exist_ok=True)

    cached_raw = load_cached_raw_data(args.raw_dir) if args.use_cached_raw else None
    if cached_raw is not None:
        raw_train_df, raw_test_df, label2id, id2label = cached_raw
        print(f"Using cached raw BANKING77 files from: {args.raw_dir}")
    else:
        dataset = load_banking77()
        train_ds = dataset["train"]
        test_ds = dataset["test"]

        label_names = list(train_ds.features["label"].names)
        label2id = {label_name: idx for idx, label_name in enumerate(label_names)}
        id2label = {idx: label_name for label_name, idx in label2id.items()}

        raw_train_df = pd.DataFrame({
            "text": list(train_ds["text"]),
            "label": list(train_ds["label"]),
        })
        raw_test_df = pd.DataFrame({
            "text": list(test_ds["text"]),
            "label": list(test_ds["label"]),
        })

    raw_train_df = normalize_raw_schema(raw_train_df, label2id, id2label)
    raw_test_df = normalize_raw_schema(raw_test_df, label2id, id2label)

    if cached_raw is None:
        raw_train_path = os.path.join(args.raw_dir, "train.csv")
        raw_test_path = os.path.join(args.raw_dir, "test.csv")
        raw_category_path = os.path.join(args.raw_dir, "category.json")
        raw_train_df.to_csv(raw_train_path, index=False)
        raw_test_df.to_csv(raw_test_path, index=False)
        save_json(
            raw_category_path,
            {
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
            },
        )
        print(f"Raw BANKING77 files saved to: {args.raw_dir}")
    else:
        print("Original files were only read; no files in sample_data/original were rewritten.")

    # Model data lives only in sample_data/clean_data.
    # Train/val are split from original train. Test keeps the original test split.
    train_source_df = clean_model_df(raw_train_df)
    test_df = clean_model_df(raw_test_df)

    train_source_df = sample_subset(train_source_df, args.samples_per_label, args.seed)
    train_df, val_df = train_test_split(
        train_source_df,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=train_source_df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    clean_train_path = os.path.join(args.clean_dir, "train.csv")
    clean_val_path = os.path.join(args.clean_dir, "val.csv")
    clean_test_path = os.path.join(args.clean_dir, "test.csv")

    train_df.to_csv(clean_train_path, index=False)
    val_df.to_csv(clean_val_path, index=False)
    test_df.to_csv(clean_test_path, index=False)

    label2id_path = os.path.join(args.mapping_dir, "label2id.json")
    id2label_path = os.path.join(args.mapping_dir, "id2label.json")
    category_stats_path = os.path.join(args.mapping_dir, "category_statistics.csv")
    dataset_stats_path = os.path.join(args.mapping_dir, "dataset_statistics.json")

    save_json(label2id_path, label2id)
    save_json(id2label_path, {str(k): v for k, v in id2label.items()})

    category_stats_df = build_category_statistics(train_df, val_df, test_df)
    category_stats_df.to_csv(category_stats_path, index=False)

    dataset_statistics = {
        "raw": {
            "train_samples": len(raw_train_df),
            "test_samples": len(raw_test_df),
            "total_samples": len(raw_train_df) + len(raw_test_df),
        },
        "clean": {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "total_samples": len(train_df) + len(val_df) + len(test_df),
            "samples_per_label": args.samples_per_label,
            "train_csv": clean_train_path,
            "val_csv": clean_val_path,
            "test_csv": clean_test_path,
            "test_source": "sample_data/original/test.csv",
        },
        "num_labels": len(label2id),
        "category_summary": {
            "train": summarize_counts(category_stats_df["train_count"]),
            "val": summarize_counts(category_stats_df["val_count"]),
            "test": summarize_counts(category_stats_df["test_count"]),
            "total": summarize_counts(category_stats_df["total_count"]),
        },
    }
    save_json(dataset_stats_path, dataset_statistics)

    print("===== PREPROCESS DONE =====")
    print(f"Clean train saved to: {clean_train_path}")
    print(f"Clean val saved to:   {clean_val_path}")
    print(f"Clean test saved to:  {clean_test_path}")
    print(f"Label2id saved to: {label2id_path}")
    print(f"Id2label saved to: {id2label_path}")
    print(f"Category statistics saved to: {category_stats_path}")
    print(f"Dataset statistics saved to:  {dataset_stats_path}")
    print_sample_statistics(
        raw_train_df=raw_train_df,
        raw_test_df=raw_test_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        num_labels=len(label2id),
    )
    print_category_statistics(category_stats_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="sample_data/original")
    parser.add_argument("--clean_dir", type=str, default="sample_data/clean_data")
    parser.add_argument("--mapping_dir", type=str, default="outputs/outputs_train/analysis_data")
    parser.add_argument("--use_cached_raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument(
        "--samples_per_label",
        type=int,
        default=0,
        help="Sample at most this many examples per intent label. Use 0 to keep the full dataset.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
