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


def validate_split_args(test_size: float, val_size: float):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1).")

    if not 0 < val_size < 1:
        raise ValueError("val_size must be in (0, 1).")

    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be < 1.")


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
    validate_split_args(test_size=args.test_size, val_size=args.val_size)

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.clean_dir, exist_ok=True)
    os.makedirs(args.mapping_dir, exist_ok=True)

    dataset = load_banking77()
    train_ds = dataset["train"]
    test_ds = dataset["test"]

    label_names = list(train_ds.features["label"].names)
    label2id = {label_name: idx for idx, label_name in enumerate(label_names)}
    id2label = {idx: label_name for label_name, idx in label2id.items()}

    # Save raw/original dataset first.
    raw_train_df = pd.DataFrame({
        "text": list(train_ds["text"]),
        "label": list(train_ds["label"]),
    })
    raw_test_df = pd.DataFrame({
        "text": list(test_ds["text"]),
        "label": list(test_ds["label"]),
    })

    raw_train_df["label"] = raw_train_df["label"].astype(int)
    raw_test_df["label"] = raw_test_df["label"].astype(int)
    raw_train_df["label_name"] = raw_train_df["label"].map(id2label)
    raw_test_df["label_name"] = raw_test_df["label"].map(id2label)

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

    # Build full processed dataset, then split into train/val/test.
    full_df = pd.concat([raw_train_df, raw_test_df], ignore_index=True)
    full_df["text"] = full_df["text"].apply(normalize_text)
    full_df = full_df[["text", "label", "label_name"]]

    train_val_df, test_df = train_test_split(
        full_df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=full_df["label"],
    )

    val_ratio_from_train_val = args.val_size / (1.0 - args.test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_from_train_val,
        random_state=args.seed,
        stratify=train_val_df["label"],
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
    print(f"Raw train saved to: {raw_train_path}")
    print(f"Raw test saved to:  {raw_test_path}")
    print(f"Category saved to:  {raw_category_path}")
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
    parser.add_argument("--mapping_dir", type=str, default="outputs")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
