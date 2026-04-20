import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REQUIRED_COLUMNS = ["text", "label", "label_name"]


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split(csv_path: str, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{split_name} split is missing required columns {missing_columns}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[REQUIRED_COLUMNS].copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    df["label_name"] = df["label_name"].astype(str)
    return df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[["text", "label"]], preserve_index=False)


def save_training_plots(log_history, output_dir: str):
    train_loss_points = []
    eval_loss_points = []
    metric_points = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    for log in log_history:
        if "loss" in log and "step" in log:
            train_loss_points.append((int(log["step"]), float(log["loss"])))

        if "eval_loss" in log and "epoch" in log:
            eval_loss_points.append((float(log["epoch"]), float(log["eval_loss"])))

        if "epoch" in log:
            epoch = float(log["epoch"])
            for metric_name in metric_points:
                metric_key = f"eval_{metric_name}"
                if metric_key in log:
                    metric_points[metric_name].append((epoch, float(log[metric_key])))

    loss_plot_path = os.path.join(output_dir, "loss_curve.png")
    metrics_plot_path = os.path.join(output_dir, "metric_performance.png")

    if train_loss_points or eval_loss_points:
        plt.figure(figsize=(10, 6))

        if train_loss_points:
            train_steps = [x[0] for x in train_loss_points]
            train_losses = [x[1] for x in train_loss_points]
            plt.plot(train_steps, train_losses, marker="o", label="train_loss")

        if eval_loss_points:
            eval_epochs = [x[0] for x in eval_loss_points]
            eval_losses = [x[1] for x in eval_loss_points]
            plt.plot(eval_epochs, eval_losses, marker="s", label="eval_loss")

        plt.title("Loss During Fine-tuning")
        plt.xlabel("Step / Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(loss_plot_path, dpi=200)
        plt.close()

    has_metric_points = any(len(points) > 0 for points in metric_points.values())
    if has_metric_points:
        plt.figure(figsize=(10, 6))

        for metric_name, points in metric_points.items():
            if not points:
                continue

            epochs = [x[0] for x in points]
            values = [x[1] for x in points]
            plt.plot(epochs, values, marker="o", label=metric_name)

        plt.title("Metric Performance During Fine-tuning")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(metrics_plot_path, dpi=200)
        plt.close()

    return {
        "loss_plot": loss_plot_path if train_loss_points or eval_loss_points else None,
        "metrics_plot": metrics_plot_path if has_metric_points else None,
    }


def main(config_path: str):
    config = load_config(config_path)
    set_seed(int(config["seed"]))

    with open(config["label2id_path"], "r", encoding="utf-8") as f:
        label2id = json.load(f)

    with open(config["id2label_path"], "r", encoding="utf-8") as f:
        id2label_raw = json.load(f)

    id2label = {int(k): v for k, v in id2label_raw.items()}
    num_labels = len(label2id)

    train_df = load_split(config["train_csv"], "train")
    val_df = load_split(config["val_csv"], "val")
    test_df = load_split(config["test_csv"], "test")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=int(config["max_length"]),
        )

    train_dataset = to_dataset(train_df).map(tokenize_function, batched=True)
    val_dataset = to_dataset(val_df).map(tokenize_function, batched=True)
    test_dataset = to_dataset(test_df).map(tokenize_function, batched=True)

    train_dataset = train_dataset.remove_columns(["text"])
    val_dataset = val_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    test_dataset.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    os.makedirs(config["output_dir"], exist_ok=True)

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=int(config["logging_steps"]),
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["batch_size"]),
        per_device_eval_batch_size=int(config["batch_size"]),
        num_train_epochs=float(config["num_train_epochs"]),
        weight_decay=float(config["weight_decay"]),
        warmup_ratio=float(config["warmup_ratio"]),
        optim=config["optimizer"],
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=int(config["seed"]),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("===== TRAINING CONFIG =====")
    print(f"Optimizer: {config['optimizer']}")
    print(f"Regularization techniques: {config['regularization_techniques']}")
    print(f"Augmentation techniques: {config['augmentation_techniques']}")

    trainer.train()
    plot_paths = save_training_plots(trainer.state.log_history, config["output_dir"])

    val_results = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="val")
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

    print("===== VALIDATION RESULTS =====")
    for key, value in val_results.items():
        print(f"{key}: {value}")

    print("===== TEST RESULTS =====")
    for key, value in test_results.items():
        print(f"{key}: {value}")

    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    with open(os.path.join(config["output_dir"], "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    with open(os.path.join(config["output_dir"], "id2label.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, ensure_ascii=False, indent=2)

    print(f"Model checkpoint and tokenizer saved to: {config['output_dir']}")
    if plot_paths["loss_plot"]:
        print(f"Loss curve saved to: {plot_paths['loss_plot']}")
    if plot_paths["metrics_plot"]:
        print(f"Metric performance plot saved to: {plot_paths['metrics_plot']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(args.config)
