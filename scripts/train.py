import argparse
import json
import os
from datetime import datetime, timezone
from time import perf_counter

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
        return yaml.safe_load(f) or {}


def write_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, ensure_ascii=False, indent=2)


def to_serializable(value):
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, datetime):
        return value.isoformat()

    return value


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


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


def save_training_plots(log_history, report_dir: str):
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

    loss_plot_path = os.path.join(report_dir, "line_loss.png")
    metrics_plot_path = os.path.join(report_dir, "line_performance.png")

    if train_loss_points or eval_loss_points:
        plt.figure(figsize=(10, 6))

        if train_loss_points:
            train_steps = [x[0] for x in train_loss_points]
            train_losses = [x[1] for x in train_loss_points]
            plt.plot(train_steps, train_losses, marker="o", label="train_loss")

        if eval_loss_points:
            eval_epochs = [x[0] for x in eval_loss_points]
            eval_losses = [x[1] for x in eval_loss_points]
            plt.plot(eval_epochs, eval_losses, marker="s", label="val_loss")

        plt.title("Training and Validation Loss")
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

        plt.title("Validation Metric Performance")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(metrics_plot_path, dpi=200)
        plt.close()

    return {
        "line_loss": loss_plot_path if train_loss_points or eval_loss_points else None,
        "line_performance": metrics_plot_path if has_metric_points else None,
    }


def save_log_history(log_history, report_dir: str):
    history_df = pd.DataFrame([to_serializable(log) for log in log_history])
    history_path = os.path.join(report_dir, "train_log_history.csv")
    history_df.to_csv(history_path, index=False)
    return history_path


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "non_trainable_parameters": int(total - trainable),
    }


def build_metrics_row(stage: str, metrics: dict, prefix: str, wall_time_seconds: float):
    row = {
        "stage": stage,
        "loss": metrics.get(f"{prefix}_loss"),
        "accuracy": metrics.get(f"{prefix}_accuracy"),
        "precision": metrics.get(f"{prefix}_precision"),
        "recall": metrics.get(f"{prefix}_recall"),
        "f1": metrics.get(f"{prefix}_f1"),
        "runtime_seconds": metrics.get(f"{prefix}_runtime"),
        "samples_per_second": metrics.get(f"{prefix}_samples_per_second"),
        "steps_per_second": metrics.get(f"{prefix}_steps_per_second"),
        "epoch": metrics.get("epoch"),
        "total_flos": metrics.get("total_flos"),
        "wall_time_seconds": wall_time_seconds,
    }
    return {k: to_serializable(v) for k, v in row.items()}


def build_train_metrics_row(train_metrics: dict, wall_time_seconds: float):
    row = {
        "stage": "train",
        "loss": train_metrics.get("train_loss"),
        "accuracy": train_metrics.get("train_accuracy"),
        "precision": train_metrics.get("train_precision"),
        "recall": train_metrics.get("train_recall"),
        "f1": train_metrics.get("train_f1"),
        "runtime_seconds": train_metrics.get("train_runtime"),
        "samples_per_second": train_metrics.get("train_samples_per_second"),
        "steps_per_second": train_metrics.get("train_steps_per_second"),
        "epoch": train_metrics.get("epoch"),
        "total_flos": train_metrics.get("total_flos"),
        "wall_time_seconds": wall_time_seconds,
    }
    return {k: to_serializable(v) for k, v in row.items()}


def build_model_info(model, tokenizer, config, dataset_sizes):
    model_config = model.config.to_dict()
    hidden_size = model_config.get("hidden_size", model_config.get("dim"))

    info = {
        "model_name": config["model_name"],
        "model_type": model_config.get("model_type"),
        "architectures": model_config.get("architectures"),
        "num_labels": model_config.get("num_labels"),
        "hidden_size": hidden_size,
        "vocab_size": model_config.get("vocab_size"),
        "max_position_embeddings": model_config.get("max_position_embeddings"),
        "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None),
        "dataset_sizes": dataset_sizes,
    }
    info.update(count_parameters(model))
    return info


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
    dataset_sizes = {
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
    }

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

    output_dir = ensure_dir(config["output_dir"])
    report_dir = ensure_dir(config.get("report_dir", os.path.join(output_dir, "train_report")))

    training_args = TrainingArguments(
        output_dir=output_dir,
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

    run_started_at = datetime.now(timezone.utc)
    total_start = perf_counter()

    print("===== TRAINING CONFIG =====")
    print(f"Optimizer: {config['optimizer']}")
    print(f"Regularization techniques: {config['regularization_techniques']}")
    print(f"Augmentation techniques: {config['augmentation_techniques']}")
    print(f"Train report dir: {report_dir}")

    train_start = perf_counter()
    train_result = trainer.train()
    train_wall_time = perf_counter() - train_start

    log_history_path = save_log_history(trainer.state.log_history, report_dir)
    plot_paths = save_training_plots(trainer.state.log_history, report_dir)

    val_start = perf_counter()
    val_results = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="val")
    val_wall_time = perf_counter() - val_start

    test_start = perf_counter()
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    test_wall_time = perf_counter() - test_start

    total_wall_time = perf_counter() - total_start
    run_finished_at = datetime.now(timezone.utc)

    metrics_rows = [
        build_train_metrics_row(train_result.metrics, train_wall_time),
        build_metrics_row("validation", val_results, "val", val_wall_time),
        build_metrics_row("test", test_results, "test", test_wall_time),
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv_path = os.path.join(report_dir, "metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    time_summary = {
        "started_at_utc": run_started_at,
        "finished_at_utc": run_finished_at,
        "train_wall_time_seconds": train_wall_time,
        "validation_wall_time_seconds": val_wall_time,
        "test_wall_time_seconds": test_wall_time,
        "total_wall_time_seconds": total_wall_time,
    }

    train_params = {
        "config": config,
        "training_arguments": training_args.to_dict(),
        "dataset_sizes": dataset_sizes,
    }

    model_info = build_model_info(model, tokenizer, config, dataset_sizes)

    summary = {
        "report_dir": report_dir,
        "output_dir": output_dir,
        "metrics_csv": metrics_csv_path,
        "train_log_history_csv": log_history_path,
        "plots": plot_paths,
        "time_summary": time_summary,
        "train_metrics": train_result.metrics,
        "validation_metrics": val_results,
        "test_metrics": test_results,
        "train_params": train_params,
        "model_info": model_info,
    }

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, ensure_ascii=False, indent=2)

    write_json(os.path.join(report_dir, "time_summary.json"), time_summary)
    write_json(os.path.join(report_dir, "train_params.json"), train_params)
    write_json(os.path.join(report_dir, "model_params.json"), model_info)
    write_json(os.path.join(report_dir, "summary.json"), summary)

    print("===== VALIDATION RESULTS =====")
    for key, value in val_results.items():
        print(f"{key}: {value}")

    print("===== TEST RESULTS =====")
    for key, value in test_results.items():
        print(f"{key}: {value}")

    print(f"Model checkpoint and tokenizer saved to: {output_dir}")
    print(f"Training metrics saved to: {metrics_csv_path}")
    print(f"Training log history saved to: {log_history_path}")
    if plot_paths["line_loss"]:
        print(f"Loss curve saved to: {plot_paths['line_loss']}")
    if plot_paths["line_performance"]:
        print(f"Performance curve saved to: {plot_paths['line_performance']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(args.config)
