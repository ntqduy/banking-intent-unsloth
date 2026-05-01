import argparse
import json
import os
import re
import sys
import warnings
from datetime import datetime, timezone
from time import perf_counter

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

warnings.filterwarnings(
    "ignore",
    message=r"Both `max_new_tokens`.*and `max_length`.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"The attention mask API under `transformers\.modeling_attn_mask_utils`.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"transformers\.modeling_attn_mask_utils",
)
warnings.simplefilter("ignore", FutureWarning)

try:
    from unsloth import FastLanguageModel
except ImportError as exc:
    raise ImportError(
        "Unsloth is required for this training script. Install the project dependencies "
        "with `pip install -r requirements.txt` on Linux/Colab/Kaggle, then run again."
    ) from exc

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TrainingArguments, set_seed
from trl import SFTTrainer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from trl import SFTConfig
except ImportError:
    SFTConfig = None


REQUIRED_COLUMNS = ["text", "label", "label_name"]


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


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


def write_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, ensure_ascii=False, indent=2)


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


def load_label_mappings(config):
    with open(config["label2id_path"], "r", encoding="utf-8") as f:
        label2id = json.load(f)
    with open(config["id2label_path"], "r", encoding="utf-8") as f:
        id2label_raw = json.load(f)

    id2label = {int(k): v for k, v in id2label_raw.items()}
    return label2id, id2label


def build_label_list(id2label: dict) -> str:
    return ", ".join(id2label[idx] for idx in sorted(id2label))


def normalize_generated_label(text: str, label2id: dict):
    cleaned = str(text).strip().lower()
    cleaned = cleaned.splitlines()[0] if cleaned else ""
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned).strip("_")

    if cleaned in label2id:
        return cleaned

    for label in label2id:
        if label.lower() in cleaned:
            return label

    return cleaned


def build_prompt(message: str, label_list: str, response: str | None = None) -> str:
    prompt = (
        "You are an intent classifier for banking customer messages.\n"
        "Choose exactly one intent label from the allowed labels.\n"
        f"Allowed labels: {label_list}\n"
        f"Message: {message}\n"
        "Intent:"
    )

    if response is None:
        return prompt

    return f"{prompt} {response}"


def to_sft_dataset(df: pd.DataFrame, label_list: str) -> Dataset:
    rows = [
        {"text": build_prompt(row.text, label_list, row.label_name)}
        for row in df.itertuples(index=False)
    ]
    return Dataset.from_list(rows)


def compute_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def estimate_inference_flops(num_parameters: int, token_count: int):
    if num_parameters <= 0 or token_count <= 0:
        return None
    return float(2 * num_parameters * token_count)


def configure_tokenizer_for_causal_lm(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|PAD_TOKEN|>"
    tokenizer.padding_side = "left"
    return tokenizer


def normalize_model_configs(model):
    for config_attr in ("config", "generation_config"):
        model_config = getattr(model, config_attr, None)
        if model_config is None:
            continue

        config_dict = model_config.to_dict() if hasattr(model_config, "to_dict") else {}
        if "use_return_dict" in config_dict and "return_dict" not in config_dict:
            model_config.return_dict = bool(config_dict["use_return_dict"])
        if config_attr == "generation_config" and hasattr(model_config, "max_length"):
            model_config.max_length = None

    return model


def predict_labels(model, tokenizer, df: pd.DataFrame, label2id: dict, label_list: str, config: dict):
    FastLanguageModel.for_inference(model)
    device = next(model.parameters()).device
    max_length = int(config["max_seq_length"])
    max_new_tokens = int(config.get("max_new_tokens", 12))
    batch_size = int(config.get("eval_batch_size", config["batch_size"]))

    predicted_labels = []
    total_input_tokens = 0
    total_generated_tokens = 0
    started = perf_counter()

    for start_idx in range(0, len(df), batch_size):
        batch_messages = df.iloc[start_idx : start_idx + batch_size]["text"].tolist()
        prompts = [build_prompt(message, label_list) for message in batch_messages]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        prompt_lengths = inputs["input_ids"].shape[1]
        generation_max_length = prompt_lengths + max_new_tokens
        total_input_tokens += int(inputs["attention_mask"].sum().item())
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_length=generation_max_length,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = output_ids[:, prompt_lengths:]
        total_generated_tokens += int(generated_ids.numel())
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predicted_labels.extend(
            normalize_generated_label(text, label2id) for text in generated_texts
        )

    elapsed = perf_counter() - started
    predicted_ids = [label2id.get(label, -1) for label in predicted_labels]
    return {
        "predicted_labels": predicted_labels,
        "predicted_ids": predicted_ids,
        "elapsed_seconds": elapsed,
        "input_token_count": total_input_tokens,
        "generated_token_count": total_generated_tokens,
        "processed_token_count": total_input_tokens + total_generated_tokens,
    }


def evaluate_generation(model, tokenizer, df, label2id, label_list, config, split_name):
    prediction_result = predict_labels(
        model=model,
        tokenizer=tokenizer,
        df=df,
        label2id=label2id,
        label_list=label_list,
        config=config,
    )
    predicted_labels = prediction_result["predicted_labels"]
    predicted_ids = prediction_result["predicted_ids"]
    elapsed = prediction_result["elapsed_seconds"]
    total_parameters = count_parameters(model)["total_parameters"]
    estimated_flops = estimate_inference_flops(
        total_parameters,
        prediction_result["processed_token_count"],
    )
    true_ids = df["label"].tolist()
    metrics = compute_metrics(true_ids, predicted_ids)
    metrics.update(
        {
            "split": split_name,
            "num_samples": int(len(df)),
            "inference_time_seconds": float(elapsed),
            "runtime_seconds": float(elapsed),
            "average_inference_time_ms_per_sample": float(elapsed / len(df) * 1000.0) if len(df) else None,
            "fps": float(len(df) / elapsed) if elapsed > 0 else None,
            "samples_per_second": float(len(df) / elapsed) if elapsed > 0 else None,
            "model_parameter_count": total_parameters,
            "input_token_count": prediction_result["input_token_count"],
            "generated_token_count": prediction_result["generated_token_count"],
            "processed_token_count": prediction_result["processed_token_count"],
            "estimated_flops": estimated_flops,
            "estimated_tflops": (estimated_flops / 1e12) if estimated_flops is not None else None,
            "estimated_flops_per_second": (estimated_flops / elapsed) if estimated_flops is not None and elapsed > 0 else None,
            "estimated_tflops_per_second": (estimated_flops / elapsed / 1e12) if estimated_flops is not None and elapsed > 0 else None,
        }
    )
    predictions = pd.DataFrame(
        {
            "sample": df["text"].tolist(),
            "ground_truth_id": true_ids,
            "ground_truth_label": df["label_name"].tolist(),
            "predicted_id": predicted_ids,
            "predicted_label": predicted_labels,
        }
    )
    predictions["correct"] = predictions["ground_truth_id"] == predictions["predicted_id"]
    return metrics, predictions


def save_log_history(log_history, report_dir: str):
    history_path = os.path.join(report_dir, "train_log_history.csv")
    pd.DataFrame([to_serializable(log) for log in log_history]).to_csv(history_path, index=False)
    return history_path


class Tee:
    def __init__(self, *streams, skip_tokens=None):
        self.streams = streams
        self.skip_tokens = skip_tokens or []
        self._buffer = ""

    def _should_skip(self, line: str) -> bool:
        return any(token in line for token in self.skip_tokens)

    def write(self, data):
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line_with_newline = f"{line}\n"
            for stream in self.streams:
                if stream is self.streams[1] and self._should_skip(line):
                    continue
                stream.write(line_with_newline)
                stream.flush()

    def flush(self):
        if self._buffer:
            for stream in self.streams:
                if stream is self.streams[1] and self._should_skip(self._buffer):
                    continue
                stream.write(self._buffer)
                stream.flush()
            self._buffer = ""
        for stream in self.streams:
            stream.flush()


def write_train_logs(
    log_history: list,
    report_dir: str,
    config: dict,
    train_result,
    train_wall_time: float,
    started_at,
    finished_at,
):
    log_path = os.path.join(report_dir, "train_logs.txt")
    lines = [
        "===== UNSLOTH TRAINING CONFIG =====",
        f"Model: {config.get('model_name')}",
        f"Epochs: {config.get('num_train_epochs')}",
        f"Batch size: {config.get('batch_size')}",
        f"Gradient accumulation steps: {config.get('gradient_accumulation_steps')}",
        f"Learning rate: {config.get('learning_rate')}",
        f"Max sequence length: {config.get('max_seq_length')}",
        f"Load in 4bit: {config.get('load_in_4bit', True)}",
        f"Started at (UTC): {started_at}",
        f"Finished at (UTC): {finished_at}",
        "",
        "===== TRAIN LOG HISTORY =====",
    ]

    for log in log_history or []:
        step = log.get("step")
        epoch = log.get("epoch")
        loss = log.get("loss")
        eval_loss = log.get("eval_loss")
        lr = log.get("learning_rate")
        samples_per_second = log.get("samples_per_second")
        parts = []
        if step is not None:
            parts.append(f"step={step}")
        if epoch is not None:
            parts.append(f"epoch={epoch}")
        if loss is not None:
            parts.append(f"loss={loss}")
        if eval_loss is not None:
            parts.append(f"eval_loss={eval_loss}")
        if lr is not None:
            parts.append(f"lr={lr}")
        if samples_per_second is not None:
            parts.append(f"samples_per_second={samples_per_second}")
        if parts:
            lines.append(" | ".join(parts))

    lines.extend(
        [
            "",
            "===== TRAIN RESULT =====",
            f"Train loss: {train_result.metrics.get('train_loss')}",
            f"Train runtime seconds: {train_result.metrics.get('train_runtime')}",
            f"Train samples per second: {train_result.metrics.get('train_samples_per_second')}",
            f"Wall time seconds: {train_wall_time}",
        ]
    )

    write_text(log_path, "\n".join(lines))
    return log_path


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "non_trainable_parameters": int(total - trainable),
        "trainable_percent": float(trainable / total * 100.0) if total else 0.0,
    }


def save_loss_plots(log_history, report_dir: str):
    train_points = []
    eval_points = []

    for log in log_history:
        if "loss" in log:
            x_value = log.get("step", log.get("epoch", len(train_points) + 1))
            train_points.append((float(x_value), float(log["loss"])))
        if "eval_loss" in log:
            x_value = log.get("step", log.get("epoch", len(eval_points) + 1))
            eval_points.append((float(x_value), float(log["eval_loss"])))

    paths = {
        "loss_png": os.path.join(report_dir, "loss_curve.png"),
        "loss_pdf": os.path.join(report_dir, "loss_curve.pdf"),
    }

    if not train_points and not eval_points:
        return {key: None for key in paths}

    plt.figure(figsize=(9, 5))
    if train_points:
        x_values, y_values = zip(*train_points)
        plt.plot(x_values, y_values, marker="o", label="train_loss")
    if eval_points:
        x_values, y_values = zip(*eval_points)
        plt.plot(x_values, y_values, marker="s", label="eval_loss")
    plt.title("Training Loss")
    plt.xlabel("Step / Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["loss_png"], dpi=200)
    plt.savefig(paths["loss_pdf"])
    plt.close()
    return paths


def save_performance_plots(val_metrics: dict, test_metrics: dict, report_dir: str):
    metric_names = ["accuracy", "precision", "recall", "f1"]
    x_positions = np.arange(len(metric_names))
    width = 0.35

    val_scores = [float(val_metrics.get(name, 0.0)) for name in metric_names]
    test_scores = [float(test_metrics.get(name, 0.0)) for name in metric_names]

    paths = {
        "performance_png": os.path.join(report_dir, "performance_metrics.png"),
        "performance_pdf": os.path.join(report_dir, "performance_metrics.pdf"),
    }

    plt.figure(figsize=(9, 5))
    plt.bar(x_positions - width / 2, val_scores, width, label="validation")
    plt.bar(x_positions + width / 2, test_scores, width, label="test")
    plt.xticks(x_positions, metric_names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Validation and Test Performance")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["performance_png"], dpi=200)
    plt.savefig(paths["performance_pdf"])
    plt.close()
    return paths


def cuda_bf16_supported():
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def parse_optional_bool(value):
    if value in (None, "auto", "Auto", "AUTO", "none", "None"):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "y"):
            return True
        if lowered in ("false", "0", "no", "n"):
            return False
    return bool(value)


def resolve_precision_flags(config):
    configured_fp16 = parse_optional_bool(config.get("fp16"))
    configured_bf16 = parse_optional_bool(config.get("bf16"))

    if configured_fp16 is not None or configured_bf16 is not None:
        fp16 = bool(configured_fp16) if configured_fp16 is not None else False
        bf16 = bool(configured_bf16) if configured_bf16 is not None else False
    elif cuda_bf16_supported():
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False

    if fp16 and bf16:
        raise ValueError("Only one precision mode can be enabled. Set either fp16 or bf16, not both.")

    return fp16, bf16


def build_training_args(config, output_dir):
    fp16, bf16 = resolve_precision_flags(config)
    common_args = {
        "output_dir": output_dir,
        "per_device_train_batch_size": int(config["batch_size"]),
        "per_device_eval_batch_size": int(config.get("eval_batch_size", config["batch_size"])),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "learning_rate": float(config["learning_rate"]),
        "num_train_epochs": float(config["num_train_epochs"]),
        "max_steps": int(config.get("max_steps", -1)),
        "warmup_steps": int(config.get("warmup_steps", 0)),
        "weight_decay": float(config["weight_decay"]),
        "optim": config["optimizer"],
        "lr_scheduler_type": config.get("lr_scheduler_type", "linear"),
        "logging_steps": int(config["logging_steps"]),
        "eval_strategy": "epoch",
        "save_strategy": config.get("save_strategy", "no"),
        "save_total_limit": int(config.get("save_total_limit", 2)),
        "report_to": "none",
        "fp16": fp16,
        "bf16": bf16,
        "seed": int(config["seed"]),
    }

    if SFTConfig is None:
        return TrainingArguments(**common_args)

    return SFTConfig(
        **common_args,
        dataset_text_field="text",
        max_length=int(config["max_seq_length"]),
        packing=bool(config.get("packing", False)),
    )


def build_sft_trainer(model, tokenizer, train_dataset, val_dataset, training_args, config):
    try:
        return SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
        )
    except TypeError:
        return SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text",
            max_seq_length=int(config["max_seq_length"]),
            args=training_args,
            packing=bool(config.get("packing", False)),
        )


def main(config_path: str):
    config = load_config(config_path)

    output_dir = ensure_dir(config["output_dir"])
    report_dir = ensure_dir(config.get("report_dir", os.path.dirname(output_dir) or output_dir))
    train_terminal_log_path = os.path.join(report_dir, "train_terminal.log")
    terminal_log_file = open(train_terminal_log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    skip_tokens = [
        "[TRAIN] Using GPU device",
        "Unsloth 2026",
        "Fast Qwen2 patching",
        "NVIDIA",
        "Max memory",
        "Torch:",
        "CUDA:",
        "CUDA Toolkit:",
        "Triton:",
        "Bfloat16",
        "Free license",
    ]
    sys.stdout = Tee(sys.stdout, terminal_log_file, skip_tokens=skip_tokens)
    sys.stderr = Tee(sys.stderr, terminal_log_file, skip_tokens=skip_tokens)
    
    try:
        # Set CUDA device if specified in config
        device_id = config.get("device")
        if device_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(int(device_id))
            print(f"[TRAIN] Using GPU device: {device_id}")
        else:
            print("[TRAIN] Using default GPU device")

        set_seed(int(config["seed"]))

        label2id, id2label = load_label_mappings(config)
        label_list = build_label_list(id2label)

        train_df = load_split(config["train_csv"], "train")
        val_df = load_split(config["val_csv"], "val")
        test_df = load_split(config["test_csv"], "test")

        max_seq_length = int(config["max_seq_length"])
        dtype = None if config.get("dtype") in (None, "none", "None") else getattr(torch, config["dtype"])

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config["model_name"],
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=bool(config.get("load_in_4bit", True)),
        )

        model = normalize_model_configs(model)
        tokenizer = configure_tokenizer_for_causal_lm(tokenizer)

        model = FastLanguageModel.get_peft_model(
            model,
            r=int(config["lora_r"]),
            target_modules=config["target_modules"],
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            bias=config.get("lora_bias", "none"),
            use_gradient_checkpointing=config.get("use_gradient_checkpointing", True),
            random_state=int(config["seed"]),
            use_rslora=bool(config.get("use_rslora", False)),
            loftq_config=None,
        )
        model_params = count_parameters(model)

        train_dataset = to_sft_dataset(train_df, label_list)
        val_dataset = to_sft_dataset(val_df, label_list)

        final_model_dir = ensure_dir(config.get("final_model_dir", output_dir))

        training_args = build_training_args(config, output_dir)
        trainer = build_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training_args=training_args,
            config=config,
        )

        started_at = datetime.now(timezone.utc)
        train_start = perf_counter()

        print("===== UNSLOTH TRAINING CONFIG =====")
        print(f"Model: {config['model_name']}")
        print(f"Output dir: {output_dir}")
        print(f"Report dir: {report_dir}")
        print(f"Labels: {len(label2id)}")
        print(f"Train/val/test: {len(train_df)}/{len(val_df)}/{len(test_df)}")

        train_result = trainer.train()
        train_wall_time = perf_counter() - train_start

        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

        write_json(os.path.join(final_model_dir, "label2id.json"), label2id)
        write_json(os.path.join(final_model_dir, "id2label.json"), {str(k): v for k, v in id2label.items()})
        write_json(os.path.join(final_model_dir, "intent_labels.json"), sorted(label2id))

        val_metrics, val_predictions = evaluate_generation(
            model, tokenizer, val_df, label2id, label_list, config, "validation"
        )
        test_metrics, test_predictions = evaluate_generation(
            model, tokenizer, test_df, label2id, label_list, config, "test"
        )

        finished_at = datetime.now(timezone.utc)
        log_history_path = save_log_history(trainer.state.log_history, report_dir)
        train_logs_path = write_train_logs(
            trainer.state.log_history,
            report_dir,
            config,
            train_result,
            train_wall_time,
            started_at,
            finished_at,
        )
        loss_plot_paths = save_loss_plots(trainer.state.log_history, report_dir)
        performance_plot_paths = save_performance_plots(val_metrics, test_metrics, report_dir)

        metrics_df = pd.DataFrame(
            [
                {
                    "split": "train",
                    "loss": train_result.metrics.get("train_loss"),
                    "runtime_seconds": train_result.metrics.get("train_runtime"),
                    "samples_per_second": train_result.metrics.get("train_samples_per_second"),
                    "wall_time_seconds": train_wall_time,
                },
                val_metrics,
                test_metrics,
            ]
        )
        metrics_csv_path = os.path.join(report_dir, "metric.csv")
        legacy_metrics_csv_path = os.path.join(report_dir, "metrics.csv")
        metrics_json_path = os.path.join(report_dir, "metrics.json")
        metrics_df.to_csv(metrics_csv_path, index=False)
        metrics_df.to_csv(legacy_metrics_csv_path, index=False)
        write_json(
            metrics_json_path,
            {
                "train": train_result.metrics,
                "validation": val_metrics,
                "test": test_metrics,
            },
        )
        val_predictions_path = os.path.join(report_dir, "val_predictions.csv")
        test_predictions_path = os.path.join(report_dir, "test_predictions.csv")
        train_config_path = os.path.join(report_dir, "train_config.json")
        model_params_path = os.path.join(report_dir, "model_params.json")
        val_predictions.to_csv(val_predictions_path, index=False)
        test_predictions.to_csv(test_predictions_path, index=False)
        write_json(train_config_path, config)
        write_json(
            model_params_path,
            {
                "model_name": config["model_name"],
                "final_model_dir": final_model_dir,
                "max_seq_length": config["max_seq_length"],
                "load_in_4bit": config.get("load_in_4bit", True),
                "lora_r": config["lora_r"],
                "lora_alpha": config["lora_alpha"],
                "lora_dropout": config["lora_dropout"],
                "target_modules": config["target_modules"],
                **model_params,
            },
        )

        summary = {
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "output_dir": output_dir,
            "final_model_dir": final_model_dir,
            "report_dir": report_dir,
            "metrics_csv": metrics_csv_path,
            "metric_csv": metrics_csv_path,
            "legacy_metrics_csv": legacy_metrics_csv_path,
            "metrics_json": metrics_json_path,
            "train_log_history_csv": log_history_path,
            "train_logs_txt": train_logs_path,
            "val_predictions_csv": val_predictions_path,
            "test_predictions_csv": test_predictions_path,
            "train_config_json": train_config_path,
            "model_params_json": model_params_path,
            "plots": {
                **loss_plot_paths,
                **performance_plot_paths,
            },
            "train_metrics": train_result.metrics,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "model_params": model_params,
            "config": config,
        }
        write_json(os.path.join(report_dir, "summary.json"), summary)

        print("===== VALIDATION RESULTS =====")
        for key, value in val_metrics.items():
            print(f"{key}: {value}")

        print("===== TEST RESULTS =====")
        for key, value in test_metrics.items():
            print(f"{key}: {value}")

        print(f"Final LoRA adapter and tokenizer saved to: {final_model_dir}")
        print(f"Temporary trainer output dir: {output_dir}")
        print(f"Training report saved to: {report_dir}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        terminal_log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
