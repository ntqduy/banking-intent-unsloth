import argparse
import json
import os
import re
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from time import perf_counter
import yaml


def bootstrap_cuda_device():
    # Respect explicit user configuration if already set
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    parser.add_argument("--mode")
    parser.add_argument("--base_config", default="configs/inference_base.yaml")
    parser.add_argument("--finetuned_config", default="configs/inference.yaml")
    args, _ = parser.parse_known_args()

    config_path = args.config
    if not config_path:
        if args.mode == "base":
            config_path = args.base_config
        elif args.mode == "finetuned":
            config_path = args.finetuned_config
        elif args.mode == "both":
            config_path = args.base_config
        else:
            config_path = args.finetuned_config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        device_id = config.get("device")
        if device_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    except Exception:
        return


bootstrap_cuda_device()

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Disable gradient checkpointing by default for inference/evaluation
os.environ.setdefault("UNSLOTH_SKIP_GRADIENT_CHECKPOINTING", "1")

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
        "Unsloth is required for inference with this checkpoint. Install dependencies "
        "with `pip install -r requirements.txt` on Linux/Colab/Kaggle."
    ) from exc

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


REQUIRED_COLUMNS = ["text", "label"]


def normalize_text(text: str) -> str:
    text = str(text).strip().lower().replace("\n", " ")
    return " ".join(text.split())


def load_yaml_config(config_source):
    if isinstance(config_source, dict):
        return dict(config_source)
    with open(config_source, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json_file(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_optional_json(json_path: str):
    if not json_path or not os.path.exists(json_path):
        return None
    return load_json_file(json_path)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def ensure_clean_data(eval_csv: str | None):
    if not eval_csv:
        return
    if os.path.exists(eval_csv):
        return
    script_path = os.path.join(os.path.dirname(__file__), "preprocess_data.py")
    print(f"[DATA] Missing {eval_csv}. Running preprocess_data.py to generate clean data...")
    subprocess.run([sys.executable, script_path, "--samples_per_label", "0"], check=True)


def append_text(path: str, text: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


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


def compute_classification_metrics(labels, preds):
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


def count_model_parameters(model):
    return int(sum(param.numel() for param in model.parameters()))


def estimate_inference_flops(num_parameters: int, token_count: int):
    if num_parameters <= 0 or token_count <= 0:
        return None
    return float(2 * num_parameters * token_count)


def configure_tokenizer_for_causal_lm(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|PAD_TOKEN|>"
    tokenizer.padding_side = "left"
    return tokenizer


def resolve_cuda_device(device_id):
    if device_id is None or not torch.cuda.is_available():
        return None
    device_id = int(device_id)
    torch.cuda.set_device(device_id)
    return device_id


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


def build_label_list(id2label: dict) -> str:
    return ", ".join(id2label[idx] for idx in sorted(id2label))


def build_prompt(message: str, label_list: str) -> str:
    return (
        "You are an intent classifier for banking customer messages.\n"
        "Choose exactly one intent label from the allowed labels.\n"
        f"Allowed labels: {label_list}\n"
        f"Message: {message}\n"
        "Intent:"
    )


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


def load_eval_split(csv_path: str, id2label: dict) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Evaluation split is missing required columns {missing_columns}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    if "label_name" not in df.columns:
        df["label_name"] = df["label"].map(id2label)
    df["label_name"] = df["label_name"].astype(str)
    return df[["text", "label", "label_name"]]


def resolve_dtype(configured_dtype):
    if configured_dtype in (None, "none", "None"):
        return None
    return getattr(torch, configured_dtype)


def resolve_mapping_path(model_dir: str, configured_path: str | None, filename: str):
    if configured_path:
        return configured_path
    candidate = os.path.join(model_dir, filename)
    if os.path.exists(candidate):
        return candidate
    return None


def load_label_mappings(id2label_path: str | None, label2id_path: str | None):
    id2label_raw = load_optional_json(id2label_path)
    label2id_raw = load_optional_json(label2id_path)

    if id2label_raw is None and label2id_raw is None:
        raise ValueError("Missing label mapping files. Set id2label_path and label2id_path.")

    id2label = {int(k): v for k, v in id2label_raw.items()} if id2label_raw else None
    label2id = {str(k): int(v) for k, v in label2id_raw.items()} if label2id_raw else None

    if id2label is None:
        id2label = {idx: label for label, idx in label2id.items()}
    if label2id is None:
        label2id = {label: idx for idx, label in id2label.items()}

    return id2label, label2id


def infer_model_display_name(model_dir: str):
    normalized = str(model_dir).replace("\\", "/").lower()
    finetune_markers = ("checkpoint", "finetune", "fine-tune", "final_weight", "final_lora", "lora_adapter")
    if any(marker in normalized for marker in finetune_markers):
        return "Qwen2.5-7B", "Fine-tuned"
    return "Qwen2.5-7B", "Base"


def infer_model_key(model_dir: str):
    _, model_type = infer_model_display_name(model_dir)
    return "finetune" if model_type == "Fine-tuned" else "base"


def default_report_dir_for_model(model_dir: str, prefix: str):
    return os.path.join("result", f"{prefix}_{infer_model_key(model_dir)}")


def selected_metrics_record(metrics: dict):
    return {
        "model_source": metrics.get("model_source"),
        "num_samples": metrics.get("num_samples"),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "gflops": (metrics.get("estimated_flops") / 1e9) if metrics.get("estimated_flops") is not None else None,
        "inference_time_seconds": metrics.get("inference_time_seconds"),
        "throughput_samples_per_second": metrics.get("throughput_samples_per_second") or metrics.get("samples_per_second"),
    }


def format_performance_lines(metrics: dict, title: str = "PERFORMANCE"):
    selected = selected_metrics_record(metrics)
    return [
        f"===== {title} =====",
        f"Accuracy:  {selected['accuracy']}",
        f"Precision: {selected['precision']}",
        f"Recall:    {selected['recall']}",
        f"F1:        {selected['f1']}",
        f"GFLOPs:    {selected['gflops']}",
    ]


def print_performance(metrics: dict, title: str = "PERFORMANCE"):
    for line in format_performance_lines(metrics, title):
        print(line)


def predict_batch(classifier, messages):
    prompts = [
        build_prompt(normalize_text(message), classifier.label_list)
        for message in messages
    ]
    started = perf_counter()
    target_device = classifier.target_device or next(classifier.model.parameters()).device
    inputs = classifier.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=classifier.max_seq_length,
    ).to(target_device)

    prompt_lengths = inputs["input_ids"].shape[1]
    generation_max_length = prompt_lengths + classifier.max_new_tokens
    input_token_count = int(inputs["attention_mask"].sum().item())
    with torch.inference_mode():
        output_ids = classifier.model.generate(
            **inputs,
            max_length=generation_max_length,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=classifier.tokenizer.pad_token_id,
        )

    generated_ids = output_ids[:, prompt_lengths:]
    generated_texts = classifier.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    labels = [normalize_generated_label(text, classifier.label2id) for text in generated_texts]
    elapsed_seconds = perf_counter() - started
    generated_token_count = int(generated_ids.numel())
    processed_token_count = input_token_count + generated_token_count
    estimated_flops = estimate_inference_flops(
        classifier.total_parameters,
        processed_token_count,
    )

    return {
        "predicted_ids": [classifier.label2id.get(label, -1) for label in labels],
        "predicted_labels": labels,
        "raw_generations": generated_texts,
        "elapsed_seconds": elapsed_seconds,
        "input_token_count": input_token_count,
        "generated_token_count": generated_token_count,
        "processed_token_count": processed_token_count,
        "estimated_flops": estimated_flops,
        "estimated_tflops": (estimated_flops / 1e12) if estimated_flops is not None else None,
        "generated_tokens_per_second": (generated_token_count / elapsed_seconds) if elapsed_seconds > 0 else None,
    }


class IntentClassification:
    def __init__(self, model_path):
        config = load_yaml_config(model_path)

        # Set CUDA device if specified in config
        device_id = resolve_cuda_device(config.get("device"))
        self.target_device = None
        if device_id is not None and torch.cuda.is_available():
            self.target_device = torch.device(f"cuda:{device_id}")

        # Clear GPU cache before loading model to avoid OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model_dir = config.get("model_name_or_path") or config.get("finetuned_model_name_or_path")
        if not self.model_dir:
            raise ValueError("Config must define model_name_or_path or finetuned_model_name_or_path.")

        self.max_seq_length = int(config.get("max_seq_length", config.get("max_length", 512)))
        self.max_new_tokens = int(config.get("max_new_tokens", 12))
        self.load_in_4bit = bool(config.get("load_in_4bit", True))
        self.dtype = resolve_dtype(config.get("dtype"))
        self.id2label_path = resolve_mapping_path(self.model_dir, config.get("id2label_path"), "id2label.json")
        self.label2id_path = resolve_mapping_path(self.model_dir, config.get("label2id_path"), "label2id.json")

        self.id2label, self.label2id = load_label_mappings(self.id2label_path, self.label2id_path)
        self.label_list = build_label_list(self.id2label)

        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_dir,
                max_seq_length=self.max_seq_length,
                dtype=self.dtype,
                load_in_4bit=self.load_in_4bit,
            )
        except Exception as e:
            if "out of memory" in str(e).lower() or "cuda error" in str(e).lower():
                raise RuntimeError(
                    f"GPU out of memory when loading model. Try one of:\n"
                    f"  1. Set 'load_in_4bit: false' in config\n"
                    f"  2. Reduce 'batch_size' in config\n"
                    f"  3. Reduce 'max_seq_length' in config\n"
                    f"  4. Set 'use_gradient_checkpointing: false' in config\n"
                    f"  5. Use different GPU: --device 1 (if available)\n"
                    f"Original error: {str(e)}"
                ) from e
            raise
        
        # Force model onto the configured device to avoid mixed-device tensors
        if self.target_device is not None:
            self.model = self.model.to(self.target_device)

        # Disable gradient checkpointing if specified in config
        use_gradient_checkpointing = config.get("use_gradient_checkpointing", True)
        if not use_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        
        self.model = normalize_model_configs(self.model)
        self.tokenizer = configure_tokenizer_for_causal_lm(self.tokenizer)
        FastLanguageModel.for_inference(self.model)
        self.total_parameters = count_model_parameters(self.model)
        self.device_id = device_id

    def __call__(self, message):
        result = predict_batch(self, [message])
        return result["predicted_labels"][0]


def evaluate_and_save_report(classifier, eval_df, report_dir: str, config: dict):
    report_dir = ensure_dir(report_dir)
    batch_size = int(config.get("batch_size", 8))
    num_correct_samples = int(config.get("num_correct_samples", 5))
    num_wrong_samples = int(config.get("num_wrong_samples", 5))

    predicted_ids = []
    predicted_labels = []
    raw_generations = []
    total_runtime_seconds = 0.0
    total_input_tokens = 0
    total_generated_tokens = 0
    total_processed_tokens = 0
    total_estimated_flops = 0.0
    started_at = datetime.now(timezone.utc)

    for start_idx in range(0, len(eval_df), batch_size):
        batch_df = eval_df.iloc[start_idx : start_idx + batch_size]
        batch_result = predict_batch(classifier, batch_df["text"].tolist())
        predicted_ids.extend(batch_result["predicted_ids"])
        predicted_labels.extend(batch_result["predicted_labels"])
        raw_generations.extend(batch_result["raw_generations"])
        total_runtime_seconds += batch_result["elapsed_seconds"]
        total_input_tokens += int(batch_result["input_token_count"])
        total_generated_tokens += int(batch_result["generated_token_count"])
        total_processed_tokens += int(batch_result["processed_token_count"])
        total_estimated_flops += float(batch_result["estimated_flops"] or 0.0)

    finished_at = datetime.now(timezone.utc)
    predictions_df = pd.DataFrame(
        {
            "sample_id": np.arange(len(eval_df)),
            "sample": eval_df["text"].tolist(),
            "ground_truth_id": eval_df["label"].tolist(),
            "ground_truth_label": eval_df["label_name"].tolist(),
            "predicted_id": predicted_ids,
            "predicted_label": predicted_labels,
            "raw_generation": raw_generations,
        }
    )
    predictions_df["correct"] = predictions_df["ground_truth_id"] == predictions_df["predicted_id"]

    metrics = compute_classification_metrics(
        predictions_df["ground_truth_id"].tolist(),
        predictions_df["predicted_id"].tolist(),
    )
    num_samples = int(len(predictions_df))
    metrics_record = {
        "model_source": classifier.model_dir,
        "num_samples": num_samples,
        "num_correct": int(predictions_df["correct"].sum()),
        "num_incorrect": int((~predictions_df["correct"]).sum()),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "inference_time_seconds": total_runtime_seconds,
        "total_runtime_seconds": total_runtime_seconds,
        "average_inference_time_ms_per_sample": (total_runtime_seconds / num_samples * 1000.0) if num_samples else None,
        "average_latency_ms_per_sample": (total_runtime_seconds / num_samples * 1000.0) if num_samples else None,
        "fps": (num_samples / total_runtime_seconds) if total_runtime_seconds > 0 else None,
        "throughput_samples_per_second": (num_samples / total_runtime_seconds) if total_runtime_seconds > 0 else None,
        "model_parameter_count": classifier.total_parameters,
        "input_token_count": total_input_tokens,
        "generated_token_count": total_generated_tokens,
        "processed_token_count": total_processed_tokens,
        "estimated_flops": total_estimated_flops,
        "estimated_tflops": total_estimated_flops / 1e12,
        "estimated_flops_per_second": (total_estimated_flops / total_runtime_seconds) if total_runtime_seconds > 0 else None,
        "estimated_tflops_per_second": (total_estimated_flops / total_runtime_seconds / 1e12) if total_runtime_seconds > 0 else None,
        "generated_tokens_per_second": (total_generated_tokens / total_runtime_seconds) if total_runtime_seconds > 0 else None,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "eval_csv": config.get("eval_csv"),
        "batch_size": batch_size,
        "max_seq_length": classifier.max_seq_length,
    }

    correct_samples_df = predictions_df[predictions_df["correct"]].head(num_correct_samples)
    wrong_samples_df = predictions_df[~predictions_df["correct"]].head(num_wrong_samples)

    metrics_csv_path = os.path.join(report_dir, "metric.csv")
    legacy_metrics_csv_path = os.path.join(report_dir, "metrics.csv")
    predictions_csv_path = os.path.join(report_dir, "predictions.csv")
    correct_samples_path = os.path.join(report_dir, "correct_samples.csv")
    wrong_samples_path = os.path.join(report_dir, "wrong_samples.csv")
    sample_predictions_path = os.path.join(report_dir, "sample_predictions.csv")
    inf_test_path = os.path.join(report_dir, "inf_test.txt")
    eval_log_path = os.path.join(report_dir, "eval_full_pipeline.txt")
    summary_json_path = os.path.join(report_dir, "summary.json")

    metrics_df = pd.DataFrame([to_serializable(metrics_record)])
    metrics_df.to_csv(metrics_csv_path, index=False)
    metrics_df.to_csv(legacy_metrics_csv_path, index=False)
    predictions_df.to_csv(predictions_csv_path, index=False)
    correct_samples_df.to_csv(correct_samples_path, index=False)
    wrong_samples_df.to_csv(wrong_samples_path, index=False)
    pd.concat([correct_samples_df, wrong_samples_df], ignore_index=True).to_csv(sample_predictions_path, index=False)

    inf_lines = []
    for row in predictions_df.itertuples(index=False):
        status = "TRUE" if row.correct else "FALSE"
        inf_lines.append(
            f"[{status}] predict={row.predicted_label} | GT={row.ground_truth_label} | message={row.sample}"
        )
    write_text(inf_test_path, "\n".join(inf_lines))

    eval_lines = [
        f"model_source: {classifier.model_dir}",
        f"eval_csv: {config.get('eval_csv')}",
        f"num_samples: {num_samples}",
        f"num_correct: {metrics_record['num_correct']}",
        f"num_incorrect: {metrics_record['num_incorrect']}",
        "",
        *format_performance_lines(metrics_record, "PERFORMANCE"),
        "",
        f"metric_csv: {metrics_csv_path}",
        f"predictions_csv: {predictions_csv_path}",
        f"inf_test_txt: {inf_test_path}",
    ]
    write_text(eval_log_path, "\n".join(eval_lines))

    summary = {
        "report_dir": report_dir,
        "metrics": metrics_record,
        "artifacts": {
            "metrics_csv": metrics_csv_path,
            "metric_csv": metrics_csv_path,
            "legacy_metrics_csv": legacy_metrics_csv_path,
            "predictions_csv": predictions_csv_path,
            "correct_samples_csv": correct_samples_path,
            "wrong_samples_csv": wrong_samples_path,
            "sample_predictions_csv": sample_predictions_path,
            "inf_test_txt": inf_test_path,
            "eval_full_pipeline_txt": eval_log_path,
            "summary_json": summary_json_path,
        },
    }
    write_json(summary_json_path, summary)
    return summary, correct_samples_df, wrong_samples_df


def compare_and_save_reports(base_summary: dict, finetune_summary: dict, report_dir: str):
    report_dir = ensure_dir(report_dir)
    base_metrics = selected_metrics_record(base_summary["metrics"])
    finetune_metrics = selected_metrics_record(finetune_summary["metrics"])

    rows = []
    for model_name, metrics in (("Base Qwen2.5-7B", base_metrics), ("Fine-tuned Unsloth QLoRA model", finetune_metrics)):
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)

    metric_csv_path = os.path.join(report_dir, "metric.csv")
    result_txt_path = os.path.join(report_dir, "result_base_finetune.txt")
    pd.DataFrame(rows).to_csv(metric_csv_path, index=False)

    lines = ["===== BASE VS FINE-TUNED COMPARISON ====="]
    for row in rows:
        lines.extend(
            [
                "",
                f"Model: {row['model']}",
                f"Accuracy:  {row.get('accuracy')}",
                f"Precision: {row.get('precision')}",
                f"Recall:    {row.get('recall')}",
                f"F1:        {row.get('f1')}",
                f"GFLOPs:    {row.get('gflops')}",
                f"Inference time seconds: {row.get('inference_time_seconds')}",
            ]
        )
    write_text(result_txt_path, "\n".join(lines))
    return {"metric_csv": metric_csv_path, "result_txt": result_txt_path, "rows": rows}


def write_both_inference_report(summaries: list[dict], report_dir: str):
    report_dir = ensure_dir(report_dir)
    report_path = os.path.join(report_dir, "inf_base_finetune.txt")
    rows = []
    for title, summary in (
        ("===== Base Model =====", summaries[0]),
        ("===== Fine-tuned Model =====", summaries[1]),
    ):
        metrics = summary["metrics"]
        rows.extend(
            [
                title,
                f"Accuracy:  {metrics.get('accuracy')}",
                f"Precision: {metrics.get('precision')}",
                f"Recall:    {metrics.get('recall')}",
                f"F1:        {metrics.get('f1')}",
                "",
            ]
        )
        inf_path = summary["artifacts"]["inf_test_txt"]
        if os.path.exists(inf_path):
            with open(inf_path, "r", encoding="utf-8") as f:
                rows.append(f.read().rstrip())
        rows.append("")
    write_text(report_path, "\n".join(rows))
    return report_path


def save_single_prediction(classifier, message: str, prediction_result: dict, report_dir: str):
    report_dir = ensure_dir(report_dir)
    prediction_path = os.path.join(report_dir, "single_prediction.json")
    prediction_txt_path = os.path.join(report_dir, "single_prediction.txt")
    record = {
        "model_source": classifier.model_dir,
        "message": message,
        "predicted_id": prediction_result["predicted_ids"][0],
        "predicted_label": prediction_result["predicted_labels"][0],
        "raw_generation": prediction_result["raw_generations"][0],
        "inference_time_seconds": prediction_result["elapsed_seconds"],
        "elapsed_seconds": prediction_result["elapsed_seconds"],
        "input_token_count": prediction_result["input_token_count"],
        "generated_token_count": prediction_result["generated_token_count"],
        "processed_token_count": prediction_result["processed_token_count"],
        "estimated_flops": prediction_result["estimated_flops"],
        "estimated_tflops": prediction_result["estimated_tflops"],
        "generated_tokens_per_second": prediction_result["generated_tokens_per_second"],
        "created_at_utc": datetime.now(timezone.utc),
    }
    write_json(prediction_path, record)
    write_text(
        prediction_txt_path,
        "\n".join(
            [
                f"model_source: {classifier.model_dir}",
                f"message: {message}",
                f"predicted_label: {prediction_result['predicted_labels'][0]}",
                f"predicted_id: {prediction_result['predicted_ids'][0]}",
                f"inference_time_seconds: {prediction_result['elapsed_seconds']}",
            ]
        ),
    )
    return prediction_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--mode", choices=["finetuned", "base", "both"], default=None)
    parser.add_argument("--base_config", type=str, default="configs/inference_base.yaml")
    parser.add_argument("--finetuned_config", type=str, default="configs/inference.yaml")
    parser.add_argument("--message", type=str)
    parser.add_argument("--eval_csv", type=str)
    parser.add_argument("--report_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()

    # Print device info
    config_to_check = args.base_config if args.mode != "finetuned" else args.finetuned_config
    config_preview = load_yaml_config(config_to_check)
    device_id = config_preview.get("device")
    if device_id is not None:
        print(f"[INFERENCE] Using GPU device: {device_id}")
    else:
        print("[INFERENCE] Using default GPU device")

    if args.mode == "both":
        base_config = load_yaml_config(args.base_config)
        finetune_config = load_yaml_config(args.finetuned_config)
        for config in (base_config, finetune_config):
            if args.eval_csv:
                config["eval_csv"] = args.eval_csv
            if args.batch_size is not None:
                config["batch_size"] = args.batch_size

        if args.message:
            for title, config in (
                ("===== Base Model =====", base_config),
                ("===== Fine-tuned Model =====", finetune_config),
            ):
                classifier = IntentClassification(config)
                prediction_result = predict_batch(classifier, [args.message])
                prediction = prediction_result["predicted_labels"][0]
                model_name, model_type = infer_model_display_name(classifier.model_dir)
                report_dir = ensure_dir(default_report_dir_for_model(classifier.model_dir, "inf"))
                save_single_prediction(classifier, args.message, prediction_result, report_dir)
                print(title)
                print(f"Model: {model_name} ({model_type})")
                print(f"message: {args.message}")
                print(f"predict: {prediction}")
            return

        eval_csv = base_config.get("eval_csv") or finetune_config.get("eval_csv")
        if not eval_csv:
            parser.error("Set eval_csv in configs or pass --eval_csv for --mode both.")

        ensure_clean_data(eval_csv)

        summaries = []
        for title, config in (
            ("===== Base Model =====", base_config),
            ("===== Fine-tuned Model =====", finetune_config),
        ):
            classifier = IntentClassification(config)
            report_dir = ensure_dir(default_report_dir_for_model(classifier.model_dir, "inf"))
            eval_df = load_eval_split(config.get("eval_csv", eval_csv), classifier.id2label)
            summary, _, _ = evaluate_and_save_report(classifier, eval_df, report_dir, config)
            model_name, model_type = infer_model_display_name(classifier.model_dir)
            print(title)
            print(f"Model: {model_name} ({model_type})")
            print_performance(summary["metrics"], f"{model_name} ({model_type}) PERFORMANCE")
            print(f"Accuracy: {summary['metrics'].get('accuracy')}")
            print(f"Precision: {summary['metrics'].get('precision')}")
            print(f"Recall: {summary['metrics'].get('recall')}")
            print(f"F1: {summary['metrics'].get('f1')}")
            print(f"Prediction log: {summary['artifacts']['inf_test_txt']}")
            with open(summary["artifacts"]["inf_test_txt"], "r", encoding="utf-8") as f:
                print(f.read().rstrip())
            summaries.append(summary)

        both_report_dir = ensure_dir(os.path.join("result", "inf_both"))
        comparison = compare_and_save_reports(
            base_summary=summaries[0],
            finetune_summary=summaries[1],
            report_dir=both_report_dir,
        )
        compare_csv_path = os.path.join(both_report_dir, "compare_base_finetune.csv")
        pd.DataFrame(comparison["rows"]).to_csv(compare_csv_path, index=False)
        both_report_path = write_both_inference_report(summaries, both_report_dir)
        print("Comparison result:", both_report_path)
        print("Comparison metric CSV:", compare_csv_path)
        return

    if args.mode == "base":
        args.config = args.config or args.base_config
    elif args.mode == "finetuned":
        args.config = args.config or args.finetuned_config
    if not args.config:
        args.config = args.finetuned_config

    config = load_yaml_config(args.config)
    if args.eval_csv:
        config["eval_csv"] = args.eval_csv
    if args.report_dir:
        config["report_dir"] = args.report_dir
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    classifier = IntentClassification(config)

    eval_csv = config.get("eval_csv")
    if not args.message and not eval_csv:
        parser.error("Provide --message for one prediction or set eval_csv for evaluation.")

    if args.message:
        prediction_result = predict_batch(classifier, [args.message])
        prediction = prediction_result["predicted_labels"][0]
        model_name, model_type = infer_model_display_name(classifier.model_dir)
        print(f"===== {model_name} ({model_type}) =====")
        print(f"message: {args.message}")
        print(f"predict: {prediction}")
        report_dir = ensure_dir(config.get("report_dir", default_report_dir_for_model(classifier.model_dir, "inf")))
        prediction_path = save_single_prediction(classifier, args.message, prediction_result, report_dir)
        print("Single prediction saved to:", prediction_path)

    elif eval_csv:
        ensure_clean_data(eval_csv)
        report_dir = ensure_dir(config.get("report_dir", default_report_dir_for_model(classifier.model_dir, "inf")))
        eval_df = load_eval_split(eval_csv, classifier.id2label)
        summary, correct_samples_df, wrong_samples_df = evaluate_and_save_report(
            classifier=classifier,
            eval_df=eval_df,
            report_dir=report_dir,
            config=config,
        )

        model_name, model_type = infer_model_display_name(classifier.model_dir)
        print(f"===== {model_name} ({model_type}) =====")
        print_performance(summary["metrics"])
        print(f"Accuracy: {summary['metrics'].get('accuracy')}")
        print(f"Precision: {summary['metrics'].get('precision')}")
        print(f"Recall: {summary['metrics'].get('recall')}")
        print(f"F1: {summary['metrics'].get('f1')}")
        print("Metric CSV:", summary["artifacts"]["metric_csv"])
        print("Predictions CSV:", summary["artifacts"]["predictions_csv"])
        print("Inference test TXT:", summary["artifacts"]["inf_test_txt"])
        with open(summary["artifacts"]["inf_test_txt"], "r", encoding="utf-8") as f:
            print(f.read().rstrip())

        if not correct_samples_df.empty:
            print("Correct samples preview:")
            for _, row in correct_samples_df.iterrows():
                print(f"- {row['sample']} | GT={row['ground_truth_label']} | Pred={row['predicted_label']}")

        if not wrong_samples_df.empty:
            print("Wrong samples preview:")
            for _, row in wrong_samples_df.iterrows():
                print(f"- {row['sample']} | GT={row['ground_truth_label']} | Pred={row['predicted_label']}")


if __name__ == "__main__":
    main()
