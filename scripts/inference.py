import argparse
import json
import os
import re
from datetime import datetime, timezone
from time import perf_counter

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
import yaml
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


def predict_batch(classifier, messages):
    prompts = [
        build_prompt(normalize_text(message), classifier.label_list)
        for message in messages
    ]
    started = perf_counter()
    inputs = classifier.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=classifier.max_seq_length,
    ).to(next(classifier.model.parameters()).device)

    prompt_lengths = inputs["input_ids"].shape[1]
    input_token_count = int(inputs["attention_mask"].sum().item())
    with torch.inference_mode():
        output_ids = classifier.model.generate(
            **inputs,
            max_new_tokens=classifier.max_new_tokens,
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

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_dir,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )
        self.tokenizer = configure_tokenizer_for_causal_lm(self.tokenizer)
        FastLanguageModel.for_inference(self.model)
        self.total_parameters = count_model_parameters(self.model)

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

    metrics_csv_path = os.path.join(report_dir, "metrics.csv")
    predictions_csv_path = os.path.join(report_dir, "predictions.csv")
    correct_samples_path = os.path.join(report_dir, "correct_samples.csv")
    wrong_samples_path = os.path.join(report_dir, "wrong_samples.csv")
    sample_predictions_path = os.path.join(report_dir, "sample_predictions.csv")
    summary_json_path = os.path.join(report_dir, "summary.json")

    pd.DataFrame([to_serializable(metrics_record)]).to_csv(metrics_csv_path, index=False)
    predictions_df.to_csv(predictions_csv_path, index=False)
    correct_samples_df.to_csv(correct_samples_path, index=False)
    wrong_samples_df.to_csv(wrong_samples_path, index=False)
    pd.concat([correct_samples_df, wrong_samples_df], ignore_index=True).to_csv(sample_predictions_path, index=False)

    summary = {
        "report_dir": report_dir,
        "metrics": metrics_record,
        "artifacts": {
            "metrics_csv": metrics_csv_path,
            "predictions_csv": predictions_csv_path,
            "correct_samples_csv": correct_samples_path,
            "wrong_samples_csv": wrong_samples_path,
            "sample_predictions_csv": sample_predictions_path,
            "summary_json": summary_json_path,
        },
    }
    write_json(summary_json_path, summary)
    return summary, correct_samples_df, wrong_samples_df


def save_single_prediction(classifier, message: str, prediction_result: dict, report_dir: str):
    report_dir = ensure_dir(report_dir)
    prediction_path = os.path.join(report_dir, "single_prediction.json")
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
    return prediction_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--message", type=str)
    parser.add_argument("--eval_csv", type=str)
    parser.add_argument("--report_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()

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
        print("Model source:", classifier.model_dir)
        print("Input message:", args.message)
        print("Predicted label:", prediction)
        report_dir = ensure_dir(config.get("report_dir", "outputs/outputs_inf_finetune"))
        prediction_path = save_single_prediction(classifier, args.message, prediction_result, report_dir)
        print("Single prediction saved to:", prediction_path)

    elif eval_csv:
        report_dir = ensure_dir(config.get("report_dir", "outputs/outputs_inf_finetune"))
        eval_df = load_eval_split(eval_csv, classifier.id2label)
        summary, correct_samples_df, wrong_samples_df = evaluate_and_save_report(
            classifier=classifier,
            eval_df=eval_df,
            report_dir=report_dir,
            config=config,
        )

        print("===== EVALUATION SUMMARY =====")
        for key, value in summary["metrics"].items():
            print(f"{key}: {value}")
        print("Metrics CSV:", summary["artifacts"]["metrics_csv"])
        print("Predictions CSV:", summary["artifacts"]["predictions_csv"])

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
