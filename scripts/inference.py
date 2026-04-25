import argparse
import json
import os
from datetime import datetime, timezone
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REQUIRED_COLUMNS = ["text", "label"]


def normalize_text(text: str) -> str:
    text = str(text).strip().lower().replace("\n", " ")
    text = " ".join(text.split())
    return text


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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


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


class IntentClassification:
    def __init__(self, model_path):
        config = load_yaml_config(model_path)

        self.requested_model_variant = str(config.get("model_variant", "auto")).lower()
        self.base_model_dir = config.get("base_model_name_or_path")
        self.finetuned_model_dir = config.get("finetuned_model_name_or_path")
        self.model_dir, self.model_variant = self._resolve_model_dir(config)
        self.tokenizer_dir = self._resolve_tokenizer_dir(config)
        self.id2label_path = self._resolve_mapping_path(config.get("id2label_path"), "id2label.json")
        self.label2id_path = self._resolve_mapping_path(config.get("label2id_path"), "label2id.json")
        self.max_length = int(config.get("max_length", 64))
        self.device = self._resolve_device(config.get("device"))

        self.id2label, self.label2id = self._load_label_mappings()
        self.tokenizer, self.tokenizer_source = self._load_tokenizer()
        self.model, self.model_source = self._load_model()

        if not self.id2label:
            self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

        if not self.label2id:
            self.label2id = {str(k): int(v) for k, v in self.model.config.label2id.items()}

        self.model.to(self.device)
        self.model.eval()

    def _resolve_model_dir(self, config):
        if config.get("model_name_or_path"):
            return config["model_name_or_path"], self.requested_model_variant

        if config.get("model_dir"):
            return config["model_dir"], self.requested_model_variant

        if self.requested_model_variant == "base" and self.base_model_dir:
            return self.base_model_dir, "base"

        if self.requested_model_variant == "finetuned" and self.finetuned_model_dir:
            return self.finetuned_model_dir, "finetuned"

        if self.finetuned_model_dir:
            return self.finetuned_model_dir, "finetuned"

        if self.base_model_dir:
            return self.base_model_dir, "base"

        raise ValueError("Missing model path. Set model_name_or_path or base/finetuned_model_name_or_path in the inference config.")

    def _resolve_tokenizer_dir(self, config):
        if config.get("tokenizer_name_or_path"):
            return config["tokenizer_name_or_path"]

        if self.model_variant == "base" and config.get("base_tokenizer_name_or_path"):
            return config["base_tokenizer_name_or_path"]

        if self.model_variant == "finetuned" and config.get("finetuned_tokenizer_name_or_path"):
            return config["finetuned_tokenizer_name_or_path"]

        if config.get("finetuned_tokenizer_name_or_path") and self.model_variant != "base":
            return config["finetuned_tokenizer_name_or_path"]

        if config.get("base_tokenizer_name_or_path"):
            return config["base_tokenizer_name_or_path"]

        return self.model_dir

    def _resolve_mapping_path(self, configured_path, filename):
        if configured_path:
            return configured_path

        if os.path.isdir(self.model_dir):
            candidate = os.path.join(self.model_dir, filename)
            if os.path.exists(candidate):
                return candidate

        return None

    def _resolve_device(self, configured_device):
        if configured_device:
            return torch.device(configured_device)

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_label_mappings(self):
        id2label_raw = load_optional_json(self.id2label_path)
        label2id_raw = load_optional_json(self.label2id_path)

        id2label = None
        if id2label_raw is not None:
            id2label = {int(k): v for k, v in id2label_raw.items()}

        label2id = None
        if label2id_raw is not None:
            label2id = {str(k): int(v) for k, v in label2id_raw.items()}

        if id2label is None and label2id is not None:
            id2label = {idx: label for label, idx in label2id.items()}

        if label2id is None and id2label is not None:
            label2id = {label: idx for idx, label in id2label.items()}

        return id2label, label2id

    def _load_tokenizer(self):
        tokenizer_candidates = [self.tokenizer_dir]
        if self.model_variant == "finetuned":
            tokenizer_candidates.append(self.model_dir)
            tokenizer_candidates.append(self.base_model_dir)

        seen = set()
        last_error = None

        for candidate in tokenizer_candidates:
            if not candidate or candidate in seen:
                continue

            seen.add(candidate)

            try:
                return AutoTokenizer.from_pretrained(candidate), candidate
            except OSError as exc:
                last_error = exc

        if last_error is not None:
            raise last_error

        raise OSError("Unable to load tokenizer. Set tokenizer_name_or_path in the inference config or CLI arguments.")

    def _load_model(self):
        model_kwargs = {}
        if self.id2label and self.label2id:
            model_kwargs.update(
                num_labels=len(self.id2label),
                label2id=self.label2id,
                id2label=self.id2label,
            )

        try:
            model = AutoModelForSequenceClassification.from_pretrained(self.model_dir, **model_kwargs)
            return model, self.model_dir
        except RuntimeError as exc:
            if "size mismatch" not in str(exc):
                raise

            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir,
                ignore_mismatched_sizes=True,
                **model_kwargs,
            )
            return model, self.model_dir

    def _estimate_flops(self, tokenized_inputs):
        if not hasattr(self.model, "floating_point_ops"):
            return None

        try:
            flops = self.model.floating_point_ops(tokenized_inputs)
        except Exception:
            return None

        if isinstance(flops, torch.Tensor):
            flops = flops.item()

        return int(flops)

    def predict_batch(self, messages):
        normalized_messages = [normalize_text(message) for message in messages]

        batch_start = perf_counter()
        tokenized_inputs = self.tokenizer(
            normalized_messages,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        approx_flops = self._estimate_flops(tokenized_inputs)
        model_inputs = {k: v.to(self.device) for k, v in tokenized_inputs.items()}

        with torch.no_grad():
            outputs = self.model(**model_inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_ids = torch.argmax(probabilities, dim=-1)
            confidence_scores = torch.gather(probabilities, 1, predicted_ids.unsqueeze(-1)).squeeze(-1)

        elapsed_seconds = perf_counter() - batch_start

        return {
            "predicted_ids": [int(x) for x in predicted_ids.detach().cpu().tolist()],
            "predicted_labels": [self.id2label[int(x)] for x in predicted_ids.detach().cpu().tolist()],
            "confidence_scores": [float(x) for x in confidence_scores.detach().cpu().tolist()],
            "elapsed_seconds": elapsed_seconds,
            "approx_flops": approx_flops,
        }

    def __call__(self, message):
        result = self.predict_batch([message])
        return result["predicted_labels"][0]


def evaluate_and_save_report(classifier, eval_df, report_dir: str, config: dict):
    report_dir = ensure_dir(report_dir)
    batch_size = int(config.get("batch_size", 16))
    num_correct_samples = int(config.get("num_correct_samples", 5))
    num_wrong_samples = int(config.get("num_wrong_samples", 5))

    predicted_ids = []
    predicted_labels = []
    confidence_scores = []
    total_runtime_seconds = 0.0
    total_flops = 0
    flops_available = False
    started_at = datetime.now(timezone.utc)

    for start_idx in range(0, len(eval_df), batch_size):
        batch_df = eval_df.iloc[start_idx : start_idx + batch_size]
        batch_result = classifier.predict_batch(batch_df["text"].tolist())

        predicted_ids.extend(batch_result["predicted_ids"])
        predicted_labels.extend(batch_result["predicted_labels"])
        confidence_scores.extend(batch_result["confidence_scores"])
        total_runtime_seconds += batch_result["elapsed_seconds"]

        if batch_result["approx_flops"] is not None:
            total_flops += int(batch_result["approx_flops"])
            flops_available = True

    finished_at = datetime.now(timezone.utc)

    predictions_df = pd.DataFrame(
        {
            "sample_id": np.arange(len(eval_df)),
            "sample": eval_df["text"].tolist(),
            "ground_truth_id": eval_df["label"].tolist(),
            "ground_truth_label": eval_df["label_name"].tolist(),
            "predicted_id": predicted_ids,
            "predicted_label": predicted_labels,
            "confidence": confidence_scores,
        }
    )
    predictions_df["correct"] = predictions_df["ground_truth_id"] == predictions_df["predicted_id"]

    metrics = compute_classification_metrics(
        predictions_df["ground_truth_id"].tolist(),
        predictions_df["predicted_id"].tolist(),
    )

    num_samples = int(len(predictions_df))
    num_correct = int(predictions_df["correct"].sum())
    num_wrong = int(num_samples - num_correct)
    avg_latency_ms = (total_runtime_seconds / num_samples * 1000.0) if num_samples else None
    throughput = (num_samples / total_runtime_seconds) if total_runtime_seconds > 0 else None
    approx_total_flops = int(total_flops) if flops_available else None
    approx_flops_per_sample = int(total_flops / num_samples) if flops_available and num_samples else None

    metrics_record = {
        "model_variant": classifier.model_variant,
        "model_source": classifier.model_source,
        "tokenizer_source": classifier.tokenizer_source,
        "num_samples": num_samples,
        "num_correct": num_correct,
        "num_incorrect": num_wrong,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "total_runtime_seconds": total_runtime_seconds,
        "average_latency_ms_per_sample": avg_latency_ms,
        "throughput_samples_per_second": throughput,
        "approx_total_flops": approx_total_flops,
        "approx_flops_per_sample": approx_flops_per_sample,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "eval_csv": config.get("eval_csv"),
        "batch_size": batch_size,
        "max_length": classifier.max_length,
    }

    correct_samples_df = predictions_df[predictions_df["correct"]].sort_values(
        by=["confidence", "sample_id"],
        ascending=[False, True],
    ).head(num_correct_samples)

    wrong_samples_df = predictions_df[~predictions_df["correct"]].sort_values(
        by=["confidence", "sample_id"],
        ascending=[False, True],
    ).head(num_wrong_samples)

    metrics_csv_path = os.path.join(report_dir, "metrics.csv")
    predictions_csv_path = os.path.join(report_dir, "predictions.csv")
    correct_samples_path = os.path.join(report_dir, "correct_samples.csv")
    wrong_samples_path = os.path.join(report_dir, "wrong_samples.csv")
    summary_json_path = os.path.join(report_dir, "summary.json")
    metrics_json_path = os.path.join(report_dir, "metrics.json")

    pd.DataFrame([to_serializable(metrics_record)]).to_csv(metrics_csv_path, index=False)
    predictions_df.to_csv(predictions_csv_path, index=False)
    correct_samples_df.to_csv(correct_samples_path, index=False)
    wrong_samples_df.to_csv(wrong_samples_path, index=False)

    summary = {
        "report_dir": report_dir,
        "metrics": metrics_record,
        "artifacts": {
            "metrics_csv": metrics_csv_path,
            "metrics_json": metrics_json_path,
            "predictions_csv": predictions_csv_path,
            "correct_samples_csv": correct_samples_path,
            "wrong_samples_csv": wrong_samples_path,
        },
    }

    write_json(metrics_json_path, metrics_record)
    write_json(summary_json_path, summary)

    return summary, correct_samples_df, wrong_samples_df


def save_single_prediction(report_dir: str, message: str, result: dict):
    single_prediction_path = os.path.join(report_dir, "single_prediction.json")
    payload = {
        "message": message,
        "prediction": result,
    }
    write_json(single_prediction_path, payload)
    return single_prediction_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--message", type=str)
    parser.add_argument("--model_variant", type=str, choices=["auto", "base", "finetuned"])
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--tokenizer_name_or_path", type=str)
    parser.add_argument("--id2label_path", type=str)
    parser.add_argument("--label2id_path", type=str)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--eval_csv", type=str)
    parser.add_argument("--report_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_correct_samples", type=int)
    parser.add_argument("--num_wrong_samples", type=int)
    args = parser.parse_args()

    config = load_yaml_config(args.config)

    if args.model_variant:
        config["model_variant"] = args.model_variant
    if args.model_name_or_path:
        config["model_name_or_path"] = args.model_name_or_path
    if args.tokenizer_name_or_path:
        config["tokenizer_name_or_path"] = args.tokenizer_name_or_path
    if args.id2label_path:
        config["id2label_path"] = args.id2label_path
    if args.label2id_path:
        config["label2id_path"] = args.label2id_path
    if args.max_length is not None:
        config["max_length"] = args.max_length
    if args.device:
        config["device"] = args.device
    if args.eval_csv:
        config["eval_csv"] = args.eval_csv
    if args.report_dir:
        config["report_dir"] = args.report_dir
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_correct_samples is not None:
        config["num_correct_samples"] = args.num_correct_samples
    if args.num_wrong_samples is not None:
        config["num_wrong_samples"] = args.num_wrong_samples

    classifier = IntentClassification(config)

    report_dir = config.get("report_dir")
    if report_dir:
        report_dir = ensure_dir(report_dir)

    eval_csv = config.get("eval_csv")
    if not args.message and not eval_csv:
        parser.error("Provide --message for single prediction or set --eval_csv / eval_csv in the config for evaluation.")

    if args.message:
        single_result = classifier.predict_batch([args.message])
        prediction = {
            "model_variant": classifier.model_variant,
            "model_source": classifier.model_source,
            "tokenizer_source": classifier.tokenizer_source,
            "predicted_id": single_result["predicted_ids"][0],
            "predicted_label": single_result["predicted_labels"][0],
            "confidence": single_result["confidence_scores"][0],
            "elapsed_seconds": single_result["elapsed_seconds"],
            "approx_flops": single_result["approx_flops"],
        }

        print("Model variant:", classifier.model_variant)
        print("Model source:", classifier.model_source)
        print("Tokenizer source:", classifier.tokenizer_source)
        print("Input message:", args.message)
        print("Predicted label:", prediction["predicted_label"])
        print("Confidence:", prediction["confidence"])

        if report_dir:
            single_prediction_path = save_single_prediction(report_dir, args.message, prediction)
            print("Single prediction saved to:", single_prediction_path)

    if eval_csv:
        eval_df = load_eval_split(eval_csv, classifier.id2label)
        if report_dir is None:
            report_dir = ensure_dir(os.path.join("outputs", f"inference_{classifier.model_variant}"))

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
        print("Correct samples CSV:", summary["artifacts"]["correct_samples_csv"])
        print("Wrong samples CSV:", summary["artifacts"]["wrong_samples_csv"])

        if not correct_samples_df.empty:
            print("Top correct samples preview:")
            for _, row in correct_samples_df.iterrows():
                print(
                    f"- sample={row['sample']} | GT={row['ground_truth_label']} | Predict={row['predicted_label']} | confidence={row['confidence']:.4f}"
                )

        if not wrong_samples_df.empty:
            print("Top wrong samples preview:")
            for _, row in wrong_samples_df.iterrows():
                print(
                    f"- sample={row['sample']} | GT={row['ground_truth_label']} | Predict={row['predicted_label']} | confidence={row['confidence']:.4f}"
                )


if __name__ == "__main__":
    main()
