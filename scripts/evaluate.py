import argparse
import os
import pandas as pd

# Environment setup for inference/evaluation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "")
os.environ.setdefault("UNSLOTH_SKIP_GRADIENT_CHECKPOINTING", "1")


def apply_overrides(config: dict, args):
    if args.eval_csv:
        config["eval_csv"] = args.eval_csv
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.device is not None:
        config["device"] = args.device
    return config


def evaluate_one(config: dict, report_dir: str, header_title: str | None = None):
    from inference import (
        IntentClassification,
        ensure_dir,
        evaluate_and_save_report,
        infer_model_display_name,
        load_eval_split,
        print_performance,
    )

    eval_csv = config.get("eval_csv")
    if not eval_csv:
        raise ValueError("Set eval_csv in the config or pass --eval_csv.")

    classifier = IntentClassification(config)
    report_dir = ensure_dir(report_dir)
    eval_df = load_eval_split(eval_csv, classifier.id2label)
    summary, correct_samples_df, wrong_samples_df = evaluate_and_save_report(
        classifier=classifier,
        eval_df=eval_df,
        report_dir=report_dir,
        config=config,
    )

    model_name, model_type = infer_model_display_name(classifier.model_dir)
    if header_title:
        print(header_title)
        print(f"Model: {model_name} ({model_type})")
    else:
        print(f"===== {model_name} ({model_type}) =====")
    print_performance(summary["metrics"], f"{model_name} ({model_type}) PERFORMANCE")
    print(f"Accuracy: {summary['metrics'].get('accuracy')}")
    print(f"Precision: {summary['metrics'].get('precision')}")
    print(f"Recall: {summary['metrics'].get('recall')}")
    print(f"F1: {summary['metrics'].get('f1')}")
    print("Evaluation log:", summary["artifacts"]["eval_full_pipeline_txt"])
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

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--mode", choices=["finetuned", "base", "both"], default=None)
    parser.add_argument("--base_config", type=str, default="configs/inference_base.yaml")
    parser.add_argument("--finetuned_config", type=str, default="configs/inference.yaml")
    parser.add_argument("--eval_csv", type=str)
    parser.add_argument("--report_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--device", type=int, help="GPU device ID (0, 1, 2, ...)")
    args = parser.parse_args()

    from inference import compare_and_save_reports, infer_model_key, load_yaml_config

    # Report device preference (actual device is set in inference.py via config)
    if args.device is None:
        base_config_preview = load_yaml_config(args.base_config)
        device_id = base_config_preview.get("device")
        if device_id is not None:
            print(f"[EVALUATE] Using GPU device: {device_id}")
        else:
            print("[EVALUATE] Using default GPU device")
    else:
        print(f"[EVALUATE] Using GPU device: {args.device} (from command line)")

    if args.mode == "both":
        base_config = apply_overrides(load_yaml_config(args.base_config), args)
        finetune_config = apply_overrides(load_yaml_config(args.finetuned_config), args)
        base_summary = evaluate_one(
            base_config,
            os.path.join("result", "evaluate_base"),
            header_title="===== Base Model =====",
        )
        finetune_summary = evaluate_one(
            finetune_config,
            os.path.join("result", "evaluate_finetune"),
            header_title="===== Fine-tuned Model =====",
        )
        both_report_dir = os.path.join("result", "evaluate_both")
        os.makedirs(both_report_dir, exist_ok=True)
        comparison = compare_and_save_reports(
            base_summary=base_summary,
            finetune_summary=finetune_summary,
            report_dir=both_report_dir,
        )
        compare_csv_path = os.path.join(both_report_dir, "compare_base_finetune.csv")
        with open(os.path.join(both_report_dir, "evaluate_base_finetune.txt"), "w", encoding="utf-8") as f:
            f.write("===== Base Model =====\n")
            f.write(f"Accuracy:  {base_summary['metrics'].get('accuracy')}\n")
            f.write(f"Precision: {base_summary['metrics'].get('precision')}\n")
            f.write(f"Recall:    {base_summary['metrics'].get('recall')}\n")
            f.write(f"F1:        {base_summary['metrics'].get('f1')}\n\n")
            with open(base_summary["artifacts"]["inf_test_txt"], "r", encoding="utf-8") as base_f:
                f.write(base_f.read().rstrip())
            f.write("\n\n===== Fine-tuned Model =====\n")
            f.write(f"Accuracy:  {finetune_summary['metrics'].get('accuracy')}\n")
            f.write(f"Precision: {finetune_summary['metrics'].get('precision')}\n")
            f.write(f"Recall:    {finetune_summary['metrics'].get('recall')}\n")
            f.write(f"F1:        {finetune_summary['metrics'].get('f1')}\n\n")
            with open(finetune_summary["artifacts"]["inf_test_txt"], "r", encoding="utf-8") as fine_f:
                f.write(fine_f.read().rstrip())
        pd.DataFrame(comparison["rows"]).to_csv(compare_csv_path, index=False)
        print("Comparison result:", os.path.join(both_report_dir, "evaluate_base_finetune.txt"))
        print("Comparison metric CSV:", compare_csv_path)
        return

    if args.mode == "base":
        config_path = args.config or args.base_config
        default_report_dir = os.path.join("result", "evaluate_base")
    elif args.mode == "finetuned":
        config_path = args.config or args.finetuned_config
        default_report_dir = os.path.join("result", "evaluate_finetune")
    else:
        config_path = args.config or args.finetuned_config
        config_preview = load_yaml_config(config_path)
        model_dir = config_preview.get("model_name_or_path") or config_preview.get("finetuned_model_name_or_path", "")
        default_report_dir = os.path.join("result", f"evaluate_{infer_model_key(model_dir)}")

    config = apply_overrides(load_yaml_config(config_path), args)
    evaluate_one(config, args.report_dir or default_report_dir)


if __name__ == "__main__":
    main()
