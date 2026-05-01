import argparse
import os

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


def evaluate_one(config: dict, report_dir: str):
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
    print_performance(summary["metrics"], f"{model_name} ({model_type}) PERFORMANCE")
    print("Evaluation log:", summary["artifacts"]["eval_full_pipeline_txt"])
    print("Metric CSV:", summary["artifacts"]["metric_csv"])
    print("Predictions CSV:", summary["artifacts"]["predictions_csv"])

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

    # Load base config to get device if not overridden
    if args.device is None:
        base_config_preview = load_yaml_config(args.base_config)
        device_id = base_config_preview.get("device")
        if device_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    if args.mode == "both":
        base_config = apply_overrides(load_yaml_config(args.base_config), args)
        finetune_config = apply_overrides(load_yaml_config(args.finetuned_config), args)
        base_summary = evaluate_one(base_config, os.path.join("result", "evaluate_base"))
        finetune_summary = evaluate_one(finetune_config, os.path.join("result", "evaluate_finetune"))
        comparison = compare_and_save_reports(
            base_summary=base_summary,
            finetune_summary=finetune_summary,
            report_dir=os.path.join("result", "eval_base_finetune"),
        )
        print("Comparison result:", comparison["result_txt"])
        print("Comparison metric CSV:", comparison["metric_csv"])
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
