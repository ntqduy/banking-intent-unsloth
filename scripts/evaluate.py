import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/inference.yaml")
    parser.add_argument("--eval_csv", type=str)
    parser.add_argument("--report_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()

    from inference import (
        IntentClassification,
        ensure_dir,
        evaluate_and_save_report,
        load_eval_split,
        load_yaml_config,
    )

    config = load_yaml_config(args.config)
    if args.eval_csv:
        config["eval_csv"] = args.eval_csv
    if args.report_dir:
        config["report_dir"] = args.report_dir
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    eval_csv = config.get("eval_csv")
    if not eval_csv:
        parser.error("Set eval_csv in the config or pass --eval_csv.")

    classifier = IntentClassification(config)
    report_dir = ensure_dir(config.get("report_dir", "outputs/outputs_eval"))
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
