import argparse
import json

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def normalize_text(text: str) -> str:
    text = str(text).strip().lower().replace("\n", " ")
    text = " ".join(text.split())
    return text


class IntentClassification:
    def __init__(self, model_path):
        with open(model_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.model_dir = config["model_dir"]
        self.id2label_path = config["id2label_path"]
        self.max_length = int(config.get("max_length", 64))

        with open(self.id2label_path, "r", encoding="utf-8") as f:
            id2label_raw = json.load(f)

        self.id2label = {int(k): v for k, v in id2label_raw.items()}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, message):
        message = normalize_text(message)

        inputs = self.tokenizer(
            message,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=-1).item()

        predicted_label = self.id2label[pred_id]
        return predicted_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--message", type=str, required=True)
    args = parser.parse_args()

    classifier = IntentClassification(args.config)
    predicted_label = classifier(args.message)

    print("Input message:", args.message)
    print("Predicted label:", predicted_label)


if __name__ == "__main__":
    main()
