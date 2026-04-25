# BANKING77 Intent Detection with Unsloth

This project fine-tunes an intent detection model on a sampled subset of BANKING77 using Unsloth LoRA/QLoRA. The model is trained as an instruction-style classifier: given a banking customer message and the allowed intent labels, it generates exactly one intent label.

## Project Structure

```text
banking-intent-unsloth/
|-- scripts/
|   |-- preprocess_data.py
|   |-- train.py
|   |-- inference.py
|-- configs/
|   |-- train.yaml
|   |-- inference.yaml
|   |-- inference_base.yaml
|-- sample_data/
|   |-- original/
|   |   |-- train.csv
|   |   |-- test.csv
|   |-- clean_data/
|   |   |-- train.csv
|   |   |-- val.csv
|   |   |-- test.csv
|-- outputs/
|   |-- outputs_train/
|   |   |-- analysis_data/
|   |   |-- model_checkpoint/
|   |-- outputs_inf_finetune/
|   |-- outputs_inf_base/
|-- train.sh
|-- inference.sh
|-- requirements.txt
|-- README.md
```

## Environment Setup

Unsloth is best run on Linux, Google Colab, or Kaggle with a CUDA GPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, use WSL2 or run the notebook/script on Colab/Kaggle if Unsloth or `bitsandbytes` installation fails.

## Data Preparation

The dataset is BANKING77. By default, preprocessing samples at most 40 examples per intent label so training can finish on limited GPU resources.

```bash
python scripts/preprocess_data.py
```

Useful options:

```bash
python scripts/preprocess_data.py --samples_per_label 40 --val_size 0.1 --seed 42
python scripts/preprocess_data.py --samples_per_label 0
```

Original raw files are kept in `sample_data/original` and are not rewritten when cached files already exist. Model-ready data is saved only under `sample_data/clean_data`.

Generated model data:

- `sample_data/clean_data/train.csv`
- `sample_data/clean_data/val.csv`
- `sample_data/clean_data/test.csv`
- `outputs/outputs_train/analysis_data/label2id.json`
- `outputs/outputs_train/analysis_data/id2label.json`
- `outputs/outputs_train/analysis_data/dataset_statistics.json`
- `outputs/outputs_train/analysis_data/category_statistics.csv`

Preprocessing includes lowercasing, whitespace cleanup, label mapping, optional per-label sampling from the original train split, and stratified train/validation splitting. The test split is the original BANKING77 test split after the same text normalization.

## Fine-Tuning with Unsloth

Training configuration is in `configs/train.yaml`.

Default model:

- `unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit`

Main hyperparameters:

| Hyperparameter | Value |
|---|---:|
| max sequence length | 512 |
| batch size | 2 |
| gradient accumulation steps | 8 |
| effective batch size | 16 |
| learning rate | 2e-4 |
| optimizer | adamw_8bit |
| epochs | 3 |
| warmup ratio | 0.1 |
| weight decay | 0.01 |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.0 |
| quantization | 4-bit QLoRA |

Train:

```bash
python scripts/train.py --config configs/train.yaml
```

Or run the full pipeline:

```bash
bash train.sh
```

Saved artifacts:

- `outputs/outputs_train/model_checkpoint/` LoRA checkpoint and tokenizer
- `outputs/outputs_train/model_checkpoint/label2id.json`
- `outputs/outputs_train/model_checkpoint/id2label.json`
- `outputs/outputs_train/metrics.csv`
- `outputs/outputs_train/metrics.json`
- `outputs/outputs_train/train_log_history.csv`
- `outputs/outputs_train/loss_curve.png`
- `outputs/outputs_train/loss_curve.pdf`
- `outputs/outputs_train/performance_metrics.png`
- `outputs/outputs_train/performance_metrics.pdf`
- `outputs/outputs_train/val_predictions.csv`
- `outputs/outputs_train/test_predictions.csv`
- `outputs/outputs_train/train_config.json`
- `outputs/outputs_train/model_params.json`
- `outputs/outputs_train/summary.json`

## Inference

The required grading interface is implemented in `scripts/inference.py`:

```python
from scripts.inference import IntentClassification

classifier = IntentClassification("configs/inference.yaml")
predicted_label = classifier("my card has not arrived yet")
print(predicted_label)
```

Run a single example:

```bash
python scripts/inference.py --config configs/inference.yaml --message "my card has not arrived yet"
```

Run test-set evaluation:

```bash
python scripts/inference.py --config configs/inference.yaml
```

Evaluate the base model before fine-tuning:

```bash
python scripts/inference.py --config configs/inference_base.yaml
```

Shell helper:

```bash
bash inference.sh finetuned "my card has not arrived yet"
bash inference.sh base "my card has not arrived yet"
```

Inference reports are saved to:

- Fine-tuned: `outputs/outputs_inf_finetune/`
- Base: `outputs/outputs_inf_base/`
- Each inference report contains `metrics.csv`, `predictions.csv`, `correct_samples.csv`, `wrong_samples.csv`, `sample_predictions.csv`, and `summary.json`.
- Metrics include accuracy, precision, recall, F1, latency, throughput in samples/second, and generated tokens/second.
- Single-message inference is saved to `single_prediction.json`.

## Result Table

Fill this table after running training and inference:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Base Qwen2.5-0.5B-Instruct | ... | ... | ... | ... |
| Fine-tuned Unsloth LoRA model | ... | ... | ... | ... |

## Video Demonstration

Add the Google Drive video link here after recording the demo:

- Video link: ...

The video should show running inference, at least one input message, the predicted intent label, and the final test accuracy.
