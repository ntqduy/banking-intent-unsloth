# BANKING77 Intent Detection with Unsloth

This project fine-tunes an intent detection model on BANKING77 using Unsloth LoRA/QLoRA. The model is trained as an instruction-style classifier: given a banking customer message and the allowed intent labels, it generates exactly one intent label.

## Project Structure

```text
banking-intent-unsloth/
|-- scripts/
|   |-- preprocess_data.py
|   |-- train.py
|   |-- inference.py
|   |-- evaluate.py
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
|-- evaluate.sh
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

The dataset is BANKING77. By default, preprocessing keeps the full original training split and creates a stratified train/validation split.

```bash
python scripts/preprocess_data.py
```

Useful options:

```bash
python scripts/preprocess_data.py --samples_per_label 0
python scripts/preprocess_data.py --samples_per_label 40 --val_size 0.1 --seed 42
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

Preprocessing includes lowercasing, whitespace cleanup, label mapping, optional per-label sampling from the original train split, and stratified train/validation splitting. The test split is the original BANKING77 test split after the same text normalization. Use `--samples_per_label 0` to train on all available original train samples.

## Fine-Tuning with Unsloth

Training configuration is in `configs/train.yaml`.

Default model:

- `unsloth/Qwen2.5-7B`

Sample count from `sample_data/original` and the default preprocessing configuration:

| Split | Number of samples | Note |
|---|---:|---|
| Original train | 10,003 | Raw BANKING77 train samples from `sample_data/original/train.csv` |
| Original test | 3,080 | Raw BANKING77 test samples from `sample_data/original/test.csv` |
| Original total | 13,083 | Original train + original test |
| Number of intent labels | 77 | From `sample_data/original/categories.json` |
| Train source after preprocessing | 10,003 | Full original train split because `samples_per_label=0` |
| Fine-tuning train split | 9,002 | 90% of the original train source after stratified split |
| Validation split | 1,001 | 10% of the original train source after stratified split |
| Test split | 3,080 | Original BANKING77 test split, used only for final evaluation |

Only the `9,002` fine-tuning train samples are used to update the LoRA weights. The validation split is used for evaluation during training, and the original test split is kept for final reporting.

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
| warmup steps | 169 |
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

Run a single example with the base model:

```bash
python scripts/inference.py --config configs/inference_base.yaml --message "my card has not arrived yet"
```

Shell helper:

```bash
bash inference.sh finetuned "my card has not arrived yet"
bash inference.sh base "my card has not arrived yet"
```

## Evaluation

Use `scripts/evaluate.py` when you want to evaluate a model on a CSV split and save the full report without running single-message inference.

Evaluate the fine-tuned LoRA checkpoint:

```bash
python scripts/evaluate.py --config configs/inference.yaml
```

Evaluate the base Qwen2.5-7B model:

```bash
python scripts/evaluate.py --config configs/inference_base.yaml
```

Shell helper:

```bash
bash evaluate.sh finetuned
bash evaluate.sh base
```

Optional arguments:

```bash
python scripts/evaluate.py \
  --config configs/inference.yaml \
  --eval_csv sample_data/clean_data/test.csv \
  --report_dir outputs/outputs_eval_custom \
  --batch_size 4
```

Evaluation outputs are saved to the configured `report_dir`. The default fine-tuned report directory is `outputs/outputs_inf_finetune/`, and the default base-model report directory is `outputs/outputs_inf_base/`.

Each evaluation report contains:

- `metrics.csv`
- `predictions.csv`
- `correct_samples.csv`
- `wrong_samples.csv`
- `sample_predictions.csv`
- `summary.json`

## Metrics

The training and inference reports save the following metrics to `metrics.csv`, `metrics.json`, or `summary.json`.

Classification metrics:

- Accuracy:

```text
Accuracy = number of correct predictions / total number of samples
```

- Precision, recall, and F1 are computed with weighted averaging across intent labels:

```text
Precision_c = TP_c / (TP_c + FP_c)
Recall_c = TP_c / (TP_c + FN_c)
F1_c = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)

Weighted Precision = sum_c support_c * Precision_c / sum_c support_c
Weighted Recall    = sum_c support_c * Recall_c / sum_c support_c
Weighted F1        = sum_c support_c * F1_c / sum_c support_c
```

Inference speed metrics:

- Inference time:

```text
Inference time = finish_time - start_time
```

- Average inference time per sample:

```text
Average inference time per sample = total inference time / number of samples
```

- FPS, also reported as throughput in samples/second:

```text
FPS = number of samples / total inference time
```

- Generated tokens per second:

```text
Generated tokens per second = number of generated tokens / total inference time
```

Computational cost metrics:

- Processed token count:

```text
Processed tokens = input tokens + generated tokens
```

- Estimated FLOPs:

```text
Estimated FLOPs = 2 * number of model parameters * processed tokens
```

- Estimated FLOPs per second:

```text
Estimated FLOPs/s = estimated FLOPs / total inference time
```

The FLOPs value is an approximation for decoder-only LLM inference. It is useful for comparing runs in this project, but it is not a hardware-profiler measurement.

## Result Table

Fill this table after running training and inference:

| Model | Accuracy | Precision | Recall | F1 | Inference Time | Estimated FLOPs |
|---|---:|---:|---:|---:|---:|---:|
| Base Qwen2.5-7B | ... | ... | ... | ... | ... | ... |
| Fine-tuned Unsloth LoRA model | ... | ... | ... | ... | ... | ... |

## Video Demonstration

Add the Google Drive video link here after recording the demo:

- Video link: ...

The video should show running inference, at least one input message, the predicted intent label, and the final test accuracy.
