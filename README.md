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

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Base Qwen2.5-0.5B-Instruct | ... | ... | ... | ... |
| Fine-tuned Unsloth LoRA model | ... | ... | ... | ... |

## Video Demonstration

Add the Google Drive video link here after recording the demo:

- Video link: ...

The video should show running inference, at least one input message, the predicted intent label, and the final test accuracy.
