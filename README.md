# BANKING77 Intent Detection with Unsloth

This project fine-tunes an intent detection model on BANKING77 using Unsloth LoRA/QLoRA. The model is trained as an instruction-style classifier: given a banking customer message and the allowed intent labels, it generates exactly one intent label.

- Dataset: [Hugging Face - banking77](https://huggingface.co/datasets/PolyAI/banking77)
- Unsloth guide: [Fine-tuning LLMs](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- Experiment machine: Google Colab with NVIDIA A100-SXM4-40GB

## Video Demonstration

Add the Google Drive video link here after recording the demo:

- Video link: [Video Demo](https://drive.google.com/drive/folders/1tJQd62qg8M38oRkKlMDLv70Cod-NcJ2R?usp=drive_link)

The video should show running inference, at least one input message, the predicted intent label, and the final test accuracy.

- Fine-tuned weights: [Finetune Model](https://drive.google.com/drive/folders/1tJQd62qg8M38oRkKlMDLv70Cod-NcJ2R?usp=drive_link)

## Project Structure

```text
banking-intent-unsloth/
|-- scripts/                  # Data prep, training, inference, evaluation
|   |-- preprocess_data.py    # Download + preprocess BANKING77
|   |-- train.py              # Fine-tune model with Unsloth + LoRA
|   |-- inference.py          # Inference for base/finetuned/both
|   |-- evaluate.py           # Evaluation + comparison reports
|-- configs/                  # YAML configs for train/inference
|   |-- train.yaml
|   |-- inference.yaml
|   |-- inference_base.yaml
|-- sample_data/              # Dataset storage
|   |-- original/             # Raw downloaded data
|   |-- clean_data/           # Preprocessed train/val/test splits
|-- result/                   # Outputs (logs, metrics, predictions)
|   |-- result_train/         # Training outputs and reports
|   |-- final_weight/         # Final LoRA adapter weights
|   |-- inf_base/             # Base-model inference outputs
|   |-- inf_finetune/         # Fine-tuned inference outputs
|   |-- inf_both/             # Base vs fine-tuned inference comparison
|   |-- evaluate_base/        # Base-model evaluation outputs
|   |-- evaluate_finetune/    # Fine-tuned evaluation outputs
|   |-- evaluate_both/        # Base vs fine-tuned evaluation comparison
|-- train.sh                  # Run preprocess + train
|-- inference.sh              # Inference helper
|-- evaluate.sh               # Evaluation helper
|-- requirements.txt
|-- train_inference_eval.ipynb
|-- README.md
```

## Environment Setup

Unsloth is best run on Linux, Google Colab, or Kaggle with a CUDA GPU.

```bash
git clone "https://github.com/ntqduy/banking-intent-unsloth.git"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, use WSL2 or run on Colab/Kaggle if Unsloth or `bitsandbytes` installation fails.

## Download

### Download weights
If you want to run inference without training, download the fine-tuned weights from [Finetune Model](https://drive.google.com/drive/folders/1tJQd62qg8M38oRkKlMDLv70Cod-NcJ2R?usp=drive_link) and extract them to `result/final_weight/` or:

1) Install `gdown` (if not already installed):

```bash
pip install gdown
```

2) Download the weights from Google Drive:

```bash
gdown 1JjAjd5sz3VXnSSIJKmBFcb3acmP7QNcz
```

3) Extract the adapter to the expected folder:

```bash
mkdir -p result/final_weight/final_lora_adapter
unzip final_lora_adapter.zip -d result/final_weight/
```

4) Verify files exist:

```bash
ls result/final_weight/final_lora_adapter
```

The folder should contain `adapter_model.safetensors`, `adapter_config.json`, and label mapping files.

### Download dataset
Manual download:
- Create folder: `sample_data/original/`
- Download from [Hugging Face - banking77](https://huggingface.co/datasets/PolyAI/banking77)
- Copy `train.csv`, `test.csv`, and `categories.json` into `sample_data/original/`

Auto download:
- Run preprocessing or training. Evaluation/inference will also auto-download when `eval_csv` is missing.
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

Notes:
- Raw files stay in `sample_data/original`.
- Preprocessed splits are saved under `sample_data/clean_data`.
- Evaluation/inference will auto-run preprocessing if `eval_csv` is missing.

## Fine-Tuning with Unsloth

### Training configuration is in `configs/train.yaml`.

Default model:
- `unsloth/Qwen2.5-7B`

### Raw Dataset (BANKING77)

| Split | Number of Samples | Note |
|---|---:|---|
| Original Train | 10,003 | Raw samples from the original dataset |
| Original Test | 3,080 | Raw samples from the original dataset |
| Original Total | 13,083 | Combined train + test |
| Intent Labels | 77 | Unique intent categories |

### Fine-tuning Data Preparation

| Split | Number of Samples | Note |
|---|---:|---|
| Fine-tuning Train | 9,002 | 90% of original train split |
| Validation | 1,001 | 10% of original train split |
| Test | 3,080 | Original test split (unchanged) |

### Main hyperparameters:

| Hyperparameter | Value |
|---|---:|
| max sequence length | 512 |
| batch size | 2 |
| gradient accumulation steps | 8 |
| effective batch size | 16 |
| learning rate | 2e-4 |
| optimizer | adamw_8bit |
| epochs | 4 |
| warmup steps | 169 |
| weight decay | 0.01 |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.0 |
| quantization | 4-bit QLoRA |

### Additional fine-tuning parameters:
- `max_steps`: -1 (use epoch-based training)
- `eval_batch_size`: 4
- `save_strategy`: epoch
- `save_total_limit`: 2
- `use_gradient_checkpointing`: true
- `load_in_4bit`: true
- `fp16`/`bf16`: auto
- `device`: 0

### Model Parameters

| Parameter Type | Count |
|---|---:|
| Total parameters | 5,063,120,384 |
| Trainable parameters | 40,370,176 |
| Non-trainable parameters | 5,022,750,208 |
| Trainable percent | 0.7973% |

### Train:

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

Single message:

```bash
python scripts/inference.py --mode finetuned --message "my card has not arrived yet"
python scripts/inference.py --mode base --message "my card has not arrived yet"
python scripts/inference.py --mode both --message "my card has not arrived yet"
```

Shell helper:

```bash
bash inference.sh finetuned "my card has not arrived yet"
bash inference.sh base "my card has not arrived yet"
bash inference.sh both "my card has not arrived yet"
bash inference.sh finetuned
bash inference.sh base
bash inference.sh both
```
## Evaluation

Use `scripts/evaluate.py` to evaluate models on a CSV split and save the full report.

```bash
python scripts/evaluate.py --mode finetuned
python scripts/evaluate.py --mode base
python scripts/evaluate.py --mode both
```

Shell helper:

```bash
bash evaluate.sh finetuned
bash evaluate.sh base
bash evaluate.sh both
```

Optional arguments:

```bash
python scripts/evaluate.py \
  --mode finetuned \
  --eval_csv sample_data/clean_data/test.csv \
  --report_dir result/evaluate_custom \
  --batch_size 4

python scripts/evaluate.py \
  --mode both \
  --eval_csv sample_data/clean_data/test.csv \
  --batch_size 4
```
## Metrics

Classification metrics:

```text
Accuracy = number of correct predictions / total number of samples

Precision_c = TP_c / (TP_c + FP_c)
Recall_c = TP_c / (TP_c + FN_c)
F1_c = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)

Weighted Precision = sum_c support_c * Precision_c / sum_c support_c
Weighted Recall    = sum_c support_c * Recall_c / sum_c support_c
Weighted F1        = sum_c support_c * F1_c / sum_c support_c
```

Inference speed metrics:

```text
Inference time = finish_time - start_time
Average inference time per sample = total inference time / number of samples
FPS = number of samples / total inference time
Generated tokens per second = number of generated tokens / total inference time
```

Computational cost metrics:

```text
Processed tokens = input tokens + generated tokens
Estimated FLOPs = 2 * number of model parameters * processed tokens
Estimated FLOPs/s = estimated FLOPs / total inference time
```

## Result Table
| Model | Accuracy | Precision | Recall | F1 | Inference Time (s) | Estimated FLOPs (GFLOPs) |
|---|---:|---:|---:|---:|---:|---:|
| Base Qwen2.5-7B | 0.5325 | 0.6301 | 0.5325 | 0.5087 | 404.9995 | 13023971.1983 |
| Fine-tuned Unsloth LoRA model | 0.9166 | 0.9206 | 0.9166 | 0.9172 | 355.7830 | 13029818.7933 |
