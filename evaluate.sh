MODE=${1:-finetuned}

if [ "$MODE" = "base" ]; then
  CONFIG=configs/inference_base.yaml
else
  CONFIG=configs/inference.yaml
fi

python scripts/evaluate.py --config "$CONFIG"
