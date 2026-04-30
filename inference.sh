MODE=${1:-finetuned}
MESSAGE=${2:-}

if [ -z "$MESSAGE" ]; then
  echo "Usage: bash inference.sh [finetuned|base] \"message\""
  echo "For full evaluation, use: bash evaluate.sh [finetuned|base]"
  exit 1
fi

if [ "$MODE" = "base" ]; then
  CONFIG=configs/inference_base.yaml
else
  CONFIG=configs/inference.yaml
fi

python scripts/inference.py --config "$CONFIG" --message "$MESSAGE"
