MODE=${1:-finetuned}
MESSAGE=${2:-}

if [ "$MODE" = "base" ]; then
  CONFIG=configs/inference_base.yaml
else
  CONFIG=configs/inference.yaml
fi

if [ -n "$MESSAGE" ]; then
  python scripts/inference.py --config "$CONFIG" --message "$MESSAGE"
else
  python scripts/inference.py --config "$CONFIG"
fi
