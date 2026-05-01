MODE=${1:-finetuned}
MESSAGE=${2:-}

if [ "$MODE" = "both" ]; then
  echo "Running inference for base then fine-tuned model..."
fi

if [ -z "$MESSAGE" ]; then
  python scripts/inference.py --mode "$MODE"
else
  python scripts/inference.py --mode "$MODE" --message "$MESSAGE"
fi
