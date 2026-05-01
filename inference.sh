MODE=${1:-finetuned}
MESSAGE=${2:-}

if [ -z "$MESSAGE" ]; then
  python scripts/inference.py --mode "$MODE"
else
  python scripts/inference.py --mode "$MODE" --message "$MESSAGE"
fi
