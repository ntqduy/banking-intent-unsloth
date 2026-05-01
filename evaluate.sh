MODE=${1:-finetuned}

python scripts/evaluate.py --mode "$MODE"
